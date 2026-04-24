"""Telegram reporter for shadow trading signals, exits, and daily summaries."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "trading_agent"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from model_improver.journal import Journal, TradeRecord
from model_improver.signal_handler import TradeSignal
from core.risk.capital_tracker import CapitalTracker

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class Reporter:
    def __init__(
        self,
        journal: Journal,
        capital_tracker: CapitalTracker,
        telegram_token: str,
        telegram_chat_id: str,
    ) -> None:
        self.journal = journal
        self.capital_tracker = capital_tracker
        self.telegram_token = str(telegram_token or "")
        self.telegram_chat_id = str(telegram_chat_id or "")

    def send_signal_alert(self, signal: TradeSignal) -> None:
        direction_label = "CE" if signal.direction == 1 else "PE"
        confidence_pct = int(round(float(signal.confidence) * 100))
        message = (
            f"🟢 SHADOW SIGNAL — {signal.instrument} {direction_label}\n"
            f"Strike: {int(signal.strike)} | Expiry: {signal.expiry_date.strftime('%d-%b-%y')}\n"
            f"Entry: ₹{signal.entry_premium:.2f} | SL: ₹{signal.sl_price:.2f} | Target: ₹{signal.target_price:.2f}\n"
            f"Confidence: {confidence_pct}% | VIX: {signal.vix:.1f}"
        )
        self._send(message)

    def send_exit_alert(self, record: TradeRecord) -> None:
        direction_label = "CE" if int(record.direction) == 1 else "PE"
        pnl_net = float(record.pnl_net or 0.0)
        charges = float(record.charges or 0.0)
        sign = "+" if pnl_net >= 0 else "-"
        message = (
            f"🔴 SHADOW EXIT — {record.instrument} {direction_label}\n"
            f"Exit reason: {record.exit_reason or 'UNKNOWN'}\n"
            f"P&L: {sign}₹{abs(pnl_net):.2f} (net) | Charges: ₹{charges:.2f}"
        )
        self._send(message)

    def send_daily_summary(self) -> None:
        df = self.journal.load_all()
        now_ist = datetime.now(IST)
        if df.empty:
            message = (
                f"📊 DAILY SHADOW SUMMARY — {now_ist.strftime('%d %b %Y')}\n"
                "Trades: 0 | Wins: 0 | Losses: 0 | Win Rate: 0%\n"
                "Gross P&L: ₹0.00 | Net P&L: ₹0.00\n"
                f"Capital: ₹{self.capital_tracker.current_capital:,.2f}"
            )
            self._send(message)
            return

        timestamp_col = pd.to_datetime(df["timestamp_entry"], errors="coerce", utc=True)
        local_dates = timestamp_col.dt.tz_convert(IST).dt.date
        today = now_ist.date()
        today_df = df.loc[local_dates == today].copy()

        if today_df.empty:
            message = (
                f"📊 DAILY SHADOW SUMMARY — {now_ist.strftime('%d %b %Y')}\n"
                "Trades: 0 | Wins: 0 | Losses: 0 | Win Rate: 0%\n"
                "Gross P&L: ₹0.00 | Net P&L: ₹0.00\n"
                f"Capital: ₹{self.capital_tracker.current_capital:,.2f}"
            )
            self._send(message)
            return

        closed = today_df[today_df["timestamp_exit"].notna()].copy()
        total_trades = int(len(today_df))
        if closed.empty:
            wins = 0
            losses = 0
            win_rate = 0.0
            gross_pnl = 0.0
            net_pnl = 0.0
        else:
            gross_series = pd.to_numeric(closed["pnl_gross"], errors="coerce").fillna(0.0)
            net_series = pd.to_numeric(closed["pnl_net"], errors="coerce").fillna(0.0)
            gross_pnl = float(gross_series.sum())
            net_pnl = float(net_series.sum())
            wins = int((net_series > 0).sum())
            losses = int((net_series <= 0).sum())
            win_rate = (wins / len(closed)) if len(closed) else 0.0

        gross_sign = "+" if gross_pnl >= 0 else "-"
        net_sign = "+" if net_pnl >= 0 else "-"
        message = (
            f"📊 DAILY SHADOW SUMMARY — {now_ist.strftime('%d %b %Y')}\n"
            f"Trades: {total_trades} | Wins: {wins} | Losses: {losses} | Win Rate: {int(round(win_rate * 100))}%\n"
            f"Gross P&L: {gross_sign}₹{abs(gross_pnl):,.2f} | Net P&L: {net_sign}₹{abs(net_pnl):,.2f}\n"
            f"Capital: ₹{self.capital_tracker.current_capital:,.2f}"
        )
        self._send(message)

    def send_engine_start_alert(self, instrument: str, started_at_ist: datetime) -> None:
        message = (
            f"🟢 {instrument} SHADOW AGENT LIVE\n"
            f"Started: {started_at_ist.strftime('%d-%b-%Y %H:%M:%S IST')}\n"
            "Mode: Paper/Shadow only\n"
            "Status: Heartbeat enabled (hourly)"
        )
        self._send(message)

    def send_hourly_live_summary(
        self,
        *,
        instrument: str,
        open_trades: int,
        total_signals_today: int,
        capital: float,
        uptime_minutes: int,
        index_price: float = 0.0,
    ) -> None:
        price_str = f"₹{index_price:,.0f}" if index_price > 0 else "N/A"
        message = (
            f"💓 {instrument} SHADOW HEARTBEAT\n"
            f"{instrument}: {price_str} | Open: {open_trades} | Signals: {total_signals_today}\n"
            f"Capital: ₹{capital:,.0f} | Uptime: {uptime_minutes}min\n"
            "Proof: Agent loop active and polling market data."
        )
        self._send(message)

    def _send(self, message: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured; skipping send.")
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
