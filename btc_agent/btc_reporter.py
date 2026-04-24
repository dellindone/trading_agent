"""Telegram reporter for BTC shadow agent events and summaries."""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
import pandas as pd
import pytz

from btc_agent.btc_journal import BtcJournal, BtcTradeRecord
from btc_agent.btc_signal_handler import BtcTradeSignal

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class BtcReporter:
    def __init__(self, telegram_token: str, telegram_chat_id: str):
        self.telegram_token = str(telegram_token or "")
        self.telegram_chat_id = str(telegram_chat_id or "")

    def send_engine_start_alert(self, started_at: datetime):
        local = started_at.astimezone(IST) if started_at.tzinfo else IST.localize(started_at)
        message = (
            "🟢 BTC SHADOW AGENT LIVE\n"
            f"Started: {local.strftime('%d-%b-%Y %H:%M:%S IST')}\n"
            "Symbol: BTCUSDT Perpetual\n"
            "Mode: Paper/Shadow only\n"
            "Status: Heartbeat enabled (hourly)"
        )
        self._send(message)

    def send_signal_alert(self, signal: BtcTradeSignal):
        side = "LONG" if int(signal.direction) == 1 else "SHORT"
        icon = "📈" if side == "LONG" else "📉"
        conf = int(round(float(signal.confidence) * 100))
        score = int(signal.bull_score) if int(signal.direction) == 1 else int(signal.bear_score)
        qty_btc = float(signal.contracts)
        notional_usd = qty_btc * float(signal.entry_price)
        htf_trend = int(getattr(signal, "htf_trend", 0) or 0)
        htf_tf = str(getattr(signal, "htf_tf", "15m"))
        htf_str = "bull (+1)" if htf_trend == 1 else ("bear (-1)" if htf_trend == -1 else "neutral (0)")
        reason = f"{'Bull' if int(signal.direction) == 1 else 'Bear'} confluence {score}/11"
        if str(getattr(signal, "setup_type", "trend")) == "reversal":
            reason = f"Reversal candlestick + momentum | {reason}"
        message = (
            f"{icon} BTC {side} ENTRY\n"
            f"Reason: {reason}\n"
            f"HTF Trend: {htf_tf} {htf_str}\n"
            f"Entry: ${float(signal.entry_price):,.2f} | SL: ${float(signal.sl_price):,.2f} | Target: ${float(signal.target_price):,.2f}\n"
            f"Qty: {qty_btc:,.4f} BTC (~${notional_usd:,.2f}) | Conf: {conf}%"
        )
        self._send(message)

    INR_TO_USD = 0.012

    def send_exit_alert(self, record: BtcTradeRecord, capital_inr: float = 0.0):
        side = "LONG" if int(record.direction) == 1 else "SHORT"
        entry = float(record.entry_price)
        exit_px = float(record.exit_price or 0.0)
        qty_btc = float(record.contracts or 0.0)
        exit_notional_usd = qty_btc * exit_px
        pnl_usd = float(record.pnl_usd or 0.0)
        pnl_inr = float(record.pnl_inr or 0.0)
        fees_inr = float(record.charges_usd or 0.0) / self.INR_TO_USD
        exit_icon = "🟢" if pnl_usd >= 0 else "🔴"
        usd_sign = "+" if pnl_usd >= 0 else "-"
        inr_sign = "+" if pnl_inr >= 0 else "-"

        message = (
            f"{exit_icon} BTC EXIT — {side}\n"
            f"Reason: {record.exit_reason or 'UNKNOWN'}\n"
            f"Entry: ${entry:,.2f} → Exit: ${exit_px:,.2f}\n"
            f"Qty: {qty_btc:,.4f} BTC (~${exit_notional_usd:,.2f})\n"
            f"P&L (Net): {usd_sign}${abs(pnl_usd):,.2f} ({inr_sign}₹{abs(pnl_inr):,.2f}) | Fees: ₹{fees_inr:,.2f}\n"
            f"Balance: ₹{float(capital_inr):,.2f}"
        )
        self._send(message)

    def send_hourly_live_summary(self, *, btc_price, open_trades, signals_today, capital_inr, uptime_minutes):
        message = (
            "💓 BTC SHADOW HEARTBEAT\n"
            f"BTC: ${float(btc_price):,.0f} | Open: {int(open_trades)} | Signals: {int(signals_today)}\n"
            f"Capital: ₹{float(capital_inr):,.0f} | Uptime: {int(uptime_minutes)}min\n"
            "Proof: Agent loop active and polling Delta Exchange."
        )
        self._send(message)

    def send_daily_summary(self, journal: BtcJournal, capital_inr: float):
        now = datetime.now(IST)
        df = journal.load_all()

        if df.empty:
            message = (
                f"📊 BTC DAILY SUMMARY — {now.strftime('%d %b %Y')}\n"
                "Trades: 0 | Wins: 0 | Losses: 0 | Win Rate: 0%\n"
                "Gross P&L: $0.00 (₹0.00)\n"
                f"Capital: ₹{float(capital_inr):,.0f}"
            )
            self._send(message)
            return

        entry_ts = pd.to_datetime(df["timestamp_entry"], errors="coerce", utc=True)
        local_dates = entry_ts.dt.tz_convert(IST).dt.date
        today_df = df.loc[local_dates == now.date()].copy()
        closed = today_df[today_df["timestamp_exit"].notna()].copy()

        total = int(len(closed))
        if total == 0:
            wins = 0
            losses = 0
            win_rate = 0.0
            gross_usd = 0.0
            gross_inr = 0.0
        else:
            pnl_usd = pd.to_numeric(closed["pnl_usd"], errors="coerce").fillna(0.0)
            pnl_inr = pd.to_numeric(closed["pnl_inr"], errors="coerce").fillna(0.0)
            gross_usd = float(pnl_usd.sum())
            gross_inr = float(pnl_inr.sum())
            wins = int((pnl_usd > 0).sum())
            losses = int((pnl_usd <= 0).sum())
            win_rate = (wins / total) * 100.0 if total else 0.0

        usd_sign = "+" if gross_usd >= 0 else "-"
        inr_sign = "+" if gross_inr >= 0 else "-"

        message = (
            f"📊 BTC DAILY SUMMARY — {now.strftime('%d %b %Y')}\n"
            f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.0f}%\n"
            f"Gross P&L: {usd_sign}${abs(gross_usd):,.2f} ({inr_sign}₹{abs(gross_inr):,.2f})\n"
            f"Capital: ₹{float(capital_inr):,.0f}"
        )
        self._send(message)

    def _send(self, message: str):
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured; skipping send.")
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
