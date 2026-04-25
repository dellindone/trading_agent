"""BTC futures shadow executor (paper trading)."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "trading_agent"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.risk.capital_tracker import CapitalTracker

from btc_agent.btc_journal import BtcJournal, BtcTradeRecord
from btc_agent.btc_signal_handler import BtcTradeSignal

logger = logging.getLogger(__name__)

FUTURES_TAKER_FEE_RATE = 0.0005
FUTURES_MAKER_FEE_RATE = 0.0002
GST_RATE = 0.18
USD_TO_INR = float(os.getenv("BTC_USD_INR", "83.0"))
INR_TO_USD = 1.0 / USD_TO_INR if USD_TO_INR > 0 else 0.012
ENTRY_FEE_MODE = str(os.getenv("BTC_ENTRY_FEE_MODE", "taker")).strip().lower()
EXIT_FEE_MODE = str(os.getenv("BTC_EXIT_FEE_MODE", "taker")).strip().lower()
COOLDOWN_MINUTES = 5           # no re-entry for 5 min after a TIMEOUT exit
TRAIL_ACTIVATION_ATR = 1.0
TRAIL_STEP_ATR = 1.0
MARGIN_PCT = 0.10


@dataclass
class BtcShadowTrade:
    trade_id: str
    signal: BtcTradeSignal
    entry_time: datetime
    current_sl: float
    best_price: float
    bars_held: int


class BtcShadowMode:
    def __init__(self, journal: BtcJournal, capital_tracker: CapitalTracker):
        self.journal = journal
        self.capital_tracker = capital_tracker
        self.model_version = "v1.0"
        self._open: dict[str, BtcShadowTrade] = {}
        self._last_timeout_exit: datetime | None = None
        self._last_timeout_direction: int = 0
        self._restore_open_trades()

    def has_open_trades(self) -> bool:
        return bool(self._open)

    def open_trade_ids(self) -> list[str]:
        return list(self._open.keys())

    def open_trade_snapshots(self) -> list[tuple[str, int, float]]:
        return [
            (str(trade.trade_id), int(trade.signal.direction), float(trade.current_sl))
            for trade in self._open.values()
        ]

    def open_trade_display_snapshots(self) -> list[dict]:
        """Lightweight snapshot for the live display — no locks needed (display-only)."""
        return [
            {
                "direction": int(t.signal.direction),
                "entry_price": float(t.signal.entry_price),
                "current_sl": float(t.current_sl),
                "target_price": float(t.signal.target_price),
                "contracts": float(t.signal.contracts),
                "entry_time": t.entry_time,
            }
            for t in self._open.values()
        ]

    def get_any_open_atr(self) -> float:
        for trade in self._open.values():
            atr = float(getattr(trade.signal, "atr", 0.0) or 0.0)
            if atr > 0:
                return atr
        return 0.0

    def _restore_open_trades(self) -> None:
        open_df = self.journal.load_open_trades()
        if open_df.empty:
            return

        for _, row in open_df.iterrows():
            trade_id = str(row["trade_id"])
            direction = int(row["direction"])
            entry_price = float(row["entry_price"])

            signal = BtcTradeSignal(
                symbol=str(row["symbol"]),
                direction=direction,
                entry_price=entry_price,
                sl_price=float(row["sl_price"]),
                target_price=float(row["target_price"]),
                contracts=float(row["contracts"]),
                confidence=float(row.get("confidence", 0.0)),
                direction_prob=float(row.get("direction_prob", 0.0)),
                atr=float(row.get("atr_at_entry", 0.0)),
                override=bool(row.get("override", False) or False),
            )

            entry_time = datetime.utcnow()
            ts = row.get("timestamp_entry")
            if ts is not None:
                parsed = pd.to_datetime(ts, errors="coerce")
                entry_time = parsed.to_pydatetime().replace(tzinfo=None) if not pd.isna(parsed) else datetime.utcnow()

            self._open[trade_id] = BtcShadowTrade(
                trade_id=trade_id,
                signal=signal,
                entry_time=entry_time,
                current_sl=float(row["sl_price"]),
                best_price=entry_price,
                bars_held=0,
            )
            self.capital_tracker._reserved_margin.setdefault(trade_id, float(signal.contracts) * MARGIN_PCT)

    def enter_trade(self, signal: BtcTradeSignal) -> BtcShadowTrade | None:
        if self._open:
            return None

        # Cooldown: skip re-entry in the same direction shortly after a TIMEOUT.
        if self._last_timeout_exit is not None and self._last_timeout_direction == signal.direction:
            elapsed = (datetime.utcnow() - self._last_timeout_exit).total_seconds() / 60
            if elapsed < COOLDOWN_MINUTES:
                return None

        required_margin = float(signal.contracts) * MARGIN_PCT
        if self.capital_tracker.get_available_capital() < required_margin:
            return None

        trade_id = str(uuid4())
        if not self.capital_tracker.reserve_margin(trade_id, required_margin):
            return None

        now = datetime.utcnow().replace(tzinfo=None)
        trade = BtcShadowTrade(
            trade_id=trade_id,
            signal=signal,
            entry_time=now,
            current_sl=float(signal.sl_price),
            best_price=float(signal.entry_price),
            bars_held=0,
        )
        self._open[trade_id] = trade

        self.journal.log_entry(
            BtcTradeRecord(
                trade_id=trade_id,
                symbol=str(signal.symbol),
                timestamp_entry=now,
                timestamp_exit=None,
                direction=int(signal.direction),
                entry_price=float(signal.entry_price),
                exit_price=None,
                sl_price=float(signal.sl_price),
                target_price=float(signal.target_price),
                contracts=float(signal.contracts),
                confidence=float(signal.confidence),
                direction_prob=float(signal.direction_prob),
                atr_at_entry=float(signal.atr),
                exit_reason=None,
                pnl_usd=None,
                pnl_inr=None,
                charges_usd=None,
                model_version=self.model_version,
                override=bool(signal.override),
                charges_inr=None,
            )
        )
        return trade

    def tick(self, current_price: float, high: float, low: float, atr: float, current_time: datetime) -> list[str]:
        current_time = current_time.replace(tzinfo=None) if hasattr(current_time, "tzinfo") else current_time
        closed: list[str] = []
        price = float(current_price)
        high = float(high)
        low = float(low)
        atr = float(atr)

        for trade_id, trade in list(self._open.items()):
            if trade.signal.direction == 1:
                if low <= trade.current_sl:
                    self._close_trade(trade_id, trade.current_sl, "SL_HIT", current_time)
                    closed.append(trade_id)
                    continue
                if high >= float(trade.signal.target_price):
                    self._close_trade(trade_id, float(trade.signal.target_price), "TP_HIT", current_time)
                    closed.append(trade_id)
                    continue
            else:
                if high >= trade.current_sl:
                    self._close_trade(trade_id, trade.current_sl, "SL_HIT", current_time)
                    closed.append(trade_id)
                    continue
                if low <= float(trade.signal.target_price):
                    self._close_trade(trade_id, float(trade.signal.target_price), "TP_HIT", current_time)
                    closed.append(trade_id)
                    continue

            trade.bars_held += 1

            self._update_trailing_sl(trade, high=high, low=low, atr=atr)

        return closed

    def force_close_all(self, current_price: float, reason: str):
        now = datetime.utcnow()
        for trade_id in list(self._open.keys()):
            self._close_trade(trade_id, float(current_price), str(reason), now)

    def _update_trailing_sl(self, trade: BtcShadowTrade, high: float, low: float, atr: float) -> None:
        if atr <= 0:
            return

        entry = float(trade.signal.entry_price)
        if trade.signal.direction == 1:
            trade.best_price = max(trade.best_price, high)
            profit = trade.best_price - entry
            if profit < (TRAIL_ACTIVATION_ATR * atr):
                return
            new_sl = trade.best_price - (TRAIL_STEP_ATR * atr)
            if new_sl > trade.current_sl:
                old_sl = float(trade.current_sl)
                trade.current_sl = float(new_sl)
                self.journal.update_trade(trade.trade_id, {"sl_price": trade.current_sl})
                logger.info(
                    "trailing_sl_moved trade_id=%s old_sl=%.2f new_sl=%.2f best_price=%.2f",
                    trade.trade_id,
                    old_sl,
                    trade.current_sl,
                    trade.best_price,
                )
        else:
            trade.best_price = min(trade.best_price, low)
            profit = entry - trade.best_price
            if profit < (TRAIL_ACTIVATION_ATR * atr):
                return
            new_sl = trade.best_price + (TRAIL_STEP_ATR * atr)
            if new_sl < trade.current_sl:
                old_sl = float(trade.current_sl)
                trade.current_sl = float(new_sl)
                self.journal.update_trade(trade.trade_id, {"sl_price": trade.current_sl})
                logger.info(
                    "trailing_sl_moved trade_id=%s old_sl=%.2f new_sl=%.2f best_price=%.2f",
                    trade.trade_id,
                    old_sl,
                    trade.current_sl,
                    trade.best_price,
                )

    def _close_trade(self, trade_id: str, exit_price: float, reason: str, timestamp_exit: datetime) -> None:
        trade = self._open.pop(trade_id, None)
        if trade is None:
            return
        logger.info("trade_closed trade_id=%s exit_price=%.2f reason=%s", trade_id, float(exit_price), str(reason))

        pnl_usd, pnl_inr, charges_usd, charges_inr = self._compute_pnl(trade, exit_price)
        # pnl_usd and pnl_inr are already NET of fees — do not deduct again.

        self.journal.log_exit(
            trade_id=trade_id,
            exit_price=float(exit_price),
            exit_reason=str(reason),
            timestamp_exit=timestamp_exit,
        )

        self.journal.update_trade(
            trade_id,
            {
                "exit_price": float(exit_price),
                "timestamp_exit": timestamp_exit,
                "exit_reason": str(reason),
                "charges_usd": float(charges_usd),
                "charges_inr": float(charges_inr),
                "pnl_usd": float(pnl_usd),
                "pnl_inr": float(pnl_inr),
            },
        )

        self.capital_tracker.release_margin(trade_id, float(pnl_inr))

    def _compute_pnl(self, trade, exit_price) -> tuple[float, float, float, float]:
        entry_price = float(trade.signal.entry_price)
        contracts = float(trade.signal.contracts)
        direction = int(trade.signal.direction)

        gross_usd = (float(exit_price) - entry_price) * contracts * direction
        entry_notional_usd = abs(entry_price * contracts)
        exit_notional_usd = abs(float(exit_price) * contracts)

        entry_fee_rate = FUTURES_MAKER_FEE_RATE if ENTRY_FEE_MODE == "maker" else FUTURES_TAKER_FEE_RATE
        exit_fee_rate = FUTURES_MAKER_FEE_RATE if EXIT_FEE_MODE == "maker" else FUTURES_TAKER_FEE_RATE
        charges_usd = (entry_notional_usd * entry_fee_rate + exit_notional_usd * exit_fee_rate) * (1.0 + GST_RATE)
        charges_inr = charges_usd * USD_TO_INR
        net_usd = gross_usd - charges_usd
        net_inr = net_usd * USD_TO_INR
        # Both pnl_usd and pnl_inr are NET of fees; charges_usd is stored separately.
        return float(net_usd), float(net_inr), float(charges_usd), float(charges_inr)
