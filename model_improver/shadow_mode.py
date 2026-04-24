"""Shadow trade execution simulator and open-position manager."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from uuid import uuid4

import pandas as pd

from config.instruments import LOT_SIZES
from core.risk.capital_tracker import CapitalTracker
from core.utils.charge_calculator import calculate_charges
from model_improver.journal import Journal, TradeRecord
from model_improver.signal_handler import TradeSignal

logger = logging.getLogger(__name__)

TRAIL_MULTIPLIERS = {
    "TIGHT": 0.3,
    "MEDIUM": 0.5,
    "WIDE": 0.7,
    "VERY_WIDE": 1.0,
}

TRAIL_INTERVAL_MINUTES = {
    "5m": 5,
    "15m": 15,
    "60m": 60,
}


@dataclass
class ShadowTrade:
    trade_id: str
    signal: TradeSignal
    entry_time: datetime
    current_sl: float
    highest_premium: float


class ShadowMode:
    def __init__(self, journal: Journal, capital_tracker: CapitalTracker) -> None:
        self.journal = journal
        self.capital_tracker = capital_tracker
        self.model_version = os.getenv("MODEL_VERSION", "v1.0")
        self._open: dict[str, ShadowTrade] = {}
        self._last_trail_update: dict[str, datetime] = {}
        self._restore_open_trades()

    def _restore_open_trades(self) -> None:
        """Rebuild in-memory state from parquet on startup after a crash/restart."""
        try:
            open_df = self.journal.load_open_trades()
        except Exception as exc:
            logger.warning("Could not load open trades for restore: %s", exc)
            return

        if open_df.empty:
            return

        restored = 0
        for _, row in open_df.iterrows():
            try:
                trade_id = str(row["trade_id"])
                instrument = str(row["instrument"]).upper()
                lot_size = int(float(row.get("lot_size") or LOT_SIZES.get(instrument, 65)))
                entry_premium = float(row["entry_premium"])

                expiry_raw = pd.to_datetime(row["expiry_date"], errors="coerce")
                expiry_date: date = expiry_raw.date() if not pd.isna(expiry_raw) else datetime.utcnow().date()

                signal = TradeSignal(
                    instrument=instrument,
                    direction=int(row.get("direction", 1)),
                    option_type=str(row.get("option_type", "CE")).upper(),
                    strike=int(float(row.get("strike", 0))),
                    expiry_date=expiry_date,
                    entry_premium=entry_premium,
                    sl_price=float(row.get("sl_price", 0.0)),
                    target_price=float(row.get("target_price", 0.0)),
                    trail_bin=str(row.get("trail_bin", "MEDIUM")),
                    trail_tf=str(row.get("trail_tf", "15m")),
                    confidence=float(row.get("confidence", 0.5)),
                    direction_prob=float(row.get("direction_prob", 0.5)),
                    vix=float(row.get("vix_at_entry", 15.0)),
                    atr=float(row.get("atr_at_entry", 0.0)),
                    lot_size=lot_size,
                )

                entry_ts = pd.to_datetime(row["timestamp_entry"], errors="coerce")
                entry_time = entry_ts.to_pydatetime() if not pd.isna(entry_ts) else datetime.utcnow()

                trade = ShadowTrade(
                    trade_id=trade_id,
                    signal=signal,
                    entry_time=entry_time,
                    current_sl=float(row.get("sl_price", 0.0)),
                    highest_premium=entry_premium,
                )
                self._open[trade_id] = trade
                self._last_trail_update[trade_id] = datetime.utcnow()

                # Re-reserve margin so capital accounting stays correct
                margin = entry_premium * lot_size
                self.capital_tracker._reserved_margin.setdefault(trade_id, margin)

                restored += 1
            except Exception as exc:
                logger.warning("Failed to restore trade %s: %s", row.get("trade_id"), exc)

        if restored:
            logger.info("restore_open_trades count=%d", restored)

    def enter_trade(self, signal: TradeSignal) -> ShadowTrade | None:
        instrument = signal.instrument.upper()
        if any(trade.signal.instrument.upper() == instrument for trade in self._open.values()):
            return None

        required_margin = float(signal.entry_premium) * int(signal.lot_size)
        if self.capital_tracker.get_available_capital() < required_margin:
            return None

        trade_id = str(uuid4())
        if not self.capital_tracker.reserve_margin(trade_id, required_margin):
            return None

        now = datetime.utcnow()
        trade = ShadowTrade(
            trade_id=trade_id,
            signal=signal,
            entry_time=now,
            current_sl=float(signal.sl_price),
            highest_premium=float(signal.entry_premium),
        )
        self._open[trade_id] = trade
        self._last_trail_update[trade_id] = now

        record = TradeRecord(
            trade_id=trade_id,
            instrument=instrument,
            timestamp_entry=now,
            timestamp_exit=None,
            direction=int(signal.direction),
            strike=int(signal.strike),
            expiry_date=signal.expiry_date,
            option_type=str(signal.option_type),
            entry_premium=float(signal.entry_premium),
            exit_premium=None,
            lot_size=int(signal.lot_size),
            sl_price=float(signal.sl_price),
            target_price=float(signal.target_price),
            trail_bin=str(signal.trail_bin),
            trail_tf=str(signal.trail_tf),
            confidence=float(signal.confidence),
            direction_prob=float(signal.direction_prob),
            exit_reason=None,
            pnl_gross=None,
            pnl_net=None,
            charges=None,
            vix_at_entry=float(signal.vix),
            atr_at_entry=float(signal.atr),
            model_version=self.model_version,
        )
        self.journal.log_entry(record)
        return trade

    def tick(self, instrument: str, current_premium: float, current_time: datetime) -> list[str]:
        instrument_key = instrument.upper()
        closed_ids: list[str] = []

        for trade_id, trade in list(self._open.items()):
            if trade.signal.instrument.upper() != instrument_key:
                continue

            premium = float(current_premium)
            if premium <= trade.current_sl:
                self._close_trade(trade_id, premium, "SL_HIT", current_time)
                closed_ids.append(trade_id)
                continue
            if premium >= float(trade.signal.target_price):
                self._close_trade(trade_id, premium, "TARGET_HIT", current_time)
                closed_ids.append(trade_id)
                continue

            self._update_trailing_stop(trade_id, premium, current_time)

        return closed_ids

    def force_close_all(self, current_premiums: dict[str, float], reason: str = "EOD") -> None:
        now = datetime.utcnow()
        for trade_id, trade in list(self._open.items()):
            instrument = trade.signal.instrument.upper()
            premium = float(current_premiums.get(instrument, trade.signal.entry_premium))
            self._close_trade(trade_id, premium, reason, now)

    def open_trades(self) -> list[ShadowTrade]:
        return list(self._open.values())

    def _update_trailing_stop(self, trade_id: str, current_premium: float, current_time: datetime) -> None:
        trade = self._open.get(trade_id)
        if trade is None:
            return

        tf = str(trade.signal.trail_tf)
        interval = TRAIL_INTERVAL_MINUTES.get(tf)
        if interval is None:
            return
        last_update = self._last_trail_update.get(trade_id, trade.entry_time)
        elapsed_min = (current_time - last_update).total_seconds() / 60.0
        if elapsed_min < interval:
            return

        trade.highest_premium = max(trade.highest_premium, float(current_premium))

        multiplier = TRAIL_MULTIPLIERS.get(str(trade.signal.trail_bin).upper(), TRAIL_MULTIPLIERS["MEDIUM"])
        # Both CE and PE here are long option premium positions.
        new_sl = trade.highest_premium * (1.0 - multiplier)

        trade.current_sl = max(float(trade.current_sl), float(new_sl))
        self._last_trail_update[trade_id] = current_time

    def _close_trade(self, trade_id: str, exit_premium: float, reason: str, timestamp_exit: datetime) -> None:
        trade = self._open.pop(trade_id, None)
        self._last_trail_update.pop(trade_id, None)
        if trade is None:
            return

        self.journal.log_exit(
            trade_id=trade_id,
            exit_premium=float(exit_premium),
            exit_reason=str(reason),
            timestamp_exit=timestamp_exit,
        )

        lot_size = int(trade.signal.lot_size)
        pnl_gross = (float(exit_premium) - float(trade.signal.entry_premium)) * lot_size
        charges = calculate_charges(
            premium=float(exit_premium),
            lot_size=lot_size,
            lots=1,
            instrument=trade.signal.instrument.upper(),
            side="SELL",
        )["total"]
        pnl_net = pnl_gross - float(charges)
        self.capital_tracker.release_margin(trade_id, pnl_net)


# Backward compatibility with prior integration naming.
ShadowModeExecutor = ShadowMode
