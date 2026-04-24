"""Position sizing and SL conversion rules for shadow/live readiness."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

ATR_MULTIPLIERS = {
    "TIGHT": 0.5,
    "MEDIUM": 1.0,
    "WIDE": 1.5,
    "VERY_WIDE": 2.0,
}

ABS_SL_MIN = 5.0
ABS_SL_MAX = 60.0
VIX_NO_NEW_TRADES = 30.0
VIX_HIGH_MIN_SL = 35.0
VIX_LOW_MAX_SL = 20.0
INTRADAY_SPIKE_THRESHOLD = 0.05
INTRADAY_SPIKE_WIDEN_MULTIPLIER = 1.5


def should_block_new_trades(vix: float) -> bool:
    return float(vix) > VIX_NO_NEW_TRADES


def _clamp_sl(value: float) -> float:
    return max(ABS_SL_MIN, min(ABS_SL_MAX, float(value)))


def stop_loss_from_bin(sl_bin: str, atr_14: float, vix: float) -> float:
    atr_base = max(0.0, float(atr_14))
    multiplier = ATR_MULTIPLIERS.get(str(sl_bin).upper(), ATR_MULTIPLIERS["MEDIUM"])
    sl_rupees = atr_base * multiplier
    if vix > 25:
        sl_rupees = max(sl_rupees, VIX_HIGH_MIN_SL)
    if vix < 12:
        sl_rupees = min(sl_rupees, VIX_LOW_MAX_SL)
    return _clamp_sl(sl_rupees)


def widened_stop_loss(current_sl_points: float, vix_pct_change_1: float) -> float:
    if float(vix_pct_change_1) > INTRADAY_SPIKE_THRESHOLD:
        return _clamp_sl(float(current_sl_points) * INTRADAY_SPIKE_WIDEN_MULTIPLIER)
    return _clamp_sl(current_sl_points)


class PositionSizer:
    """Phase-aware lot sizing.

    Phase 1 (shadow): always 1 lot.
    Phase 2 (future live): risk-based sizing, capped to 2% risk per trade.
    """

    def __init__(self, shadow_mode: bool = True) -> None:
        self.shadow_mode = bool(shadow_mode)

    def get_lots(self, capital: float, sl_per_unit: float, lot_size: int, vix: float) -> int:
        if self.shadow_mode:
            return 1

        sl = max(float(sl_per_unit), 1e-9)
        lot = max(int(lot_size), 1)
        risk_budget = max(float(capital), 0.0) * 0.02
        lots = math.floor(risk_budget / (sl * lot))
        lots = max(1, min(int(lots), 5))
        return lots

    def get_margin_required(self, premium: float, lot_size: int, lots: int) -> float:
        return float(premium) * int(lot_size) * int(lots) * 1.10


position_sizer = PositionSizer()

