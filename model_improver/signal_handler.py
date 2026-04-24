"""Signal conversion layer from model predictions to executable shadow trades."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "trading_agent"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from config.instruments import LOT_SIZES
from core.data.option_chain import option_chain_service
from core.model.predict import ModelPrediction
from core.risk.position_sizer import stop_loss_from_bin
from core.strategy.strike_selector import strike_selector

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    instrument: str
    direction: int
    option_type: str
    strike: int
    expiry_date: date
    entry_premium: float
    sl_price: float
    target_price: float
    trail_bin: str
    trail_tf: str
    confidence: float
    direction_prob: float
    vix: float
    atr: float
    lot_size: int


class SignalHandler:
    MIN_CONFIDENCE = 0.60
    MIN_RR = 1.5
    SHADOW_LOTS = 1

    def process(
        self,
        prediction: ModelPrediction,
        feature_row: pd.DataFrame,
        instrument: str,
    ) -> TradeSignal | None:
        instrument_key = instrument.upper()
        if len(feature_row) != 1:
            raise ValueError("feature_row must be a single-row DataFrame")

        row = feature_row.iloc[-1]
        vix = float(row.get("vix", 0.0))
        atr = float(row.get("atr_14", 0.0))
        if vix > 30.0:
            return None
        if prediction.confidence < self.MIN_CONFIDENCE:
            return None
        if atr <= 0:
            return None

        lot_size = LOT_SIZES.get(instrument_key)
        if lot_size is None or lot_size <= 0:
            return None

        sl_price = float(stop_loss_from_bin(prediction.sl_bin, atr, vix))
        # phase1_target and sl_price are both in index points (ATR-based).
        # Compare them directly for the RR gate — no lot-size division here.
        target_price = float(prediction.phase1_target)
        if target_price <= 0 or sl_price <= 0:
            return None

        rr = target_price / sl_price
        if rr < self.MIN_RR:
            return None

        direction_label = "BULLISH" if prediction.direction == 1 else "BEARISH"
        option_type = "CE" if prediction.direction == 1 else "PE"
        direction = 1 if option_type == "CE" else 0
        side_prob = float(prediction.direction_prob if prediction.direction == 1 else (1.0 - prediction.direction_prob))

        chain = option_chain_service.get_best_instrument(instrument_key, direction_label)
        if not chain or not chain.get("processed"):
            return None
        selected = strike_selector.select(
            chain["processed"],
            chain.get("atm", float(row.get("close", 0.0))),
            direction_label,
        )
        if not selected:
            return None

        entry_premium = float(selected.get("lp", 0.0))
        strike = int(float(selected.get("strike", 0.0)))
        expiry = chain.get("expiry")
        if entry_premium <= 0 or strike <= 0 or expiry is None:
            return None

        return TradeSignal(
            instrument=instrument_key,
            direction=direction,
            option_type=option_type,
            strike=strike,
            expiry_date=expiry,
            entry_premium=entry_premium,
            sl_price=sl_price,
            target_price=target_price,
            trail_bin=str(prediction.trail_bin),
            trail_tf=str(prediction.trail_tf),
            confidence=float(prediction.confidence),
            direction_prob=side_prob,
            vix=vix,
            atr=atr,
            lot_size=int(lot_size),
        )
