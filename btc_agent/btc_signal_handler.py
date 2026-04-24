"""Model inference to BTCUSDT futures trade signal."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BtcTradeSignal:
    symbol: str
    direction: int
    entry_price: float
    sl_price: float
    target_price: float
    contracts: float
    confidence: float
    direction_prob: float
    atr: float
    bull_score: int = 0
    bear_score: int = 0
    setup_type: str = "trend"
    htf_trend: int = 0
    htf_tf: str = "15m"


FUTURES_TAKER_FEE_RATE = 0.0005
GST_RATE = 0.18
ROUND_TRIP_FEE = (2 * FUTURES_TAKER_FEE_RATE) * (1.0 + GST_RATE)  # 0.118% effective (taker+taker+GST)
BTC_CONTRACT   = 0.001     # minimum position increment (0.001 BTC)


class BtcSignalHandler:
    MIN_CONFIDENCE  = 0.55
    RISK_PCT        = 0.02   # risk 2% of capital per trade
    SL_ATR_MULT     = 1.5
    TP_ATR_MULT     = 3.0
    MIN_RR          = 1.5
    MAX_FEE_PCT_OF_TP = 0.30
    MAX_LEVERAGE    = 2.0    # 200% = 2x — keeps margin safe on small capital
    REVERSAL_SIZE_MULT = 0.50
    REVERSAL_MIN_CONFIDENCE = 0.60

    def __init__(self):
        model_dir = Path(__file__).resolve().parents[1] / "data" / "btc" / "models"
        model_path = model_dir / "lgbm_signal_model.pkl"
        meta_path = model_dir / "model_meta.json"

        with model_path.open("rb") as f:
            self.model = pickle.load(f)

        with meta_path.open("r") as f:
            meta = json.load(f)

        self.feature_cols = [str(c) for c in meta.get("feature_cols", [])]
        self.reverse_map = {int(k): int(v) for k, v in (meta.get("reverse_map") or {}).items()}
        self.ema_fast = int(meta.get("ema_fast", 8) or 8)
        self.ema_slow = int(meta.get("ema_slow", 21) or 21)
        usd_to_inr = float(os.getenv("BTC_USD_INR", "83.0"))
        self.INR_TO_USD = (1.0 / usd_to_inr) if usd_to_inr > 0 else 0.012
        self.last_rejection_reason = "NOT_EVALUATED"

    def _reject(self, reason: str) -> None:
        self.last_rejection_reason = str(reason)
        return None

    def process(self, feature_row: pd.DataFrame, current_price: float, capital_inr: float) -> BtcTradeSignal | None:
        if len(feature_row) != 1:
            raise ValueError("feature_row must be a single-row DataFrame")

        current_price = float(current_price)
        capital_inr = float(capital_inr)
        if current_price <= 0 or capital_inr <= 0:
            return self._reject("INVALID_INPUT")

        # Confluence gate first: only evaluate model if a directional setup fired.
        row = feature_row.iloc[-1]
        long_signal = int(row.get("long_signal", 0) or 0)
        short_signal = int(row.get("short_signal", 0) or 0)
        if long_signal == short_signal:
            return self._reject("NO_DIRECTIONAL_CONFLUENCE")
        direction = 1 if long_signal == 1 else -1
        bull_score = int(row.get("bull_score", 0) or 0)
        bear_score = int(row.get("bear_score", 0) or 0)
        htf_tf = "45m" if "45m_smc_trend" in feature_row.columns else "15m"
        htf_trend = int(row.get("45m_smc_trend", row.get("15m_smc_trend", 0)) or 0)
        is_reversal = (
            (direction == 1 and int(row.get("reversal_long_signal", 0) or 0) == 1)
            or (direction == -1 and int(row.get("reversal_short_signal", 0) or 0) == 1)
        )

        x = feature_row.copy()
        for col in self.feature_cols:
            if col not in x.columns:
                x[col] = 0.0
        if self.feature_cols:
            x = x[self.feature_cols]

        probs = self.model.predict_proba(x)[0]
        if len(probs) < 2:
            return self._reject("MODEL_OUTPUT_INVALID")

        # Binary model: class-1 probability = win probability.
        win_probability = float(probs[1])
        min_confidence = self.REVERSAL_MIN_CONFIDENCE if is_reversal else self.MIN_CONFIDENCE
        if win_probability < min_confidence:
            return self._reject(f"CONFIDENCE_LOW({win_probability:.3f}<{min_confidence:.2f})")

        # Prefer 15m ATR for meaningful SL/TP distances; fall back to 1m ATR.
        atr_series = feature_row.get("15m_atr_14", feature_row.get("atr_14"))
        if atr_series is None:
            return self._reject("ATR_MISSING")
        atr = float(atr_series.iloc[-1])
        if atr <= 0:
            # Hard fallback: try 1m ATR
            atr = float(feature_row["atr_14"].iloc[-1]) if "atr_14" in feature_row.columns else 0.0
        if atr <= 0:
            return self._reject("ATR_INVALID")

        sl_dist = atr * self.SL_ATR_MULT
        tp_dist = atr * self.TP_ATR_MULT
        if sl_dist <= 0:
            return self._reject("SL_DISTANCE_INVALID")

        rr = tp_dist / sl_dist
        if rr < self.MIN_RR:
            return self._reject(f"RR_TOO_LOW({rr:.2f}<{self.MIN_RR:.2f})")

        sl_distance_pct = sl_dist / current_price
        if sl_distance_pct <= 0:
            return self._reject("SL_DISTANCE_PCT_INVALID")

        # Fee viability: gross TP profit must cover at least (1 / MAX_FEE_PCT_OF_TP) x fees.
        # tp_pct = TP move as % of price; fee is 0.1% of notional.
        # Minimum tp_pct required = ROUND_TRIP_FEE / MAX_FEE_PCT_OF_TP = 0.1% / 0.30 = 0.33%
        tp_pct = tp_dist / current_price
        if tp_pct < ROUND_TRIP_FEE / self.MAX_FEE_PCT_OF_TP:
            min_tp_pct = ROUND_TRIP_FEE / self.MAX_FEE_PCT_OF_TP
            return self._reject(f"FEE_UNVIABLE(tp_pct={tp_pct:.4f}<min={min_tp_pct:.4f})")

        # Position sizing in BTC terms (0.001 BTC increments).
        # Risk X% of capital in USD; divide by SL distance in $ to get BTC size.
        capital_usd = capital_inr * self.INR_TO_USD
        risk_mult = self.REVERSAL_SIZE_MULT if is_reversal else 1.0
        risk_usd = capital_usd * self.RISK_PCT * risk_mult
        raw_btc = risk_usd / sl_dist  # BTC needed to lose exactly risk_usd on SL hit
        # Round down to nearest 0.001 BTC increment, enforce minimum.
        contracts = max(BTC_CONTRACT, int(raw_btc / BTC_CONTRACT) * BTC_CONTRACT)
        # Cap at 2x leverage: notional = contracts * price <= capital_usd * MAX_LEVERAGE.
        max_btc = (capital_usd * self.MAX_LEVERAGE) / current_price
        max_btc_rounded = int(max_btc / BTC_CONTRACT) * BTC_CONTRACT
        contracts = min(contracts, max(BTC_CONTRACT, max_btc_rounded))
        if contracts <= 0:
            return self._reject("SIZE_INVALID")

        if direction == 1:
            sl_price = current_price - sl_dist
            target_price = current_price + tp_dist
        else:
            sl_price = current_price + sl_dist
            target_price = current_price - tp_dist

        self.last_rejection_reason = "SIGNAL_READY"
        return BtcTradeSignal(
            symbol="BTCUSDT",
            direction=direction,
            entry_price=current_price,
            sl_price=float(sl_price),
            target_price=float(target_price),
            contracts=float(contracts),
            confidence=float(win_probability),
            direction_prob=float(win_probability),
            atr=float(atr),
            bull_score=bull_score,
            bear_score=bear_score,
            setup_type="reversal" if is_reversal else "trend",
            htf_trend=htf_trend,
            htf_tf=htf_tf,
        )
