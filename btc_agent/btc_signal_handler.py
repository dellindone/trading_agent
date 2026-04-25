"""Model inference to BTCUSDT futures trade signal."""

from __future__ import annotations

import json
import os
import pickle
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from btc_agent.drift_monitor import DriftMonitor
from btc_agent.regime_classifier import REGIMES, classify_regime


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
    override: bool = False


FUTURES_TAKER_FEE_RATE = 0.0005
GST_RATE = 0.18
ROUND_TRIP_FEE = (2 * FUTURES_TAKER_FEE_RATE) * (1.0 + GST_RATE)  # 0.118% effective (taker+taker+GST)
BTC_CONTRACT   = 0.001     # minimum position increment (0.001 BTC)


class BtcSignalHandler:
    MIN_CONFIDENCE  = 0.55
    BASE_RISK_PCT   = 0.01   # maximum risk fraction at full confidence
    RISK_PCT        = BASE_RISK_PCT   # risk 1% of capital per trade (backward-compatible alias)
    SL_ATR_MULT     = 1.5
    TP_ATR_MULT     = 3.0
    MAX_FEE_PCT_OF_TP = 0.30
    MAX_LEVERAGE    = 10.0   # 1000% = 10x leverage cap
    REVERSAL_SIZE_MULT = 0.50
    REVERSAL_MIN_CONFIDENCE = 0.60
    MAX_CONCURRENT_POSITIONS = 1
    OVERRIDE_6H_WINDOW = timedelta(hours=6)
    OVERRIDE_6H_TRIP_R = -5.0
    OVERRIDE_6H_RELEASE_R = -2.0
    OVERRIDE_DAY_TRIP_R = -8.0
    OVERRIDE_WEEK_TRIP_R = -20.0

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
        self.MIN_CONFIDENCE = float(meta.get("best_threshold", self.MIN_CONFIDENCE) or self.MIN_CONFIDENCE)
        self._regime_models: dict[str, object] = {}
        for regime in REGIMES:
            path = model_dir / f"lgbm_signal_model_{regime}.pkl"
            if path.exists():
                with path.open("rb") as f:
                    self._regime_models[regime] = pickle.load(f)
        feature_stats = meta.get("feature_stats", {})
        self.drift_monitor = DriftMonitor(feature_stats)
        self.last_drift_alerts: list[str] = []
        self.last_regime: str = "unknown"
        usd_to_inr = float(os.getenv("BTC_USD_INR", "83.0"))
        self.INR_TO_USD = (1.0 / usd_to_inr) if usd_to_inr > 0 else 0.012
        self.last_rejection_reason = "NOT_EVALUATED"
        self.last_override_reached_model = False

        self.override_candidates_seen = 0
        self.override_blocked_6h = 0
        self.override_blocked_day = 0
        self.override_blocked_week = 0
        self.override_reached_model = 0
        self.override_rejected_model = 0
        self.override_executed = 0
        self.override_shadow_entry_blocked = 0

        now = datetime.now(UTC)
        self._override_6h_events: deque[tuple[datetime, float]] = deque()
        self._override_6h_sum = 0.0
        self._override_6h_tripped = False
        self._override_day_key = self._utc_day_key(now)
        self._override_week_key = self._utc_week_key(now)
        self._override_day_sum = 0.0
        self._override_week_sum = 0.0
        self._override_day_tripped = False
        self._override_week_tripped = False

    def _reject(self, reason: str) -> BtcTradeSignal | None:
        self.last_rejection_reason = str(reason)
        return None

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _utc_day_key(ts: datetime) -> tuple[int, int, int]:
        d = ts.date()
        return (d.year, d.month, d.day)

    @staticmethod
    def _utc_week_key(ts: datetime) -> tuple[int, int]:
        iso = ts.isocalendar()
        return (iso.year, iso.week)

    def _apply_override_rollovers(self, now_utc: datetime) -> None:
        day_key = self._utc_day_key(now_utc)
        if day_key != self._override_day_key:
            self._override_day_key = day_key
            self._override_day_sum = 0.0
            self._override_day_tripped = False

        week_key = self._utc_week_key(now_utc)
        if week_key != self._override_week_key:
            self._override_week_key = week_key
            self._override_week_sum = 0.0
            self._override_week_tripped = False

    def _prune_override_6h(self, now_utc: datetime) -> None:
        cutoff = now_utc - self.OVERRIDE_6H_WINDOW
        while self._override_6h_events and self._override_6h_events[0][0] < cutoff:
            _, old_r = self._override_6h_events.popleft()
            self._override_6h_sum -= float(old_r)

    def _refresh_override_6h_latch(self) -> None:
        if self._override_6h_tripped:
            if self._override_6h_sum > self.OVERRIDE_6H_RELEASE_R:
                self._override_6h_tripped = False
        elif self._override_6h_sum <= self.OVERRIDE_6H_TRIP_R:
            self._override_6h_tripped = True

    def _refresh_override_fuses(self, now_utc: datetime) -> None:
        self._apply_override_rollovers(now_utc)
        self._prune_override_6h(now_utc)
        self._refresh_override_6h_latch()

    def rebuild_override_fuses(self, close_events: list[tuple[datetime, float]]) -> None:
        now = datetime.now(UTC)
        self._override_6h_events.clear()
        self._override_6h_sum = 0.0
        self._override_6h_tripped = False
        self._override_day_key = self._utc_day_key(now)
        self._override_week_key = self._utc_week_key(now)
        self._override_day_sum = 0.0
        self._override_week_sum = 0.0
        self._override_day_tripped = False
        self._override_week_tripped = False

        for ts, realized_r in sorted(close_events, key=lambda x: self._as_utc(x[0])):
            self.ingest_override_realized_r(ts, realized_r)
        self._refresh_override_fuses(now)

    def ingest_override_realized_r(self, timestamp_exit: datetime, realized_r: float) -> None:
        ts_utc = self._as_utc(timestamp_exit)
        r_value = float(realized_r)
        self._refresh_override_fuses(ts_utc)

        self._override_6h_events.append((ts_utc, r_value))
        self._override_6h_sum += r_value
        self._override_day_sum += r_value
        self._override_week_sum += r_value

        self._refresh_override_6h_latch()
        if self._override_day_sum <= self.OVERRIDE_DAY_TRIP_R:
            self._override_day_tripped = True
        if self._override_week_sum <= self.OVERRIDE_WEEK_TRIP_R:
            self._override_week_tripped = True

    def _check_override_fuse_block(self, now_utc: datetime) -> str | None:
        self._refresh_override_fuses(now_utc)
        if self._override_6h_tripped:
            return "6h"
        if self._override_day_tripped:
            return "day"
        if self._override_week_tripped:
            return "week"
        return None

    def note_override_result(self, executed: bool, model_rejected: bool) -> None:
        if executed:
            self.override_executed += 1
        elif model_rejected:
            self.override_rejected_model += 1
        else:
            self.override_shadow_entry_blocked += 1

    def _get_sl_tp_mults(self, row: pd.Series, is_reversal: bool, direction: int) -> tuple[float, float] | None:
        in_bull_ob = int(row.get("in_bull_ob", 0) or 0)
        in_bear_ob = int(row.get("in_bear_ob", 0) or 0)
        bull_fvg = int(row.get("bull_fvg", 0) or 0)
        bear_fvg = int(row.get("bear_fvg", 0) or 0)

        if direction == 1:
            in_dir_ob = in_bull_ob == 1
            in_dir_fvg = bull_fvg == 1
        else:
            in_dir_ob = in_bear_ob == 1
            in_dir_fvg = bear_fvg == 1

        if is_reversal and in_dir_ob and in_dir_fvg:
            return 1.0, 4.0

        if (not is_reversal) and in_dir_ob and in_dir_fvg:
            return 1.2, 3.5

        if (not is_reversal) and (in_dir_ob or in_dir_fvg) and not (in_dir_ob and in_dir_fvg):
            return 1.5, 3.0

        return None  # no directional structure — reject

    def process(
        self,
        feature_row: pd.DataFrame,
        current_price: float,
        capital_inr: float,
        open_trades: int = 0,
        max_concurrent: int = 1,
        override: bool = False,
    ) -> BtcTradeSignal | None:
        self.last_override_reached_model = False
        if len(feature_row) != 1:
            raise ValueError("feature_row must be a single-row DataFrame")

        current_price = float(current_price)
        capital_inr = float(capital_inr)
        if current_price <= 0 or capital_inr <= 0:
            return self._reject("INVALID_INPUT")

        # Confluence gate first: only evaluate model if a directional setup fired.
        row = feature_row.iloc[-1]
        self.last_drift_alerts = self.drift_monitor.check(row)
        regime = classify_regime(row)
        self.last_regime = regime
        active_model = self._regime_models.get(regime, self.model)
        long_signal = int(row.get("long_signal", 0) or 0)
        short_signal = int(row.get("short_signal", 0) or 0)
        if long_signal == short_signal:
            return self._reject("NO_DIRECTIONAL_CONFLUENCE")
        direction = 1 if long_signal == 1 else -1

        if open_trades >= max_concurrent:
            return self._reject(f"MAX_CONCURRENT({open_trades}>={max_concurrent})")
        # Use HTF RSI (45m preferred, 15m fallback, 1m last resort).
        # 1m RSI spikes to 80-90 on any strong momentum candle — not meaningful for a filter.
        rsi = float(
            row.get("45m_rsi_14")
            or row.get("15m_rsi_14")
            or row.get("rsi_14")
            or 50.0
        )
        if direction == 1 and rsi > 80:
            return self._reject(f"RSI_OVERBOUGHT({rsi:.1f})")
        if direction == -1 and rsi < 20:
            return self._reject(f"RSI_OVERSOLD({rsi:.1f})")
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

        if bool(override):
            self.override_candidates_seen += 1
            block = self._check_override_fuse_block(datetime.now(UTC))
            if block == "6h":
                self.override_blocked_6h += 1
                return self._reject("OVERRIDE_FUSE_6H")
            if block == "day":
                self.override_blocked_day += 1
                return self._reject("OVERRIDE_FUSE_DAY")
            if block == "week":
                self.override_blocked_week += 1
                return self._reject("OVERRIDE_FUSE_WEEK")
            self.override_reached_model += 1
            self.last_override_reached_model = True

        probs = active_model.predict_proba(x)[0]
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

        mults = self._get_sl_tp_mults(row, is_reversal, direction)
        if mults is None:
            return self._reject("NO_STRUCTURE_CONFLUENCE")
        sl_mult, tp_mult = mults
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult
        if sl_dist <= 0:
            return self._reject("SL_DISTANCE_INVALID")

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
        confidence_scale = (win_probability - min_confidence) / (1.0 - min_confidence)
        confidence_scale = max(0.1, min(1.0, confidence_scale))
        kelly_risk_pct = self.RISK_PCT * confidence_scale
        risk_usd = capital_usd * kelly_risk_pct * risk_mult
        raw_btc = risk_usd / sl_dist  # BTC needed to lose exactly risk_usd on SL hit
        # Round down to nearest 0.001 BTC increment, enforce minimum.
        contracts = max(BTC_CONTRACT, int(raw_btc / BTC_CONTRACT) * BTC_CONTRACT)
        # Cap at MAX_LEVERAGE: notional = contracts * price <= capital_usd * MAX_LEVERAGE.
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
            override=bool(override),
        )
