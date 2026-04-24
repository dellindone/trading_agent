import numpy as np
import pandas as pd


def _bucket_ratio(value: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.35, 0.75, 1.25, np.inf]
    labels = ["TIGHT", "MEDIUM", "WIDE", "VERY_WIDE"]
    # Persist as concrete strings instead of categorical codes to keep
    # downstream interfaces stable across parquet/CSV round-trips.
    return pd.cut(value, bins=bins, labels=labels).astype("string")


def _phase1_multiplier(favorable_ratio: pd.Series) -> pd.Series:
    conditions = [
        favorable_ratio <= 0.75,
        favorable_ratio <= 1.25,
        favorable_ratio <= 1.75,
    ]
    choices = [0.5, 1.0, 1.5]
    return pd.Series(np.select(conditions, choices, default=2.0), index=favorable_ratio.index)


def _select_trail_tf(favorable_ratio: pd.Series) -> pd.Series:
    conditions = [
        favorable_ratio <= 0.75,
        favorable_ratio <= 1.5,
    ]
    choices = ["5m", "15m"]
    return pd.Series(np.select(conditions, choices, default="60m"), index=favorable_ratio.index)


def _forward_window_max(series: pd.Series, horizon: int) -> pd.Series:
    return series[::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)


def _forward_window_min(series: pd.Series, horizon: int) -> pd.Series:
    return series[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)


def build_labels(feature_df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Canonical label builder for the project.

    Output columns:
    - direction (Int64): 1 bullish, 0 bearish
    - adverse_excursion (float)
    - favorable_excursion (float)
    - sl_bin (string): TIGHT/MEDIUM/WIDE/VERY_WIDE
    - trail_bin (string): TIGHT/MEDIUM/WIDE/VERY_WIDE
    - phase1_target (float)
    - trail_tf (string): 5m/15m/60m
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if "atr_14" not in feature_df.columns:
        raise ValueError("df must contain atr_14. Call compute_pattern_context first.")

    future_close = feature_df["close"].shift(-horizon)
    future_high_max = _forward_window_max(feature_df["high"], horizon)
    future_low_min = _forward_window_min(feature_df["low"], horizon)

    direction = (future_close > feature_df["close"]).astype("Int64")

    bullish_adverse = (feature_df["close"] - future_low_min).clip(lower=0.0)
    bearish_adverse = (future_high_max - feature_df["close"]).clip(lower=0.0)
    adverse_excursion = np.where(direction.fillna(0).astype(int) == 1, bullish_adverse, bearish_adverse)

    bullish_favorable = (future_high_max - feature_df["close"]).clip(lower=0.0)
    bearish_favorable = (feature_df["close"] - future_low_min).clip(lower=0.0)
    favorable_excursion = np.where(direction.fillna(0).astype(int) == 1, bullish_favorable, bearish_favorable)

    atr_base = feature_df["atr_14"].replace(0.0, np.nan)
    adverse_ratio = pd.Series(adverse_excursion, index=feature_df.index) / atr_base
    favorable_ratio = pd.Series(favorable_excursion, index=feature_df.index) / atr_base

    phase1_multiplier = _phase1_multiplier(favorable_ratio)

    labels = pd.DataFrame(index=feature_df.index)
    labels["direction"] = direction
    labels["adverse_excursion"] = pd.Series(adverse_excursion, index=feature_df.index)
    labels["favorable_excursion"] = pd.Series(favorable_excursion, index=feature_df.index)
    labels["sl_bin"] = _bucket_ratio(adverse_ratio)
    labels["trail_bin"] = _bucket_ratio(favorable_ratio)
    labels["phase1_target"] = phase1_multiplier * feature_df["atr_14"]
    labels["trail_tf"] = _select_trail_tf(favorable_ratio)

    return labels.iloc[:-horizon].copy()
