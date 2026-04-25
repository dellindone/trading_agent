"""Regime classification utilities for BTC model routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ["bull_normal", "bull_high_vol", "bear_normal", "bear_high_vol"]
MIN_REGIME_SAMPLES = 500   # minimum bars to train a regime model


def classify_regime(row: pd.Series) -> str:
    """Classify a single feature row into one of 4 regime strings."""
    sma_200 = row.get("sma_200", np.nan)
    close = row.get("close", np.nan)
    if pd.isna(sma_200) or pd.isna(close):
        return "unknown"

    trend = "bull" if float(close) > float(sma_200) else "bear"

    atr_recent = row.get("atr_14_recent", np.nan)
    atr_hist = row.get("atr_14_hist", np.nan)
    if pd.isna(atr_recent) or pd.isna(atr_hist):
        vol = "normal"
    else:
        vol = "high_vol" if float(atr_recent) > 1.5 * float(atr_hist) else "normal"

    return f"{trend}_{vol}"


def add_regime_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'atr_14_recent', 'atr_14_hist', 'regime' columns to a full feature df."""
    df = df.copy()
    if "atr_14" in df.columns:
        df["atr_14_recent"] = df["atr_14"].rolling(20, min_periods=5).mean()
        df["atr_14_hist"] = df["atr_14"].rolling(200, min_periods=50).mean()
    else:
        df["atr_14_recent"] = np.nan
        df["atr_14_hist"] = np.nan
    df["regime"] = df.apply(classify_regime, axis=1)
    return df
