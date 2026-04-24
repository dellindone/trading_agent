import numpy as np
import pandas as pd

from core.features.indicators import atr


def _compute_consecutive_streak(mask: pd.Series) -> pd.Series:
    streak = []
    current = 0

    for value in mask.fillna(False):
        if value:
            current += 1
        else:
            current = 0
        streak.append(current)

    return pd.Series(streak, index=mask.index)


def _session_open_reference(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame["open"]

    if frame.index.tz is None:
        local_dates = frame.index
    else:
        local_dates = frame.index.tz_convert("Asia/Kolkata")

    session_key = pd.Series(local_dates.date, index=frame.index)
    return frame["open"].groupby(session_key).transform("first")


def compute_pattern_context(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    previous_high = featured["high"].shift(1)
    previous_low = featured["low"].shift(1)

    featured["rolling_high_20"] = featured["close"].rolling(20, min_periods=1).max()
    featured["rolling_low_20"] = featured["close"].rolling(20, min_periods=1).min()
    featured["pct_from_high_20"] = (featured["close"] - featured["rolling_high_20"]) / featured["rolling_high_20"]
    featured["pct_from_low_20"] = (featured["close"] - featured["rolling_low_20"]) / featured["rolling_low_20"]

    featured["near_resistance"] = (featured["pct_from_high_20"] > -0.005).astype(int)
    featured["near_support"] = (featured["pct_from_low_20"] < 0.005).astype(int)

    non_zero_pct = (featured["volume"].fillna(0) > 0).mean()
    if non_zero_pct < 0.5:
        featured["volume_ma_20"] = 0.0
        featured["volume_ratio"] = 0.0
        featured["high_volume"] = 0
        featured["low_volume"] = 0
    else:
        featured["volume_ma_20"] = featured["volume"].rolling(20, min_periods=1).mean()
        featured["volume_ratio"] = (
            featured["volume"] / featured["volume_ma_20"].replace(0.0, np.nan)
        ).fillna(0.0)
        featured["high_volume"] = (featured["volume_ratio"] > 1.5).astype(int)
        featured["low_volume"] = (featured["volume_ratio"] < 0.5).astype(int)

    featured["consecutive_up"] = _compute_consecutive_streak(featured["close"] > featured["open"])
    featured["consecutive_down"] = _compute_consecutive_streak(featured["close"] < featured["open"])

    featured["inside_bar"] = ((featured["high"] <= previous_high) & (featured["low"] >= previous_low)).astype(int)
    featured["outside_bar"] = ((featured["high"] > previous_high) & (featured["low"] < previous_low)).astype(int)

    featured["atr_14"] = atr(featured, period=14)
    featured["atr_pct"] = featured["atr_14"] / featured["close"]

    session_open = _session_open_reference(featured)
    featured["close_vs_open_session"] = (featured["close"] / session_open) - 1

    return featured
