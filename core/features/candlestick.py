import numpy as np
import pandas as pd
import talib


def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    prev_close = df["close"].shift(1)
    body_size   = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    upper_wick  = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick  = df[["open", "close"]].min(axis=1) - df["low"]

    featured["body_size"]  = body_size
    featured["body_pct"]   = body_size / (candle_range + 1e-9)
    featured["upper_wick"] = upper_wick
    featured["lower_wick"] = lower_wick
    featured["range_pct"]  = candle_range / df["close"]

    # TA-Lib pattern recognition (100=bullish, -100=bearish, 0=none)
    featured["is_doji"]           = (talib.CDLDOJI(o, h, l, c) != 0).astype(int)
    featured["is_hammer"]         = (talib.CDLHAMMER(o, h, l, c) == 100).astype(int)
    featured["is_shooting_star"]  = (talib.CDLSHOOTINGSTAR(o, h, l, c) == -100).astype(int)
    engulfing                     = talib.CDLENGULFING(o, h, l, c)
    featured["is_engulfing_bull"] = (engulfing == 100).astype(int)
    featured["is_engulfing_bear"] = (engulfing == -100).astype(int)
    # Pin bar: long wick (either side) relative to body — keep manual (no TA-Lib equivalent)
    featured["is_pin_bar"] = (
        (lower_wick > 2.5 * body_size) | (upper_wick > 2.5 * body_size)
    ).astype(int)

    featured["gap_up"]      = (df["open"] > prev_close * 1.002).astype(int)
    featured["gap_down"]    = (df["open"] < prev_close * 0.998).astype(int)
    featured["momentum_3"]  = (df["close"] / df["close"].shift(3)) - 1
    featured["momentum_5"]  = (df["close"] / df["close"].shift(5)) - 1

    return featured
