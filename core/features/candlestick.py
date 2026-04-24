import pandas as pd


def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    previous_open = featured["open"].shift(1)
    previous_close = featured["close"].shift(1)

    body_size = (featured["close"] - featured["open"]).abs()
    candle_range = featured["high"] - featured["low"]
    upper_wick = featured["high"] - featured[["open", "close"]].max(axis=1)
    lower_wick = featured[["open", "close"]].min(axis=1) - featured["low"]

    featured["body_size"] = body_size
    featured["body_pct"] = body_size / (candle_range + 1e-9)
    featured["upper_wick"] = upper_wick
    featured["lower_wick"] = lower_wick

    featured["is_doji"] = (featured["body_pct"] < 0.1).astype(int)
    featured["is_hammer"] = (
        (lower_wick > (2 * body_size))
        & (upper_wick < body_size)
        & (featured["close"] > featured["open"])
    ).astype(int)
    featured["is_shooting_star"] = (
        (upper_wick > (2 * body_size))
        & (lower_wick < body_size)
        & (featured["close"] < featured["open"])
    ).astype(int)
    featured["is_engulfing_bull"] = (
        (previous_close < previous_open)
        & (featured["close"] > featured["open"])
        & (featured["close"] > previous_open)
        & (featured["open"] < previous_close)
    ).astype(int)
    featured["is_engulfing_bear"] = (
        (previous_close > previous_open)
        & (featured["close"] < featured["open"])
        & (featured["close"] < previous_open)
        & (featured["open"] > previous_close)
    ).astype(int)
    featured["is_pin_bar"] = ((lower_wick > (2.5 * body_size)) | (upper_wick > (2.5 * body_size))).astype(int)

    featured["range_pct"] = candle_range / featured["close"]
    featured["gap_up"] = (featured["open"] > (previous_close * 1.002)).astype(int)
    featured["gap_down"] = (featured["open"] < (previous_close * 0.998)).astype(int)
    featured["momentum_3"] = (featured["close"] / featured["close"].shift(3)) - 1
    featured["momentum_5"] = (featured["close"] / featured["close"].shift(5)) - 1

    return featured
