import numpy as np
import pandas as pd

from core.features.indicators import atr, ema

try:
    import talib
except ImportError:
    talib = None


class RegimeDetector:
    def _suffix(self, column: str, tf: str) -> str:
        return column if tf == "5m" else f"{column}_{tf}"

    def _adx_manual(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        atr_values = atr(df, period=period)
        plus_dm_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
        minus_dm_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()

        plus_di = 100 * (plus_dm_smoothed / atr_values.replace(0.0, np.nan))
        minus_di = 100 * (minus_dm_smoothed / atr_values.replace(0.0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        if talib is not None:
            return pd.Series(
                talib.ADX(
                    df["high"].astype(float).values,
                    df["low"].astype(float).values,
                    df["close"].astype(float).values,
                    timeperiod=period,
                ),
                index=df.index,
            )
        return self._adx_manual(df, period=period)

    def _pivot_high(self, highs: pd.Series, lookback: int = 5) -> pd.Series:
        # Live-safe pivot proxy: only use current/past bars (no centered window).
        # This avoids lookahead leakage that would otherwise diverge between
        # training and live inference near the right edge of the series.
        window = lookback + 1
        rolling_max = highs.rolling(window, min_periods=1).max()
        return ((highs == rolling_max) & highs.notna()).astype(int)

    def _pivot_low(self, lows: pd.Series, lookback: int = 5) -> pd.Series:
        # Live-safe pivot proxy: only use current/past bars (no centered window).
        window = lookback + 1
        rolling_min = lows.rolling(window, min_periods=1).min()
        return ((lows == rolling_min) & lows.notna()).astype(int)

    def detect(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        featured = df.copy()
        suffix_tf = tf

        ema_20 = ema(featured["close"], 20)
        ema_50 = ema(featured["close"], 50)
        ema_200 = ema(featured["close"], 200) if tf == "D" else ema_50.copy()

        adx_14 = self._adx(featured, period=14)
        adx_strong = (adx_14 > 25).astype(int)

        ema_slope = ema_20.diff()

        trend_regime = pd.Series(0, index=featured.index, dtype=int)
        trend_regime[(adx_14 > 25) & (ema_slope > 0)] = 1
        trend_regime[(adx_14 > 25) & (ema_slope < 0)] = 2

        pivot_high = self._pivot_high(featured["high"], lookback=5)
        pivot_low = self._pivot_low(featured["low"], lookback=5)

        pivot_high_value = featured["high"].where(pivot_high == 1)
        pivot_low_value = featured["low"].where(pivot_low == 1)

        previous_pivot_high = pivot_high_value.ffill().shift(1)
        previous_pivot_low = pivot_low_value.ffill().shift(1)

        hh = ((pivot_high == 1) & (pivot_high_value > previous_pivot_high)).astype(int)
        ll = ((pivot_low == 1) & (pivot_low_value < previous_pivot_low)).astype(int)

        featured[self._suffix("trend_regime", suffix_tf)] = trend_regime
        featured[self._suffix("ema_20", suffix_tf)] = ema_20
        featured[self._suffix("ema_50", suffix_tf)] = ema_50
        featured[self._suffix("ema_200", suffix_tf)] = ema_200
        featured[self._suffix("ema_stack_bull", suffix_tf)] = ((ema_20 > ema_50) & (ema_50 > ema_200)).astype(int)
        featured[self._suffix("ema_stack_bear", suffix_tf)] = ((ema_20 < ema_50) & (ema_50 < ema_200)).astype(int)
        featured[self._suffix("adx_14", suffix_tf)] = adx_14
        featured[self._suffix("adx_strong", suffix_tf)] = adx_strong
        featured[self._suffix("price_above_ema20", suffix_tf)] = (featured["close"] > ema_20).astype(int)
        featured[self._suffix("price_above_ema50", suffix_tf)] = (featured["close"] > ema_50).astype(int)
        featured[self._suffix("pivot_high", suffix_tf)] = pivot_high
        featured[self._suffix("pivot_low", suffix_tf)] = pivot_low
        featured[self._suffix("hh", suffix_tf)] = hh
        featured[self._suffix("ll", suffix_tf)] = ll

        return featured
