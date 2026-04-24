"""
Phase 2: Feature Engineering
Translates all Pine Script indicators into Python/pandas:
  - Moving Averages (EMA 8, EMA 21, SMA 50, SMA 200)
  - Pivot Highs/Lows → Support/Resistance levels
  - Break of Structure (BOS) - bullish & bearish
  - ATR-based dynamic trendlines with breakout detection
  - SMC: CHoCH detection, Order Block zones, Fair Value Gaps
  - ATR, RSI, MACD, Volume indicators (for ML features)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Allow importing from the shared core package
_TRADING_AGENT_ROOT = Path(__file__).resolve().parents[1] / "trading_agent"
if str(_TRADING_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_AGENT_ROOT))

from core.features.candlestick import compute_candlestick_features  # noqa: E402


# ── Moving Averages ──────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_8"]   = df["close"].ewm(span=8,   adjust=False).mean()
    df["ema_21"]  = df["close"].ewm(span=21,  adjust=False).mean()
    df["sma_50"]  = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    return df


# ── Pivot Highs / Lows ───────────────────────────────────────────────────────

def pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    """Returns pivot high price where confirmed, else NaN (Pine Script: ta.pivothigh)."""
    result = pd.Series(np.nan, index=series.index)
    arr = series.values
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == max(window):
            result.iloc[i] = arr[i]
    return result


def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    """Returns pivot low price where confirmed, else NaN (Pine Script: ta.pivotlow)."""
    result = pd.Series(np.nan, index=series.index)
    arr = series.values
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == min(window):
            result.iloc[i] = arr[i]
    return result


def add_pivots(df: pd.DataFrame, left: int = 10, right: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["pivot_high"] = pivot_high(df["high"], left, right)
    df["pivot_low"]  = pivot_low(df["low"],   left, right)
    return df


# ── Support / Resistance Levels ──────────────────────────────────────────────

def get_nearest_levels(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    For each bar, find:
      - nearest_resistance: closest pivot high above current close
      - nearest_support:    closest pivot low below current close
    """
    df = df.copy()
    n = len(df)
    nearest_resistance = np.full(n, np.nan)
    nearest_support = np.full(n, np.nan)

    closes = df["close"].to_numpy(dtype=float)
    index_ns = df.index.view("int64")

    ph = df["pivot_high"].dropna()
    pl = df["pivot_low"].dropna()
    ph_idx = ph.index.view("int64")
    pl_idx = pl.index.view("int64")
    ph_vals = ph.to_numpy(dtype=float)
    pl_vals = pl.to_numpy(dtype=float)

    for i in range(n):
        close = closes[i]
        idx_ns = index_ns[i]

        # Candidate pivot highs up to current bar (last `lookback` pivots only).
        right_ph = np.searchsorted(ph_idx, idx_ns, side="right")
        if right_ph > 0:
            left_ph = max(0, right_ph - lookback)
            window_ph = ph_vals[left_ph:right_ph]
            above = window_ph[window_ph > close]
            if above.size > 0:
                nearest_resistance[i] = np.min(above)

        # Candidate pivot lows up to current bar (last `lookback` pivots only).
        right_pl = np.searchsorted(pl_idx, idx_ns, side="right")
        if right_pl > 0:
            left_pl = max(0, right_pl - lookback)
            window_pl = pl_vals[left_pl:right_pl]
            below = window_pl[window_pl < close]
            if below.size > 0:
                nearest_support[i] = np.max(below)

    df["nearest_resistance"] = nearest_resistance
    df["nearest_support"] = nearest_support

    return df


# ── Break of Structure (BOS) ─────────────────────────────────────────────────

def add_bos(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    Bullish BOS:  close crosses above last confirmed pivot high
    Bearish BOS:  close crosses below last confirmed pivot low
    (Script 2 logic)
    """
    df = df.copy()
    ph = pivot_high(df["high"], left, right)
    pl = pivot_low(df["low"],   left, right)

    last_high = np.nan
    last_low  = np.nan
    high_broken = False
    low_broken  = False

    bull_bos = np.zeros(len(df), dtype=int)
    bear_bos = np.zeros(len(df), dtype=int)
    bos_level_bull = np.full(len(df), np.nan)
    bos_level_bear = np.full(len(df), np.nan)

    for i in range(len(df)):
        # update last confirmed pivot
        if not np.isnan(ph.iloc[i]):
            last_high = ph.iloc[i]
            high_broken = False
        if not np.isnan(pl.iloc[i]):
            last_low = pl.iloc[i]
            low_broken = False

        close = df["close"].iloc[i]

        # Bullish BOS: close crosses above last pivot high
        if not np.isnan(last_high) and not high_broken and close > last_high:
            bull_bos[i] = 1
            bos_level_bull[i] = last_high
            high_broken = True

        # Bearish BOS: close crosses below last pivot low
        if not np.isnan(last_low) and not low_broken and close < last_low:
            bear_bos[i] = 1
            bos_level_bear[i] = last_low
            low_broken = True

    df["bull_bos"]       = bull_bos
    df["bear_bos"]       = bear_bos
    df["bos_level_bull"] = bos_level_bull
    df["bos_level_bear"] = bos_level_bear
    return df


# ── ATR-based Trendlines with Breaks ─────────────────────────────────────────

def add_trendline_breaks(df: pd.DataFrame, length: int = 14, mult: float = 1.0) -> pd.DataFrame:
    """
    Dynamic trendlines using ATR slope (Script 3: LuxAlgo Trendlines with Breaks).
    upper: descending resistance trendline
    lower: ascending support trendline
    upos=1: price broke above upper trendline (bullish)
    dnos=1: price broke below lower trendline (bearish)
    """
    df = df.copy()

    atr = compute_atr(df, length)
    slope = atr / length * mult

    ph = pivot_high(df["high"], length, length)
    pl = pivot_low(df["low"],   length, length)

    upper      = np.full(len(df), np.nan)
    lower      = np.full(len(df), np.nan)
    slope_ph   = np.zeros(len(df))
    slope_pl   = np.zeros(len(df))
    upos       = np.zeros(len(df), dtype=int)
    dnos       = np.zeros(len(df), dtype=int)

    sp_h = 0.0
    sp_l = 0.0
    up   = np.nan
    lo   = np.nan
    _upos = 0
    _dnos = 0

    for i in range(len(df)):
        s = slope.iloc[i] if not np.isnan(slope.iloc[i]) else 0.0

        if not np.isnan(ph.iloc[i]):
            sp_h = s
            up   = ph.iloc[i]
        else:
            up = (up - sp_h) if not np.isnan(up) else np.nan

        if not np.isnan(pl.iloc[i]):
            sp_l = s
            lo   = pl.iloc[i]
        else:
            lo = (lo + sp_l) if not np.isnan(lo) else np.nan

        upper[i] = up
        lower[i] = lo

        close = df["close"].iloc[i]

        prev_upos = _upos
        prev_dnos = _dnos

        if not np.isnan(ph.iloc[i]):
            _upos = 0
        elif not np.isnan(up) and close > (up - sp_h * length):
            _upos = 1

        if not np.isnan(pl.iloc[i]):
            _dnos = 0
        elif not np.isnan(lo) and close < (lo + sp_l * length):
            _dnos = 1

        upos[i] = 1 if _upos > prev_upos else 0  # newly crossed up
        dnos[i] = 1 if _dnos > prev_dnos else 0  # newly crossed down

    df["tl_upper"]       = upper
    df["tl_lower"]       = lower
    df["tl_bull_break"]  = upos   # 1 = broke above down-trendline (bullish)
    df["tl_bear_break"]  = dnos   # 1 = broke below up-trendline (bearish)
    return df


# ── SMC: CHoCH Detection ─────────────────────────────────────────────────────

def add_choch(df: pd.DataFrame, swing_len: int = 50) -> pd.DataFrame:
    """
    Change of Character (CHoCH): price crosses opposite swing pivot AGAINST current trend.
    BOS: price crosses swing pivot IN LINE with current trend.
    (Simplified from Script 4 SMC logic)
    """
    df = df.copy()

    ph = pivot_high(df["high"], swing_len // 2, swing_len // 2)
    pl = pivot_low(df["low"],   swing_len // 2, swing_len // 2)

    swing_high_level = np.full(len(df), np.nan)
    swing_low_level  = np.full(len(df), np.nan)
    bull_choch = np.zeros(len(df), dtype=int)
    bear_choch = np.zeros(len(df), dtype=int)
    swing_bull_bos = np.zeros(len(df), dtype=int)
    swing_bear_bos = np.zeros(len(df), dtype=int)

    last_sh = np.nan
    last_sl = np.nan
    sh_crossed = False
    sl_crossed = False
    trend_bias = 0  # 1=bull, -1=bear, 0=undefined

    for i in range(len(df)):
        if not np.isnan(ph.iloc[i]):
            last_sh = ph.iloc[i]
            sh_crossed = False
        if not np.isnan(pl.iloc[i]):
            last_sl = pl.iloc[i]
            sl_crossed = False

        swing_high_level[i] = last_sh
        swing_low_level[i]  = last_sl

        close = df["close"].iloc[i]

        # Cross above swing high
        if not np.isnan(last_sh) and not sh_crossed and close > last_sh:
            if trend_bias == -1:
                bull_choch[i] = 1   # CHoCH: was bearish, now breaking high
            else:
                swing_bull_bos[i] = 1  # BOS in direction of bullish trend
            trend_bias = 1
            sh_crossed = True

        # Cross below swing low
        if not np.isnan(last_sl) and not sl_crossed and close < last_sl:
            if trend_bias == 1:
                bear_choch[i] = 1   # CHoCH: was bullish, now breaking low
            else:
                swing_bear_bos[i] = 1  # BOS in direction of bearish trend
            trend_bias = -1
            sl_crossed = True

    df["swing_high_level"] = swing_high_level
    df["swing_low_level"]  = swing_low_level
    df["bull_choch"]       = bull_choch
    df["bear_choch"]       = bear_choch
    df["swing_bull_bos"]   = swing_bull_bos
    df["swing_bear_bos"]   = swing_bear_bos
    df["smc_trend"]        = 0
    # forward-fill trend
    trend_vals = np.zeros(len(df), dtype=int)
    t = 0
    for i in range(len(df)):
        if bull_choch[i] or swing_bull_bos[i]:
            t = 1
        elif bear_choch[i] or swing_bear_bos[i]:
            t = -1
        trend_vals[i] = t
    df["smc_trend"] = trend_vals
    return df


# ── Fair Value Gaps ───────────────────────────────────────────────────────────

def add_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish FVG:  low[0] > high[2]  (gap between candle[0] low and candle[2] high)
    Bearish FVG:  high[0] < low[2]
    (Script 4 SMC logic)
    """
    df = df.copy()
    bull_fvg = np.zeros(len(df), dtype=int)
    bear_fvg = np.zeros(len(df), dtype=int)
    fvg_top  = np.full(len(df), np.nan)
    fvg_bot  = np.full(len(df), np.nan)

    for i in range(2, len(df)):
        # Bullish FVG: current low > 2-bars-ago high
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            bull_fvg[i] = 1
            fvg_top[i]  = df["low"].iloc[i]
            fvg_bot[i]  = df["high"].iloc[i - 2]
        # Bearish FVG: current high < 2-bars-ago low
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            bear_fvg[i] = 1
            fvg_top[i]  = df["low"].iloc[i - 2]
            fvg_bot[i]  = df["high"].iloc[i]

    df["bull_fvg"] = bull_fvg
    df["bear_fvg"] = bear_fvg
    df["fvg_top"]  = fvg_top
    df["fvg_bot"]  = fvg_bot
    return df


# ── Order Block Zones ─────────────────────────────────────────────────────────

def add_order_blocks(df: pd.DataFrame, left: int = 10, right: int = 10) -> pd.DataFrame:
    """
    Order Block (OB): the last bearish/bullish candle before a BOS.
    Bullish OB: last bearish candle before price breaks above pivot high.
    Bearish OB: last bullish candle before price breaks below pivot low.
    """
    df = df.copy()

    bull_ob_top = np.full(len(df), np.nan)
    bull_ob_bot = np.full(len(df), np.nan)
    bear_ob_top = np.full(len(df), np.nan)
    bear_ob_bot = np.full(len(df), np.nan)

    ph = pivot_high(df["high"], left, right)
    pl = pivot_low(df["low"],   left, right)

    last_high = np.nan
    last_low  = np.nan
    high_broken = False
    low_broken  = False

    for i in range(left, len(df)):
        if not np.isnan(ph.iloc[i]):
            last_high = ph.iloc[i]
            high_broken = False
        if not np.isnan(pl.iloc[i]):
            last_low = pl.iloc[i]
            low_broken = False

        close = df["close"].iloc[i]

        # Bullish BOS → find last bearish candle before this bar
        if not np.isnan(last_high) and not high_broken and close > last_high:
            high_broken = True
            # scan back to find last bearish candle
            for j in range(i - 1, max(i - 20, 0), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:  # bearish candle
                    bull_ob_top[i] = df["high"].iloc[j]
                    bull_ob_bot[i] = df["low"].iloc[j]
                    break

        # Bearish BOS → find last bullish candle before this bar
        if not np.isnan(last_low) and not low_broken and close < last_low:
            low_broken = True
            for j in range(i - 1, max(i - 20, 0), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:  # bullish candle
                    bear_ob_top[i] = df["high"].iloc[j]
                    bear_ob_bot[i] = df["low"].iloc[j]
                    break

    df["bull_ob_top"] = bull_ob_top
    df["bull_ob_bot"] = bull_ob_bot
    df["bear_ob_top"] = bear_ob_top
    df["bear_ob_bot"] = bear_ob_bot

    # Boolean: is close inside a recent OB zone?
    df["in_bull_ob"] = 0
    df["in_bear_ob"] = 0
    last_bull_top = np.nan
    last_bull_bot = np.nan
    last_bear_top = np.nan
    last_bear_bot = np.nan

    for i in range(len(df)):
        if not np.isnan(bull_ob_top[i]):
            last_bull_top = bull_ob_top[i]
            last_bull_bot = bull_ob_bot[i]
        if not np.isnan(bear_ob_top[i]):
            last_bear_top = bear_ob_top[i]
            last_bear_bot = bear_ob_bot[i]

        c = df["close"].iloc[i]
        if not np.isnan(last_bull_top) and last_bull_bot <= c <= last_bull_top:
            df.iloc[i, df.columns.get_loc("in_bull_ob")] = 1
        if not np.isnan(last_bear_top) and last_bear_bot <= c <= last_bear_top:
            df.iloc[i, df.columns.get_loc("in_bear_ob")] = 1

    return df


# ── Standard Indicators ───────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low  = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def add_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR
    df["atr_14"] = compute_atr(df, 14)

    # RSI
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12  = df["close"].ewm(span=12, adjust=False).mean()
    ema26  = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"]  = df["volume"] / df["volume_sma_20"]

    # Candle features
    df["body_size"]     = (df["close"] - df["open"]).abs() / df["atr_14"]
    df["upper_wick"]    = (df["high"] - df[["close","open"]].max(axis=1)) / df["atr_14"]
    df["lower_wick"]    = (df[["close","open"]].min(axis=1) - df["low"]) / df["atr_14"]
    df["is_bullish_candle"] = (df["close"] > df["open"]).astype(int)

    # Price position relative to MAs
    df["close_vs_ema8"]   = (df["close"] - df["ema_8"])   / df["atr_14"]
    df["close_vs_ema21"]  = (df["close"] - df["ema_21"])  / df["atr_14"]
    df["close_vs_sma50"]  = (df["close"] - df["sma_50"])  / df["atr_14"]
    df["close_vs_sma200"] = (df["close"] - df["sma_200"]) / df["atr_14"]
    df["ema8_vs_ema21"]   = (df["ema_8"] - df["ema_21"])  / df["atr_14"]

    # Distance to S/R
    df["dist_to_resistance"] = (df["nearest_resistance"] - df["close"]) / df["atr_14"]
    df["dist_to_support"]    = (df["close"] - df["nearest_support"])    / df["atr_14"]

    return df


# ── Master Feature Builder ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, pivot_left: int = 10, pivot_right: int = 10) -> pd.DataFrame:
    """
    Run all feature engineering steps in order.
    Input: raw OHLCV DataFrame with columns [open, high, low, close, volume]
    Output: fully featured DataFrame
    """
    logger.debug("Building moving averages...")
    df = add_moving_averages(df)

    logger.debug("Detecting pivots...")
    df = add_pivots(df, pivot_left, pivot_right)

    logger.debug("Computing S/R levels...")
    df = get_nearest_levels(df)

    logger.debug("Adding standard indicators (ATR, RSI, MACD, volume)...")
    df = add_standard_indicators(df)

    logger.debug("Adding candlestick pattern features...")
    df = compute_candlestick_features(df)

    logger.debug("Adding BOS signals...")
    df = add_bos(df, left=3, right=3)

    logger.debug("Adding trendline breaks...")
    df = add_trendline_breaks(df, length=14, mult=1.0)

    logger.debug("Adding SMC: CHoCH...")
    df = add_choch(df, swing_len=50)

    logger.debug("Adding Fair Value Gaps...")
    df = add_fvg(df)

    logger.debug("Adding Order Block zones...")
    df = add_order_blocks(df, left=10, right=10)

    # Drop rows with NaNs from indicator warmup
    df.dropna(subset=["sma_200", "rsi_14", "atr_14"], inplace=True)

    return df
