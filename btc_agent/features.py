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
    window = left + right + 1
    roll_max = series.rolling(window, min_periods=window).max().shift(-right)
    return series.where(series == roll_max)


def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    """Returns pivot low price where confirmed, else NaN (Pine Script: ta.pivotlow)."""
    window = left + right + 1
    roll_min = series.rolling(window, min_periods=window).min().shift(-right)
    return series.where(series == roll_min)


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

    ph_arr = ph.to_numpy(dtype=float)
    pl_arr = pl.to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)

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
        if not np.isnan(ph_arr[i]):
            last_high = ph_arr[i]
            high_broken = False
        if not np.isnan(pl_arr[i]):
            last_low = pl_arr[i]
            low_broken = False

        close = close_arr[i]

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

    ph_arr = ph.to_numpy(dtype=float)
    pl_arr = pl.to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    slope_arr = slope.to_numpy(dtype=float)

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
        s = slope_arr[i] if not np.isnan(slope_arr[i]) else 0.0

        if not np.isnan(ph_arr[i]):
            sp_h = s
            up   = ph_arr[i]
        else:
            up = (up - sp_h) if not np.isnan(up) else np.nan

        if not np.isnan(pl_arr[i]):
            sp_l = s
            lo   = pl_arr[i]
        else:
            lo = (lo + sp_l) if not np.isnan(lo) else np.nan

        upper[i] = up
        lower[i] = lo
        slope_ph[i] = sp_h
        slope_pl[i] = sp_l

        close = close_arr[i]

        prev_upos = _upos
        prev_dnos = _dnos

        if not np.isnan(ph_arr[i]):
            _upos = 0
        elif not np.isnan(up) and close > (up - sp_h * length):
            _upos = 1

        if not np.isnan(pl_arr[i]):
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

    ph_arr = ph.to_numpy(dtype=float)
    pl_arr = pl.to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)

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
        if not np.isnan(ph_arr[i]):
            last_sh = ph_arr[i]
            sh_crossed = False
        if not np.isnan(pl_arr[i]):
            last_sl = pl_arr[i]
            sl_crossed = False

        swing_high_level[i] = last_sh
        swing_low_level[i]  = last_sl

        close = close_arr[i]

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
    n = len(df)

    low_arr = df["low"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)

    bull_event = np.zeros(n, dtype=bool)
    bear_event = np.zeros(n, dtype=bool)
    bull_fvg_top_event = np.full(n, np.nan)
    bull_fvg_bot_event = np.full(n, np.nan)
    bear_fvg_top_event = np.full(n, np.nan)
    bear_fvg_bot_event = np.full(n, np.nan)

    if n >= 3:
        bull_event[2:] = low_arr[2:] > high_arr[:-2]
        bear_event[2:] = high_arr[2:] < low_arr[:-2]

        bull_fvg_top_event[2:] = np.where(bull_event[2:], low_arr[2:], np.nan)
        bull_fvg_bot_event[2:] = np.where(bull_event[2:], high_arr[:-2], np.nan)

        bear_fvg_top_event[2:] = np.where(bear_event[2:], low_arr[:-2], np.nan)
        bear_fvg_bot_event[2:] = np.where(bear_event[2:], high_arr[2:], np.nan)

    bull_fvg = np.zeros(n, dtype=int)
    bear_fvg = np.zeros(n, dtype=int)
    fvg_top = np.full(n, np.nan)
    fvg_bot = np.full(n, np.nan)

    active_bull_top = np.nan
    active_bull_bot = np.nan
    active_bear_top = np.nan
    active_bear_bot = np.nan
    active_bull_idx = -1
    active_bear_idx = -1

    for i in range(n):
        if bull_event[i]:
            active_bull_top = bull_fvg_top_event[i]
            active_bull_bot = bull_fvg_bot_event[i]
            active_bull_idx = i

        if bear_event[i]:
            active_bear_top = bear_fvg_top_event[i]
            active_bear_bot = bear_fvg_bot_event[i]
            active_bear_idx = i

        c = close_arr[i]

        if not np.isnan(active_bull_bot) and c < active_bull_bot:
            active_bull_top = np.nan
            active_bull_bot = np.nan
            active_bull_idx = -1

        if not np.isnan(active_bear_top) and c > active_bear_top:
            active_bear_top = np.nan
            active_bear_bot = np.nan
            active_bear_idx = -1

        if not np.isnan(active_bull_top) and active_bull_bot <= c <= active_bull_top:
            bull_fvg[i] = 1

        if not np.isnan(active_bear_top) and active_bear_bot <= c <= active_bear_top:
            bear_fvg[i] = 1

        if active_bull_idx == -1 and active_bear_idx == -1:
            fvg_top[i] = np.nan
            fvg_bot[i] = np.nan
        elif active_bull_idx >= active_bear_idx:
            fvg_top[i] = active_bull_top
            fvg_bot[i] = active_bull_bot
        else:
            fvg_top[i] = active_bear_top
            fvg_bot[i] = active_bear_bot

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

    ph_arr = ph.to_numpy(dtype=float)
    pl_arr = pl.to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)
    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)

    last_high = np.nan
    last_low  = np.nan
    high_broken = False
    low_broken  = False

    for i in range(left, len(df)):
        if not np.isnan(ph_arr[i]):
            last_high = ph_arr[i]
            high_broken = False
        if not np.isnan(pl_arr[i]):
            last_low = pl_arr[i]
            low_broken = False

        close = close_arr[i]

        # Bullish BOS → find last bearish candle before this bar
        if not np.isnan(last_high) and not high_broken and close > last_high:
            high_broken = True
            # scan back to find last bearish candle
            for j in range(i - 1, max(i - 20, 0), -1):
                if close_arr[j] < open_arr[j]:  # bearish candle
                    bull_ob_top[i] = high_arr[j]
                    bull_ob_bot[i] = low_arr[j]
                    break

        # Bearish BOS → find last bullish candle before this bar
        if not np.isnan(last_low) and not low_broken and close < last_low:
            low_broken = True
            for j in range(i - 1, max(i - 20, 0), -1):
                if close_arr[j] > open_arr[j]:  # bullish candle
                    bear_ob_top[i] = high_arr[j]
                    bear_ob_bot[i] = low_arr[j]
                    break

    df["bull_ob_top"] = bull_ob_top
    df["bull_ob_bot"] = bull_ob_bot
    df["bear_ob_top"] = bear_ob_top
    df["bear_ob_bot"] = bear_ob_bot

    # Boolean: is close inside a recent OB zone?
    in_bull_ob = np.zeros(len(df), dtype=int)
    in_bear_ob = np.zeros(len(df), dtype=int)
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

        c = close_arr[i]

        # Invalidate bull OB when close trades below its bottom
        if not np.isnan(last_bull_bot) and c < last_bull_bot:
            last_bull_top = np.nan
            last_bull_bot = np.nan

        # Invalidate bear OB when close trades above its top
        if not np.isnan(last_bear_top) and c > last_bear_top:
            last_bear_top = np.nan
            last_bear_bot = np.nan

        if not np.isnan(last_bull_top) and last_bull_bot <= c <= last_bull_top:
            in_bull_ob[i] = 1
        if not np.isnan(last_bear_top) and last_bear_bot <= c <= last_bear_top:
            in_bear_ob[i] = 1

    df["in_bull_ob"] = in_bull_ob
    df["in_bear_ob"] = in_bear_ob

    return df


# ── Standard Indicators ───────────────────────────────────────────────────────

def add_market_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    try:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("index is not DatetimeIndex")
        idx_utc = idx.tz_convert("UTC") if idx.tz is not None else idx
        hour = idx_utc.hour
        minute = idx_utc.minute

        df["session_asia"] = np.isin(hour, [0, 1, 2, 3, 4, 5, 6, 7]).astype(int)
        df["session_london"] = np.isin(hour, [7, 8, 9, 10, 11, 12, 13, 14, 15]).astype(int)
        df["session_ny"] = np.isin(hour, [13, 14, 15, 16, 17, 18, 19, 20]).astype(int)
        df["session_overlap"] = np.isin(hour, [13, 14, 15]).astype(int)
        df["session_open_impulse"] = ((hour == 7) | ((hour == 13) & (minute >= 30))).astype(int)
    except Exception:
        df["session_asia"] = 0
        df["session_london"] = 0
        df["session_ny"] = 0
        df["session_overlap"] = 0
        df["session_open_impulse"] = 0
        idx_utc = None

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]
    if isinstance(idx_utc, pd.DatetimeIndex):
        date_group = idx_utc.normalize()
        cumtp = tp_vol.groupby(date_group).cumsum()
        cumvol = df["volume"].groupby(date_group).cumsum()
        vwap = cumtp / cumvol.replace(0, np.nan)
        df["vwap"] = vwap
    else:
        df["vwap"] = np.nan

    if "taker_buy_base" in df.columns:
        buy_vol = pd.to_numeric(df["taker_buy_base"], errors="coerce").fillna(0.0)
        sell_vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) - buy_vol
        vol_delta = buy_vol - sell_vol
    else:
        wick_range = (df["high"] - df["low"]).replace(0, np.nan)
        buy_fraction = (2.0 * df["close"] - df["high"] - df["low"]) / (2.0 * wick_range)
        buy_fraction = buy_fraction.clip(-1, 1).fillna(0)
        vol_delta = buy_fraction * df["volume"]

    vol_delta_roll = vol_delta.rolling(14, min_periods=1).sum()
    vol_roll = df["volume"].rolling(14, min_periods=1).sum()
    df["vol_delta"] = vol_delta
    df["vol_delta_norm"] = (vol_delta_roll / vol_roll.replace(0, np.nan)).fillna(0).clip(-1, 1)

    return df


def add_funding_oi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "funding_rate" in df.columns and not pd.to_numeric(df["funding_rate"], errors="coerce").isna().all():
        fr = pd.to_numeric(df["funding_rate"], errors="coerce")
        fr_mean = fr.rolling(48, min_periods=10).mean()
        fr_std = fr.rolling(48, min_periods=10).std().replace(0, np.nan)
        df["funding_rate_norm"] = ((fr - fr_mean) / fr_std).fillna(0).clip(-4, 4)
        df["funding_rate_extreme"] = (df["funding_rate_norm"].abs() > 2).astype(int)
    else:
        df["funding_rate_norm"] = 0
        df["funding_rate_extreme"] = 0

    if "open_interest" in df.columns and not pd.to_numeric(df["open_interest"], errors="coerce").isna().all():
        oi = pd.to_numeric(df["open_interest"], errors="coerce")
        df["oi_change_pct"] = oi.pct_change(1).fillna(0).clip(-0.5, 0.5)
        df["oi_impulse"] = df["oi_change_pct"].rolling(5, min_periods=1).sum()
    else:
        df["oi_change_pct"] = 0
        df["oi_impulse"] = 0

    return df


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
    if "vwap" in df.columns:
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["atr_14"]

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

    df = add_market_microstructure(df)

    logger.debug("Detecting pivots...")
    df = add_pivots(df, pivot_left, pivot_right)

    logger.debug("Computing S/R levels...")
    df = get_nearest_levels(df)

    logger.debug("Adding standard indicators (ATR, RSI, MACD, volume)...")
    df = add_standard_indicators(df)

    df = add_funding_oi(df)

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
