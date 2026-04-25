"""BTC trade signal scoring and forward-label simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
FORWARD_BARS = 360


def apply_ema_pair_features(df: pd.DataFrame, ema_fast: int = 8, ema_slow: int = 21) -> pd.DataFrame:
    """Recompute EMA-derived confluence/model features for the selected pair."""
    frame = df.copy()
    fast = max(2, int(ema_fast))
    slow = max(fast + 1, int(ema_slow))

    c = frame["close"].to_numpy(dtype=float)
    frame["ema_8"]  = talib.EMA(c, timeperiod=fast)
    frame["ema_21"] = talib.EMA(c, timeperiod=slow)

    atr = pd.to_numeric(frame.get("atr_14"), errors="coerce")
    if atr is not None:
        denom = atr.replace(0, np.nan)
        frame["close_vs_ema8"] = (frame["close"] - frame["ema_8"]) / denom
        frame["close_vs_ema21"] = (frame["close"] - frame["ema_21"]) / denom
        frame["ema8_vs_ema21"] = (frame["ema_8"] - frame["ema_21"]) / denom

    return frame


def compute_entry_signals(df, ema_fast: int = 8, ema_slow: int = 21) -> pd.DataFrame:
    """Compute bullish/bearish confluence scores and binary entry flags."""
    frame = apply_ema_pair_features(df.copy(), ema_fast=ema_fast, ema_slow=ema_slow)

    bull_score = pd.Series(0, index=frame.index)
    bull_score += (frame["bull_bos"] == 1).astype(int)
    bull_score += (frame["tl_bull_break"] == 1).astype(int)
    bull_score += (frame["bull_choch"] == 1).astype(int)
    bull_score += (frame["swing_bull_bos"] == 1).astype(int)
    bull_score += (frame["ema_8"] > frame["ema_21"]).astype(int)
    bull_score += (frame["close"] > frame["sma_50"]).astype(int)
    bull_score += (frame["rsi_14"] > 50).astype(int)
    bull_score += (frame["macd_hist"] > 0).astype(int)
    bull_score += (frame["in_bull_ob"] == 1).astype(int)
    bull_score += (frame["bull_fvg"] == 1).astype(int)
    bull_score += (frame["smc_trend"] == 1).astype(int)

    bear_score = pd.Series(0, index=frame.index)
    bear_score += (frame["bear_bos"] == 1).astype(int)
    bear_score += (frame["tl_bear_break"] == 1).astype(int)
    bear_score += (frame["bear_choch"] == 1).astype(int)
    bear_score += (frame["swing_bear_bos"] == 1).astype(int)
    bear_score += (frame["ema_8"] < frame["ema_21"]).astype(int)
    bear_score += (frame["close"] < frame["sma_50"]).astype(int)
    bear_score += (frame["rsi_14"] < 50).astype(int)
    bear_score += (frame["macd_hist"] < 0).astype(int)
    bear_score += (frame["in_bear_ob"] == 1).astype(int)
    bear_score += (frame["bear_fvg"] == 1).astype(int)
    bear_score += (frame["smc_trend"] == -1).astype(int)

    frame["bull_score"] = bull_score
    frame["bear_score"] = bear_score

    # HTF direction filter:
    # - Primary gate: 45m trend (resampled from 15m candles).
    # - Fallback gate: 15m trend when 45m context is unavailable.
    # Longs require HTF trend to be not bearish; shorts require not bullish.
    htf_bull_ok = pd.Series(True, index=frame.index)
    htf_bear_ok = pd.Series(True, index=frame.index)
    if "45m_smc_trend" in frame.columns:
        htf_bull_ok &= (frame["45m_smc_trend"] >= 0)   # not -1 (bearish)
        htf_bear_ok &= (frame["45m_smc_trend"] <= 0)   # not +1 (bullish)
    elif "15m_smc_trend" in frame.columns:
        htf_bull_ok &= (frame["15m_smc_trend"] >= 0)   # not -1 (bearish)
        htf_bear_ok &= (frame["15m_smc_trend"] <= 0)   # not +1 (bullish)

    # Reversal override: allow counter-trend entries only with strict confirmation.
    # This helps catch early reversals while keeping HTF trend discipline.
    def _col(name: str) -> pd.Series:
        return frame[name] if name in frame.columns else pd.Series(0, index=frame.index)

    trend_long_signal = (bull_score >= 6) & htf_bull_ok
    trend_short_signal = (bear_score >= 6) & htf_bear_ok

    reversal_bull_pattern = (_col("is_hammer") == 1) | (_col("is_engulfing_bull") == 1)
    reversal_bear_pattern = (_col("is_shooting_star") == 1) | (_col("is_engulfing_bear") == 1)
    reversal_bull_momentum = (frame["rsi_14"] > 52) & (frame["macd_hist"] > 0)
    reversal_bear_momentum = (frame["rsi_14"] < 48) & (frame["macd_hist"] < 0)

    reversal_long_signal = (
        (bull_score >= 8)
        & (~htf_bull_ok)
        & reversal_bull_pattern
        & reversal_bull_momentum
    )
    reversal_short_signal = (
        (bear_score >= 8)
        & (~htf_bear_ok)
        & reversal_bear_pattern
        & reversal_bear_momentum
    )

    frame["trend_long_signal"] = trend_long_signal.astype(int)
    frame["trend_short_signal"] = trend_short_signal.astype(int)
    frame["reversal_long_signal"] = reversal_long_signal.astype(int)
    frame["reversal_short_signal"] = reversal_short_signal.astype(int)

    frame["long_signal"] = trend_long_signal.astype(int)
    frame["short_signal"] = trend_short_signal.astype(int)
    return frame


def _first_hit_index(mask: np.ndarray, fallback: int) -> np.ndarray:
    any_hit = mask.any(axis=1)
    first = np.argmax(mask, axis=1)
    return np.where(any_hit, first, fallback)


def label_trades(
    df,
    sl_mult: float = SL_ATR_MULT,
    tp_mult: float = TP_ATR_MULT,
    forward_bars: int = FORWARD_BARS,
    ema_fast: int = 8,
    ema_slow: int = 21,
) -> pd.DataFrame:
    """Forward-simulate labeled outcomes for signaled bars only.

    Output labels:
      +1 => TP hit before SL (win)
       0 => timeout/no-hit within forward window
      -1 => SL hit first
    """
    frame = compute_entry_signals(df.copy(), ema_fast=ema_fast, ema_slow=ema_slow)

    n = len(frame)
    labels = np.zeros(n, dtype=int)
    trade_dir = np.zeros(n, dtype=int)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)
    exit_bars = np.zeros(n, dtype=int)
    actual_rr = np.full(n, np.nan)

    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    atrs = frame["atr_14"].to_numpy(dtype=float)
    long_sig = frame["long_signal"].to_numpy()
    short_sig = frame["short_signal"].to_numpy()

    long_only = (long_sig == 1) & (short_sig == 0)
    short_only = (short_sig == 1) & (long_sig == 0)
    signaled_mask = long_only | short_only

    trade_dir[long_only] = 1
    trade_dir[short_only] = -1

    idx_all = np.flatnonzero(signaled_mask)
    if idx_all.size:
        valid_atr = np.isfinite(atrs[idx_all]) & (atrs[idx_all] > 0)
        idx = idx_all[valid_atr]

        if idx.size:
            direction = trade_dir[idx]
            entry = closes[idx]

            sl_dist = atrs[idx] * float(sl_mult)
            tp_dist = atrs[idx] * float(tp_mult)

            is_long = direction == 1
            sl = np.where(is_long, entry - sl_dist, entry + sl_dist)
            tp = np.where(is_long, entry + tp_dist, entry - tp_dist)

            sl_prices[idx] = sl
            tp_prices[idx] = tp

            fwd = int(forward_bars)

            high_pad = np.concatenate([highs[1:], np.full(fwd, np.nan)])
            low_pad = np.concatenate([lows[1:], np.full(fwd, np.nan)])
            close_pad = np.concatenate([closes[1:], np.full(fwd, np.nan)])

            high_w = np.lib.stride_tricks.sliding_window_view(high_pad, fwd)[idx]
            low_w = np.lib.stride_tricks.sliding_window_view(low_pad, fwd)[idx]
            close_w = np.lib.stride_tricks.sliding_window_view(close_pad, fwd)[idx]

            horizon = np.minimum(fwd, n - idx - 1)

            sl_hit = np.where(is_long[:, None], low_w <= sl[:, None], high_w >= sl[:, None])
            tp_hit = np.where(is_long[:, None], high_w >= tp[:, None], low_w <= tp[:, None])

            first_sl = _first_hit_index(sl_hit, fwd)
            first_tp = _first_hit_index(tp_hit, fwd)

            has_sl = first_sl < horizon
            has_tp = first_tp < horizon

            sl_first = has_sl & (~has_tp | (first_sl <= first_tp))
            tp_first = has_tp & (~has_sl | (first_tp < first_sl))
            timed_out = ~(sl_first | tp_first)

            labels[idx] = np.where(tp_first, 1, np.where(sl_first, -1, 0))
            exit_bars[idx] = np.where(sl_first, first_sl + 1, np.where(tp_first, first_tp + 1, horizon))

            timeout_pos = np.clip(horizon - 1, 0, None)
            timeout_exit = np.where(horizon > 0, close_w[np.arange(idx.size), timeout_pos], entry)
            exit_price = np.where(sl_first, sl, np.where(tp_first, tp, timeout_exit))

            denom = sl_dist
            pnl = np.where(is_long, exit_price - entry, entry - exit_price)
            actual_rr[idx] = np.divide(
                pnl,
                denom,
                out=np.full(idx.size, np.nan, dtype=float),
                where=denom > 0,
            )

    frame["label"] = labels
    frame["trade_dir"] = trade_dir
    frame["sl_price"] = sl_prices
    frame["tp_price"] = tp_prices
    frame["exit_bars"] = exit_bars
    frame["actual_rr"] = actual_rr

    signaled = ((frame["long_signal"] == 1) | (frame["short_signal"] == 1))
    return frame.loc[signaled].copy()


def print_label_stats(df):
    total = len(df)
    longs = int((df["trade_dir"] == 1).sum()) if "trade_dir" in df.columns else 0
    shorts = int((df["trade_dir"] == -1).sum()) if "trade_dir" in df.columns else 0
    long_wins = int(((df["trade_dir"] == 1) & (df["label"] == 1)).sum()) if "label" in df.columns else 0
    short_wins = int(((df["trade_dir"] == -1) & (df["label"] == 1)).sum()) if "label" in df.columns else 0
    timeouts = int((df["label"] == 0).sum()) if "label" in df.columns else 0

    long_wr = (long_wins / longs * 100.0) if longs else 0.0
    short_wr = (short_wins / shorts * 100.0) if shorts else 0.0
    timeout_pct = (timeouts / total * 100.0) if total else 0.0

    print(f"Total bars: {total:,}")
    print(f"Long signals: {longs:,} | Wins: {long_wins:,} | WR: {long_wr:.1f}%")
    print(f"Short signals: {shorts:,} | Wins: {short_wins:,} | WR: {short_wr:.1f}%")
    print(f"Timeouts: {timeouts:,} | Timeout %: {timeout_pct:.1f}%")

    if "exit_bars" in df.columns:
        mean_exit = pd.to_numeric(df["exit_bars"], errors="coerce")
        mean_exit = mean_exit[mean_exit > 0].mean()
        if pd.notna(mean_exit):
            print(f"Avg exit bars: {mean_exit:.1f}")


if __name__ == "__main__":
    print("Use label_trades(df) from this module in your training/data pipeline.")
