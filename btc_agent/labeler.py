"""BTC trade signal scoring and forward-label simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
FORWARD_BARS = 60


def compute_entry_signals(df) -> pd.DataFrame:
    """Compute bullish/bearish confluence scores and binary entry flags."""
    frame = df.copy()

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

    # HTF direction filter: 15m longs only when 1h + 4h are not bearish,
    # 15m shorts only when 1h + 4h are not bullish.
    # Columns are present when this function is called after the HTF merge in
    # run_train_btc.py (Step 3.5) or after _merge_htf_context() in btc_engine.py.
    # 15m is the primary HTF gate — fast enough to give timely entries/exits.
    # 1h is intentionally excluded: it lags too much and blocks valid 15m setups.
    htf_bull_ok = pd.Series(True, index=frame.index)
    htf_bear_ok = pd.Series(True, index=frame.index)
    if "15m_smc_trend" in frame.columns:
        htf_bull_ok &= (frame["15m_smc_trend"] >= 0)   # not -1 (bearish)
        htf_bear_ok &= (frame["15m_smc_trend"] <= 0)   # not +1 (bullish)

    frame["long_signal"]  = ((bull_score >= 6) & htf_bull_ok).astype(int)
    frame["short_signal"] = ((bear_score >= 6) & htf_bear_ok).astype(int)
    return frame


def label_trades(
    df,
    sl_mult: float = SL_ATR_MULT,
    tp_mult: float = TP_ATR_MULT,
    forward_bars: int = FORWARD_BARS,
) -> pd.DataFrame:
    """Forward-simulate labeled outcomes for signaled bars only.

    Output labels:
      +1 => TP hit before SL (win)
      -1 => SL hit first OR timeout/no-hit within forward window (loss)
    """
    frame = compute_entry_signals(df.copy())

    n = len(frame)
    labels = np.zeros(n, dtype=int)
    trade_dir = np.zeros(n, dtype=int)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)
    exit_bars = np.zeros(n, dtype=int)
    actual_rr = np.full(n, np.nan)

    closes = frame["close"].to_numpy()
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    atrs = frame["atr_14"].to_numpy()
    long_sig = frame["long_signal"].to_numpy()
    short_sig = frame["short_signal"].to_numpy()

    for i in range(n):
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        direction = 0
        if long_sig[i] == 1 and short_sig[i] == 0:
            direction = 1
        elif short_sig[i] == 1 and long_sig[i] == 0:
            direction = -1
        else:
            continue

        trade_dir[i] = direction

        entry = closes[i]
        sl_dist = atr * float(sl_mult)
        tp_dist = atr * float(tp_mult)

        if direction == 1:
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        sl_prices[i] = sl
        tp_prices[i] = tp

        hit = 0
        max_j = min(i + int(forward_bars), n - 1)
        for j in range(i + 1, max_j + 1):
            if direction == 1:
                if lows[j] <= sl:
                    hit = -1
                    exit_bars[i] = j - i
                    actual_rr[i] = -1.0
                    break
                if highs[j] >= tp:
                    hit = 1
                    exit_bars[i] = j - i
                    actual_rr[i] = tp_mult / sl_mult
                    break
            else:
                if highs[j] >= sl:
                    hit = -1
                    exit_bars[i] = j - i
                    actual_rr[i] = -1.0
                    break
                if lows[j] <= tp:
                    hit = 1
                    exit_bars[i] = j - i
                    actual_rr[i] = tp_mult / sl_mult
                    break

        # No neutral class here: timeout/no-hit is a losing trade.
        labels[i] = 1 if hit == 1 else -1

    frame["label"] = labels
    frame["trade_dir"] = trade_dir
    frame["sl_price"] = sl_prices
    frame["tp_price"] = tp_prices
    frame["exit_bars"] = exit_bars
    frame["actual_rr"] = actual_rr

    # Keep only rows where confluence signaled an entry.
    signaled = ((frame["long_signal"] == 1) | (frame["short_signal"] == 1))
    return frame.loc[signaled].copy()


def print_label_stats(df):
    total = len(df)
    longs = int((df["trade_dir"] == 1).sum()) if "trade_dir" in df.columns else 0
    shorts = int((df["trade_dir"] == -1).sum()) if "trade_dir" in df.columns else 0
    long_wins = int(((df["trade_dir"] == 1) & (df["label"] == 1)).sum()) if "label" in df.columns else 0
    short_wins = int(((df["trade_dir"] == -1) & (df["label"] == 1)).sum()) if "label" in df.columns else 0

    long_wr = (long_wins / longs * 100.0) if longs else 0.0
    short_wr = (short_wins / shorts * 100.0) if shorts else 0.0

    print(f"Total bars: {total:,}")
    print(f"Long signals: {longs:,} | Wins: {long_wins:,} | WR: {long_wr:.1f}%")
    print(f"Short signals: {shorts:,} | Wins: {short_wins:,} | WR: {short_wr:.1f}%")

    if "exit_bars" in df.columns:
        mean_exit = pd.to_numeric(df["exit_bars"], errors="coerce")
        mean_exit = mean_exit[mean_exit > 0].mean()
        if pd.notna(mean_exit):
            print(f"Avg exit bars: {mean_exit:.1f}")


if __name__ == "__main__":
    print("Use label_trades(df) from this module in your training/data pipeline.")
