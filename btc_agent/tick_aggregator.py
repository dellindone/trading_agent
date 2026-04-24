"""Builds OHLCV candles from raw trade ticks received via WebSocket."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone

import pandas as pd

_TF_SECONDS: dict[str, int] = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
}

# Keep up to 7 days of ticks at ~10 trades/sec = ~6M ticks max.
# 200_000 is plenty for candle building without blowing memory.
_MAX_TICKS = 200_000


class TickAggregator:
    """Thread-safe tick store that resamples into OHLCV candles on demand."""

    def __init__(self) -> None:
        self._ticks: deque[tuple[float, float, float]] = deque(maxlen=_MAX_TICKS)
        # Each entry: (unix_ts_seconds_float, price, size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path (called from WebSocket thread)
    # ------------------------------------------------------------------

    def add_tick(self, price: float, size: float, ts_ms: float) -> None:
        """Add a single trade tick.

        Args:
            price:  Trade price.
            size:   Trade size (contracts or BTC quantity).
            ts_ms:  Trade timestamp in milliseconds (Unix epoch).
        """
        ts_sec = float(ts_ms) / 1000.0 if float(ts_ms) > 1e10 else float(ts_ms)
        with self._lock:
            self._ticks.append((ts_sec, float(price), float(size)))

    # ------------------------------------------------------------------
    # Read path (called from engine / poll thread)
    # ------------------------------------------------------------------

    def get_candles(self, tf: str, bars: int) -> pd.DataFrame:
        """Return the last `bars` completed OHLCV candles for `tf`.

        The current (live, incomplete) candle is included as the last row
        so the engine always has a live feature row.

        Returns an empty DataFrame if fewer than 2 ticks are stored.
        """
        if tf not in _TF_SECONDS:
            raise ValueError(f"Unsupported timeframe '{tf}'. Choose from {list(_TF_SECONDS)}")

        tf_sec = _TF_SECONDS[tf]
        with self._lock:
            ticks = list(self._ticks)

        if len(ticks) < 2:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(ticks, columns=["ts", "price", "size"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("ts").sort_index()

        rule = _tf_to_pandas_rule(tf_sec)
        ohlcv = df["price"].resample(rule, closed="left", label="left").ohlc()
        ohlcv["volume"] = df["size"].resample(rule, closed="left", label="left").sum()
        ohlcv = ohlcv.dropna(subset=["open"])

        if ohlcv.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return ohlcv[["open", "high", "low", "close", "volume"]].tail(bars).copy()

    def tick_count(self) -> int:
        with self._lock:
            return len(self._ticks)

    def oldest_tick_age_seconds(self) -> float | None:
        with self._lock:
            if not self._ticks:
                return None
            oldest_ts = self._ticks[0][0]
        return datetime.now(timezone.utc).timestamp() - oldest_ts

    def has_enough_history(self, tf: str, min_bars: int = 100) -> bool:
        """True when the tick buffer covers at least min_bars of `tf`."""
        age = self.oldest_tick_age_seconds()
        if age is None:
            return False
        tf_sec = _TF_SECONDS.get(tf, 60)
        return age >= tf_sec * min_bars


def _tf_to_pandas_rule(tf_sec: int) -> str:
    """Convert timeframe in seconds to a pandas resample rule string."""
    minutes = tf_sec // 60
    if minutes < 60:
        return f"{minutes}min"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}D"
