"""BTCUSDT OHLCV data collection from Binance (bulk) + Delta (recent sync)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
DELTA_BASE = "https://api.delta.exchange"
SYMBOL_BINANCE = "BTCUSDT"
SYMBOL_DELTA = "BTCUSDT"
TIMEFRAMES = ["1m", "15m", "1h", "4h", "1d"]
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "btc" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_INTERVAL_SECONDS = {
    "1m": 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated Binance OHLCV fetch (1000 candles/request)."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_rows: list[list] = []
    cursor = int(start_ms)

    while cursor < int(end_ms):
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": int(end_ms),
            "limit": 1000,
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < 1000:
            break

        last_open_ms = int(rows[-1][0])
        cursor = last_open_ms + 1
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades"])

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df


def fetch_delta_candles(resolution: str = "15m", bars: int = 500) -> pd.DataFrame:
    """Fetch recent BTCUSDT candles from Delta Exchange."""
    if resolution not in _INTERVAL_SECONDS:
        raise ValueError(f"Unsupported resolution '{resolution}'. Allowed: {TIMEFRAMES}")

    bars = max(1, int(bars))
    end_ts = int(time.time())
    start_ts = int(end_ts - (bars * _INTERVAL_SECONDS[resolution]))

    url = f"{DELTA_BASE}/v2/history/candles"
    params = {
        "symbol": SYMBOL_DELTA,
        "resolution": resolution,
        "start": start_ts,
        "end": end_ts,
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    result = payload.get("result") or []
    if not result:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(result, dict):
        rows = result.get("candles") or result.get("result") or []
    else:
        rows = result

    normalized: list[dict] = []
    for item in rows:
        if isinstance(item, dict):
            ts = item.get("time") or item.get("timestamp") or item.get("t")
            row = {
                "timestamp": ts,
                "open": item.get("open") or item.get("o"),
                "high": item.get("high") or item.get("h"),
                "low": item.get("low") or item.get("l"),
                "close": item.get("close") or item.get("c"),
                "volume": item.get("volume") or item.get("v"),
            }
        elif isinstance(item, (list, tuple)) and len(item) >= 6:
            row = {
                "timestamp": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
            }
        else:
            continue
        normalized.append(row)

    if not normalized:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(normalized)

    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    # Convert ms epoch to seconds if needed.
    ts_num = ts_num.where(ts_num <= 10_000_000_000, ts_num // 1000)
    df["timestamp"] = pd.to_datetime(ts_num, unit="s", utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def collect_all_timeframes(years_back: int = 4):
    """Download all configured timeframes from Binance and save parquet files."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    outputs: dict[str, Path] = {}
    for tf in TIMEFRAMES:
        tf_years = 2 if tf == "1m" else years_back
        start_ms = int(end_ms - tf_years * 365 * 24 * 60 * 60 * 1000)
        df = fetch_binance_klines(SYMBOL_BINANCE, tf, start_ms, end_ms)
        out_path = DATA_DIR / f"BTCUSDT_{tf}.parquet"
        df.to_parquet(out_path)
        outputs[tf] = out_path
    return outputs


def sync_delta_candles():
    """Sync recent Delta candles into each timeframe parquet (deduplicated index)."""
    outputs: dict[str, Path] = {}
    for tf in TIMEFRAMES:
        delta_df = fetch_delta_candles(resolution=tf, bars=500)
        out_path = DATA_DIR / f"BTCUSDT_{tf}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            if "timestamp" in existing.columns:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
                existing = existing.dropna(subset=["timestamp"]).set_index("timestamp")
            else:
                if not isinstance(existing.index, pd.DatetimeIndex):
                    existing.index = pd.to_datetime(existing.index, utc=True, errors="coerce")
                existing = existing[~existing.index.isna()]

            merged = pd.concat([existing, delta_df], axis=0)
        else:
            merged = delta_df

        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.to_parquet(out_path)
        outputs[tf] = out_path

    return outputs


if __name__ == "__main__":
    collect_all_timeframes(years_back=4)
    sync_delta_candles()
