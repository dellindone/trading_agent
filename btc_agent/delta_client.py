"""Delta Exchange REST + WebSocket client for BTC market data."""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from typing import Iterable

import httpx
import pandas as pd

try:
    import websocket  # websocket-client
except Exception:  # pragma: no cover - optional dependency fallback
    websocket = None
try:
    import certifi
except Exception:  # pragma: no cover - optional dependency fallback
    certifi = None

DELTA_BASE = "https://api.delta.exchange"
DELTA_WS_URL = "wss://socket.delta.exchange"

logger = logging.getLogger(__name__)


class DeltaClient:
    _INTERVAL_SECONDS = {
        "1m": 60,
        "15m": 15 * 60,
        "1h": 60 * 60,
        "4h": 4 * 60 * 60,
        "1d": 24 * 60 * 60,
    }

    def __init__(self) -> None:
        from btc_agent.tick_aggregator import TickAggregator

        self._ws: websocket.WebSocketApp | None = None if websocket else None
        self._ws_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._ws_lock = threading.Lock()
        self._last_ws_price: float | None = None
        self._last_ws_symbol: str | None = None
        self._last_ws_ts: float = 0.0
        self._ws_symbols: list[str] = ["BTCUSD", "BTCUSDT"]
        self.tick_aggregator = TickAggregator()
        # Optional callback: called on every trade tick with (price, size, ts_ms).
        self._tick_callback: object = None

    def set_tick_callback(self, callback) -> None:
        """Register a callable(price, size, ts_ms) fired on every WS trade tick."""
        self._tick_callback = callback

    # --------------------------- Public API ---------------------------

    def start_ws(self, symbols: Iterable[str] | None = None) -> None:
        """Start websocket ticker stream (idempotent)."""
        if websocket is None:
            logger.warning("websocket-client not installed; using REST fallback only")
            return

        with self._ws_lock:
            if symbols is not None:
                cleaned = [str(s).strip().upper() for s in symbols if str(s).strip()]
                if cleaned:
                    self._ws_symbols = cleaned

            if self._ws_thread is not None and self._ws_thread.is_alive():
                return

            self._ws_stop.clear()
            self._ws_thread = threading.Thread(target=self._ws_loop, name="delta-ws", daemon=True)
            self._ws_thread.start()
            logger.info("delta websocket started symbols=%s", self._ws_symbols)

    def stop_ws(self) -> None:
        """Stop websocket ticker stream."""
        self._ws_stop.set()
        with self._ws_lock:
            ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def get_btc_price(self) -> float | None:
        """Return live BTC price (WS preferred, REST fallback)."""
        now = time.time()
        # Use websocket price if updated recently.
        if self._last_ws_price is not None and (now - self._last_ws_ts) <= 8.0:
            return float(self._last_ws_price)

        # Fallback REST (BTCUSDT ticker).
        url = f"{DELTA_BASE}/v2/tickers/BTCUSDT"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
            result = payload.get("result") or {}
            # Best mid (bid+ask)/2 is the most accurate real-time executable price.
            # Falls back to last trade (close), then mark_price.
            quotes = result.get("quotes") or {}
            best_bid = quotes.get("best_bid")
            best_ask = quotes.get("best_ask")
            if best_bid is not None and best_ask is not None:
                try:
                    return (float(best_bid) + float(best_ask)) / 2.0
                except Exception:
                    pass
            price_raw = (
                result.get("close")
                or result.get("last_traded_price")
                or result.get("mark_price")
            )
            return float(price_raw) if price_raw is not None else None
        except Exception as exc:
            logger.warning("delta REST ticker failed: %s", exc)
            return None

    def get_ohlcv(self, resolution: str = "15m", bars: int = 350) -> pd.DataFrame:
        """Return BTCUSDT OHLCV candles with UTC timestamp index."""
        if resolution not in self._INTERVAL_SECONDS:
            raise ValueError(f"Unsupported resolution '{resolution}'.")

        bars = max(int(bars), 1)
        end_ts = int(time.time())
        start_ts = end_ts - bars * self._INTERVAL_SECONDS[resolution]

        url = f"{DELTA_BASE}/v2/history/candles"
        params = {
            "symbol": "BTCUSDT",
            "resolution": resolution,
            "start": start_ts,
            "end": end_ts,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = payload.get("result") or []
        if isinstance(result, dict):
            candles = result.get("candles") or result.get("result") or []
        else:
            candles = result

        rows: list[dict] = []
        for item in candles:
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
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        ts = ts.where(ts <= 10_000_000_000, ts // 1000)
        df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]].tail(bars)

    # ------------------------ WebSocket internals ------------------------

    def _ws_loop(self) -> None:
        while not self._ws_stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    DELTA_WS_URL,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                )
                with self._ws_lock:
                    self._ws = ws
                sslopt = self._ws_sslopt()
                ws.run_forever(ping_interval=20, ping_timeout=10, sslopt=sslopt)
            except Exception as exc:
                logger.warning("delta websocket loop error: %s", exc)

            if not self._ws_stop.is_set():
                time.sleep(2.0)

    def _on_ws_open(self, ws) -> None:
        # Subscribe to ticker (for live price) + all_trades (for tick aggregation).
        payload = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "v2/ticker",  "symbols": self._ws_symbols},
                    {"name": "all_trades", "symbols": self._ws_symbols},
                ]
            },
        }
        ws.send(json.dumps(payload))
        logger.info("delta websocket subscribed channels=v2/ticker,all_trades symbols=%s", self._ws_symbols)

    def _on_ws_message(self, _ws, message: str) -> None:
        try:
            msg = json.loads(message)
        except Exception:
            return

        if not isinstance(msg, dict):
            return

        msg_type = msg.get("type", "")

        # --- v2/ticker: update live price cache ---
        if msg_type == "v2/ticker":
            symbol = str(msg.get("symbol") or msg.get("sy") or "").upper()
            if symbol and symbol not in {s.upper() for s in self._ws_symbols}:
                return

            # Prefer bid/ask mid → last trade (close) → mark_price.
            # This matches actual trade prices used in candles.
            quotes = msg.get("quotes") or {}
            best_bid = quotes.get("best_bid")
            best_ask = quotes.get("best_ask")
            if best_bid is not None and best_ask is not None:
                try:
                    raw_price = (float(best_bid) + float(best_ask)) / 2.0
                except Exception:
                    raw_price = None
            else:
                raw_price = None
            if raw_price is None:
                raw_price = (
                    msg.get("close")
                    or msg.get("last_traded_price")
                    or msg.get("mark_price")
                    or msg.get("c")
                    or msg.get("p")
                )
            if raw_price is None:
                return

            try:
                price = float(raw_price)
                self._last_ws_price = price
                self._last_ws_symbol = symbol or None
                self._last_ws_ts = time.time()
                if self._tick_callback is not None:
                    try:
                        self._tick_callback(price, 0.0, time.time() * 1000)
                    except Exception as cb_exc:
                        logger.debug("tick_callback error: %s", cb_exc)
            except Exception:
                pass

        # --- all_trades / all_trades_snapshot: feed tick aggregator ---
        elif msg_type in ("all_trades", "all_trades_snapshot"):
            symbol = str(msg.get("symbol") or "").upper()
            if symbol and symbol not in {s.upper() for s in self._ws_symbols}:
                return

            # Snapshot wraps a list of trades; individual messages are a single trade.
            if msg_type == "all_trades_snapshot":
                trade_list = msg.get("trades") or []
            else:
                trade_list = [msg]

            for trade in trade_list:
                raw_price = trade.get("price") or trade.get("p")
                raw_size  = trade.get("size")  or trade.get("q") or trade.get("s") or 0
                raw_ts    = trade.get("timestamp") or trade.get("created_at") or trade.get("t")

                try:
                    price = float(raw_price)
                    size  = float(raw_size)
                    # Delta Exchange timestamps are microseconds (16-digit).
                    # Convert μs → ms for TickAggregator.
                    if raw_ts is not None:
                        raw_ts_f = float(raw_ts)
                        ts_ms = raw_ts_f / 1000.0 if raw_ts_f > 1e12 else raw_ts_f
                    else:
                        ts_ms = time.time() * 1000
                    self.tick_aggregator.add_tick(price=price, size=size, ts_ms=ts_ms)
                    # Keep live price in sync from trades too.
                    self._last_ws_price = price
                    self._last_ws_ts = time.time()
                    if self._tick_callback is not None:
                        try:
                            self._tick_callback(price, size, ts_ms)
                        except Exception as cb_exc:
                            logger.debug("tick_callback error: %s", cb_exc)
                except Exception:
                    continue

    def _on_ws_error(self, _ws, error) -> None:
        logger.warning("delta websocket error: %s", error)

    def _on_ws_close(self, _ws, close_status_code, close_msg) -> None:
        logger.info("delta websocket closed status=%s msg=%s", close_status_code, close_msg)

    @staticmethod
    def _ws_sslopt() -> dict:
        if certifi is not None:
            return {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": certifi.where()}
        return {"cert_reqs": ssl.CERT_REQUIRED}


delta_client = DeltaClient()
