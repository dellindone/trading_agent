"""Main 24/7 BTC shadow engine — tick-driven SL/TP, candle-boundary entry signals."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytz
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]          # .../trading_agent/
PACKAGE_ROOT = ROOT / "trading_agent"               # .../trading_agent/trading_agent/ (for core.*)
for _p in (str(ROOT), str(PACKAGE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load env from both common locations used in this repo layout.
load_dotenv(ROOT / ".env")
load_dotenv(PACKAGE_ROOT / ".env")

from btc_agent.btc_journal import BtcJournal, BtcTradeRecord
from btc_agent.btc_reporter import BtcReporter
from btc_agent.btc_shadow_mode import BtcShadowMode
from btc_agent.btc_signal_handler import BtcSignalHandler
from btc_agent.delta_client import delta_client
from btc_agent.features import build_features
from btc_agent.labeler import compute_entry_signals
from btc_agent.regime_classifier import add_regime_cols
from btc_agent.train import HTF_FEATURES
from core.risk.capital_tracker import CapitalTracker

logger = logging.getLogger(__name__)
USD_TO_INR = float(os.getenv("BTC_USD_INR", "85.0"))
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class TickSnapshot:
    price: float
    high: float
    low: float
    ts: datetime


class BtcEngine:
    BARS_1M = 500
    BARS_15M = 1200
    BARS_1H = 500

    def __init__(self, capital_inr: float, model_version: str):
        self.capital_inr = float(capital_inr)
        self.model_version = str(model_version)

        data_dir = Path(__file__).resolve().parents[1] / "data" / "btc"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.delta_client = delta_client
        self.signal_handler = BtcSignalHandler()
        self.journal = BtcJournal(data_dir=data_dir)
        self.signal_handler.rebuild_override_fuses(self.journal.load_closed_override_events(lookback_days=7))
        self.capital_tracker = CapitalTracker(data_dir=data_dir, initial_capital=self.capital_inr)
        self.shadow_mode = BtcShadowMode(journal=self.journal, capital_tracker=self.capital_tracker)
        self.shadow_mode.model_version = self.model_version
        self.reporter = BtcReporter(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        self.close_on_shutdown = str(os.getenv("BTC_CLOSE_ON_SHUTDOWN", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self._started_at: datetime | None = None
        self._last_hourly_key: str | None = None
        self._last_daily_summary_date: date | None = None
        self._running = True
        self._latest_atr: float = 0.0
        self._live_bar_key: str | None = None
        self._live_bar_high: float | None = None
        self._live_bar_low: float | None = None
        # Candle-boundary tracking: fire _poll() the moment a new 1m candle opens.
        self._last_candle_minute: int = -1
        self._poll_lock = threading.Lock()   # re-entrancy guard (WebSocket worker)
        # Live ticker display state.
        self._prev_tick_price: float = 0.0
        self._tick_count: int = 0
        self._last_eval_summary: str = "awaiting_first_candle"
        self._last_eval_data: dict = {}
        self._display_lock = threading.Lock()
        self._latest_display_price: float = 0.0
        self._latest_display_time: datetime | None = None
        self._last_rendered_eval_summary: str = ""
        self._last_render_ts: float = 0.0
        self._render_interval_sec: float = 0.20
        self._render_heartbeat_sec: float = 2.0
        self._display_line_count: int = 0
        # Keep websocket callback lightweight; heavy work runs off-thread.
        self._tick_work_lock = threading.Lock()
        self._tick_worker_running = False
        self._pending_tick: TickSnapshot | None = None
        self._sl_tp_candle_fallback_count: int = 0
        # (tf_str) -> (last_candle_timestamp | None, featured_df)
        self._feat_cache: dict[str, tuple[object, pd.DataFrame]] = {}

    def run(self):
        self._install_signal_handlers()
        renderer = threading.Thread(target=self._render_loop, name="btc-render", daemon=True)
        renderer.start()
        # Register tick callback BEFORE starting WS so no ticks are missed.
        self.delta_client.set_tick_callback(self._on_tick)
        self.delta_client.start_ws(symbols=["BTCUSDT"])
        self._started_at = datetime.now(UTC)
        self.reporter.send_engine_start_alert(self._started_at)
        self._bootstrap_from_history()

        # Housekeeping loop — heartbeats and daily summary only.
        # All SL/TP and entry decisions are driven by _on_tick() via WebSocket.
        while self._running:
            now = datetime.now(UTC)
            self._maybe_send_hourly_heartbeat(now)
            self._maybe_send_daily_summary(now)
            time.sleep(1.0)

    def _get_candles(self, tf: str, bars: int = 350) -> pd.DataFrame:
        """Return OHLCV candles.

        1m always uses REST — tick-aggregated 1m bars have gaps for quiet periods
        (no trades → bar dropped), which shifts RSI/MACD away from exchange values.
        HTF (15m+) uses tick-built candles when warm; gaps are negligible at those
        resolutions. Falls back to REST if tick buffer is not ready.
        """
        if tf != "1m":
            agg = self.delta_client.tick_aggregator
            if agg.has_enough_history(tf, min_bars=bars):
                candles = agg.get_candles(tf, bars=bars)
                if not candles.empty:
                    logger.debug("candles_source=tick tf=%s bars=%d", tf, len(candles))
                    return candles
        logger.debug("candles_source=rest tf=%s", tf)
        return self.delta_client.get_ohlcv(tf, bars=bars)

    def _get_features(self, raw: pd.DataFrame, tf: str) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        last_ts = raw.index[-1]
        cached = self._feat_cache.get(tf)
        if cached is not None and cached[0] == last_ts:
            return cached[1]
        feat = build_features(raw)
        if not feat.empty:
            self._feat_cache[tf] = (last_ts, feat)
        return feat

    # ------------------------------------------------------------------
    # Tick-driven core — called on every WebSocket trade tick
    # ------------------------------------------------------------------

    def _on_tick(self, price: float, size: float, ts_ms: float) -> None:
        """Called from the WebSocket thread on every trade tick.

        Responsibilities:
          1. Update live bar high/low with this tick.
          2. Check SL/TP for all open trades immediately (no 1-second lag).
          3. Detect new 1-minute candle boundary → trigger _poll() for entry signals.
        """
        if not self._running:
            return

        now = datetime.now(UTC)
        price = float(price)
        self._tick_count += 1

        # 0. Update display state; renderer thread owns terminal writes.
        with self._display_lock:
            self._latest_display_price = price
            self._latest_display_time = now

        # 1. Update live bar bounds tick-by-tick.
        self._update_live_bar_bounds(now, price)
        snap_high = self._live_bar_high if self._live_bar_high is not None else price
        snap_low = self._live_bar_low if self._live_bar_low is not None else price
        self._schedule_tick_work(price, snap_high, snap_low, now)

    def _render_loop(self) -> None:
        while self._running:
            with self._display_lock:
                price = self._latest_display_price
                now = self._latest_display_time

            if price > 0 and now is not None:
                self._print_live_ticker(price, now)
            time.sleep(self._render_interval_sec)

    def _schedule_tick_work(self, price: float, snap_high: float, snap_low: float, now: datetime) -> None:
        start_worker = False
        with self._tick_work_lock:
            if self._pending_tick is not None:
                snap_high = max(float(snap_high), float(self._pending_tick.high))
                snap_low = min(float(snap_low), float(self._pending_tick.low))
            self._pending_tick = TickSnapshot(price=float(price), high=float(snap_high), low=float(snap_low), ts=now)
            if not self._tick_worker_running:
                self._tick_worker_running = True
                start_worker = True

        if start_worker:
            worker = threading.Thread(
                target=self._tick_work_loop,
                name="btc-tick-work",
                daemon=True,
            )
            worker.start()

    def _tick_work_loop(self) -> None:
        while True:
            with self._tick_work_lock:
                pending = self._pending_tick
                self._pending_tick = None

            if pending is None:
                with self._tick_work_lock:
                    self._tick_worker_running = False
                return

            self._process_tick_work(pending)

    def _process_tick_work(self, snap: TickSnapshot) -> None:
        price = float(snap.price)
        snap_high = float(snap.high)
        snap_low = float(snap.low)
        now = snap.ts
        # 2. Check SL/TP using the latest known tick state.
        if self.shadow_mode.has_open_trades():
            atr = self._latest_atr if self._latest_atr > 0 else self.shadow_mode.get_any_open_atr()
            for tid, direction, sl in self.shadow_mode.open_trade_snapshots():
                would_hit = (snap_low <= sl) if direction == 1 else (snap_high >= sl)
                logger.debug(
                    "sl_check trade_id=%s dir=%d snap_low=%.2f snap_high=%.2f sl=%.2f hit=%s",
                    tid,
                    direction,
                    snap_low,
                    snap_high,
                    sl,
                    would_hit,
                )
            try:
                closed_ids = self.shadow_mode.tick(
                    current_price=price,
                    high=snap_high,
                    low=snap_low,
                    atr=float(atr),
                    current_time=now,
                )
                if closed_ids:
                    self._send_exit_alerts(closed_ids)
            except Exception as e:
                logger.warning("tick_sl_tp_error open=%s err=%s", self.shadow_mode.open_trade_ids(), e)

        # 3. Detect new 1m candle boundary and run entry signal evaluation.
        candle_minute = int(now.timestamp()) // 60
        if candle_minute != self._last_candle_minute:
            self._last_candle_minute = candle_minute
            if self._poll_lock.acquire(blocking=False):
                try:
                    self._poll()
                except Exception as e:
                    logger.exception("poll_failed: %s", e)
                finally:
                    self._poll_lock.release()

    def _print_live_ticker(self, price: float, now: datetime) -> None:
        now_ts = now.timestamp()
        summary_changed = self._last_eval_summary != self._last_rendered_eval_summary
        heartbeat_due = (now_ts - self._last_render_ts) >= self._render_heartbeat_sec
        price_changed = price != self._prev_tick_price
        if not summary_changed and not heartbeat_due and not price_changed:
            return

        arrow = "▲" if price > self._prev_tick_price else ("▼" if price < self._prev_tick_price else "─")
        with self._display_lock:
            tick_age_s = (now_ts - self._latest_display_time.timestamp()) if self._latest_display_time else 0.0
        stale = f"  [STALE {tick_age_s:.0f}s]" if tick_age_s > 10 else ""

        time_str = datetime.fromtimestamp(now_ts, tz=IST).strftime("%H:%M:%S IST")

        d = self._last_eval_data
        bull   = d.get("bull_score", "-")
        bear   = d.get("bear_score", "-")
        r1m    = d.get("rsi_1m", 0.0)
        r15m   = d.get("rsi_15m", 0.0)
        r45m   = d.get("rsi_45m", 0.0)
        atr    = d.get("atr_15m", 0.0)
        h15    = d.get("htf_15m", 0)
        h45    = d.get("htf_45m", 0)
        hgate  = d.get("htf_gate", 0)
        hgtf   = d.get("htf_gate_tf", "?")
        efst   = d.get("ema_fast", "?")
        eslw   = d.get("ema_slow", "?")
        lsig   = d.get("long_signal", 0)
        ssig   = d.get("short_signal", 0)
        code   = d.get("outcome_code", self._last_eval_summary)
        regime = d.get("regime", "?")
        ot     = d.get("open_trades", 0)

        capital = float(self.capital_tracker.current_capital)

        def _htf(v: int) -> str:
            return f"{v:+d}" if v != 0 else " 0"

        W = 63
        sep = "━" * W

        trade_lines: list[str] = []
        for snap in self.shadow_mode.open_trade_display_snapshots():
            direction = snap["direction"]
            entry     = snap["entry_price"]
            contracts = snap["contracts"]
            sl        = snap["current_sl"]
            tp        = snap["target_price"]
            gross_usd = (price - entry) * contracts * direction
            gross_inr = gross_usd * USD_TO_INR
            side      = "LONG" if direction == 1 else "SHORT"
            age_s     = (datetime.utcnow() - snap["entry_time"]).total_seconds()
            age_str   = f"{int(age_s // 60)}m{int(age_s % 60):02d}s"
            upnl_sign = "+" if gross_usd >= 0 else ""
            pct = (price - entry) / entry * direction * 100 if entry > 0 else 0.0
            trade_lines.append(sep)
            trade_lines.append(f"  TRADE    {side}   entry=${entry:,.2f}   SL=${sl:,.2f}   TP=${tp:,.2f}")
            trade_lines.append(
                f"  UNREALIZ {upnl_sign}${gross_usd:,.2f} ({upnl_sign}{pct:.2f}%)  "
                f"({upnl_sign}₹{gross_inr:,.0f})   {contracts:.4f} BTC   age {age_str}"
            )

        # PROJ block: always rendered for fixed line count (prevents separator
        # accumulation from variable-height redraws). Shows live levels when
        # confluence is met (*), placeholders otherwise.
        if not trade_lines:
            l_sl_d = d.get("long_sl_dist", 0.0)
            l_tp_d = d.get("long_tp_dist", 0.0)
            s_sl_d = d.get("short_sl_dist", 0.0)
            s_tp_d = d.get("short_tp_dist", 0.0)
            trade_lines.append(sep)
            if l_sl_d > 0:
                l_tag = "LIVE *" if lsig else "PROJ  "
                l_sl  = price - l_sl_d
                l_tp  = price + l_tp_d
                trade_lines.append(
                    f"  LONG  {l_tag}  SL~${l_sl:,.2f} (-{l_sl_d/price*100:.2f}%)"
                    f"  entry~${price:,.2f}"
                    f"  TP~${l_tp:,.2f} (+{l_tp_d/price*100:.2f}%)"
                )
            else:
                trade_lines.append(f"  LONG   ---  awaiting first candle")
            if s_sl_d > 0:
                s_tag = "LIVE *" if ssig else "PROJ  "
                s_sl  = price + s_sl_d
                s_tp  = price - s_tp_d
                trade_lines.append(
                    f"  SHORT {s_tag}  SL~${s_sl:,.2f} (+{s_sl_d/price*100:.2f}%)"
                    f"  entry~${price:,.2f}"
                    f"  TP~${s_tp:,.2f} (-{s_tp_d/price*100:.2f}%)"
                )
            else:
                trade_lines.append(f"  SHORT  ---  awaiting first candle")

        lines = [
            sep,
            f"  {arrow}  BTC  ${price:>12,.2f}    {time_str}    ticks: {self._tick_count:,}{stale}",
            f"  REGIME  {regime:<20}  OPEN  {ot}    CAP  ₹{capital:,.0f}",
            *trade_lines,
            sep,
            f"  SCORES   Bull {bull}/11   Bear {bear}/11",
            f"  RSI      1m {r1m:5.1f}   15m {r15m:5.1f}   45m {r45m:5.1f}    ATR 15m {atr:.0f}",
            f"  HTF      15m {_htf(h15)}   45m {_htf(h45)}   Gate {_htf(hgate)} ({hgtf})    EMA {efst}/{eslw}",
            f"  CONFLU   long={lsig}  short={ssig}  (pre-model flags)",
            f"  DECISION {code}",
            sep,
        ]

        if self._display_line_count > 0:
            sys.stdout.write(f"\033[{self._display_line_count}A\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        self._display_line_count = len(lines)
        self._last_render_ts = now_ts
        self._prev_tick_price = price
        self._last_rendered_eval_summary = self._last_eval_summary

    @staticmethod
    def _drop_incomplete_candles(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
        if df.empty:
            return df

        now = datetime.now(UTC)
        idx = df.index
        if getattr(idx, "tz", None) is None:
            now_ts = pd.Timestamp(now).tz_localize(None)
        else:
            now_ts = pd.Timestamp(now).tz_convert(idx.tz)

        tf_delta = pd.to_timedelta(int(tf_minutes), unit="m")
        complete_mask = (idx + tf_delta) <= now_ts
        return df.loc[complete_mask].copy()

    def _resample_45m_from_15m(self, raw_15m: pd.DataFrame) -> pd.DataFrame:
        """Build 45m candles from closed 15m bars (no lookahead)."""
        if raw_15m.empty:
            return pd.DataFrame()

        base = self._drop_incomplete_candles(raw_15m, tf_minutes=15)
        if base.empty:
            return pd.DataFrame()

        agg_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        cols = [c for c in agg_map if c in base.columns]
        if set(["open", "high", "low", "close"]) - set(cols):
            return pd.DataFrame()

        frame = (
            base[cols]
            .resample("45min", closed="left", label="left")
            .agg({k: v for k, v in agg_map.items() if k in cols})
            .dropna(subset=["open", "high", "low", "close"])
        )
        if frame.empty:
            return frame

        return self._drop_incomplete_candles(frame, tf_minutes=45)

    def _poll(self):
        # 1. Fetch candles for all TFs (tick-built if warm, REST if not).
        raw_1m = self._get_candles("1m", bars=self.BARS_1M)
        raw_15m = self._get_candles("15m", bars=self.BARS_15M)
        raw_1h = self._get_candles("1h", bars=self.BARS_1H)
        raw_45m = self._resample_45m_from_15m(raw_15m)
        if raw_1m.empty:
            logger.warning("poll_skipped: no_1m_candles")
            return
        closed_1m = self._drop_incomplete_candles(raw_1m, tf_minutes=1)
        if not closed_1m.empty:
            self._check_sl_tp_on_closed_candle(closed_1m.iloc[-1])

        # 2. Build features on CLOSED bars only so indicators match Delta Exchange UI.
        # The current partial bar is excluded — its RSI/MACD would reflect only
        # 1-2 ticks and diverge wildly from the last closed bar seen on the chart.
        # Entry price is always fetched live from get_btc_price() so execution
        # price is unaffected by this change.
        feat_1m  = self._get_features(closed_1m, "1m")
        feat_15m = self._get_features(raw_15m, "15m")
        feat_1h  = self._get_features(raw_1h,  "1h")
        feat_45m = self._get_features(raw_45m, "45m")
        if feat_1m.empty:
            logger.warning("poll_skipped: empty_1m_features")
            return

        # 3. Merge HTF context into 1m (forward-fill, no lookahead).
        merged = self._merge_htf_context(feat_1m, feat_15m, "15m")
        merged = self._merge_htf_context(merged, feat_1h, "1h")
        merged = self._merge_htf_context(merged, feat_45m, "45m")

        # 3b. Add regime columns (atr_14_recent / atr_14_hist / regime) so
        #     classify_regime() in the signal handler has the required columns.
        merged = add_regime_cols(merged)

        # 4. Compute confluence entry signals.
        merged = compute_entry_signals(
            merged,
            ema_fast=int(self.signal_handler.ema_fast),
            ema_slow=int(self.signal_handler.ema_slow),
        )
        if merged.empty:
            logger.warning("poll_skipped: empty_merged_frame")
            return

        # 5. Last row is live feature row.
        feature_row = merged.iloc[[-1]].copy()
        last = merged.iloc[-1]

        # 6. Fetch current BTC price.
        current_price = self.delta_client.get_btc_price()
        if current_price is None or float(current_price) <= 0:
            logger.warning("poll_skipped: missing_live_price")
            return
        current_price = float(current_price)

        # Extract confluence scores for logging.
        bull_score   = int(last.get("bull_score", 0) or 0)
        bear_score   = int(last.get("bear_score", 0) or 0)
        long_signal  = int(last.get("long_signal", 0) or 0)
        short_signal = int(last.get("short_signal", 0) or 0)
        override_candidate = bool(last.get("override", 0) or 0)
        htf_trend_15m = int(last.get("15m_smc_trend", 0) or 0)
        htf_trend_45m = int(last.get("45m_smc_trend", 0) or 0)
        has_45m_context = "45m_smc_trend" in merged.columns
        htf_gate_tf = "45m" if has_45m_context else "15m"
        htf_trend = htf_trend_45m if htf_gate_tf == "45m" else htf_trend_15m
        htf_source = "45m_primary" if has_45m_context else "15m_fallback(no_45m_features)"
        atr_15m = float(last.get("15m_atr_14", 0.0) or 0.0)
        rsi_1m  = float(last.get("rsi_14", 0.0) or 0.0)
        rsi_45m = float(last.get("45m_rsi_14", 0.0) or 0.0)
        rsi_15m = float(last.get("15m_rsi_14", 0.0) or 0.0)

        open_trades = int(len(self.journal.load_open_trades()))

        # 7. Infer signal.
        signal = self.signal_handler.process(
            feature_row=feature_row,
            current_price=current_price,
            capital_inr=float(self.capital_tracker.current_capital),
            open_trades=open_trades,
            override=override_candidate,
        )
        rejection_reason = self.signal_handler.last_rejection_reason
        drift_alerts = self.signal_handler.last_drift_alerts
        if drift_alerts:
            logger.warning("feature_drift_detected alerts=%s", drift_alerts[:5])

        # 8. Enter shadow trade + alert.
        trade = None
        if signal is not None:
            trade = self.shadow_mode.enter_trade(signal)
            if trade is not None:
                self.reporter.send_signal_alert(signal)
        if self.signal_handler.last_override_reached_model:
            self.signal_handler.note_override_result(
                executed=(signal is not None and trade is not None),
                model_rejected=(signal is None),
            )

        # 9. Cache ATR (prefer 15m ATR for meaningful SL distances) for live tick management.
        atr = float(last.get("15m_atr_14", 0.0) or last.get("atr_14", 0.0) or 0.0)
        self._latest_atr = atr if atr > 0 else self._latest_atr

        # 10. Detailed candle evaluation log.
        now = datetime.now(UTC)
        direction_str = "LONG" if long_signal else ("SHORT" if short_signal else "NONE")
        htf_str = "+1(bull)" if htf_trend == 1 else ("-1(bear)" if htf_trend == -1 else "0(neut)")

        outcome_code = "WAIT"
        if signal is not None:
            if trade is not None:
                outcome_code = f"FIRE:{'L' if signal.direction == 1 else 'S'} c={signal.confidence:.2f}"
                outcome = (
                    f"*** SIGNAL FIRED *** dir={('LONG' if signal.direction == 1 else 'SHORT')} "
                    f"conf={signal.confidence:.3f} sl={signal.sl_price:.2f} "
                    f"tp={signal.target_price:.2f} size={signal.contracts:.4f}BTC"
                )
            else:
                outcome_code = "ENTRY_BLOCKED"
                outcome = (
                    f"skip: ENTRY_BLOCKED (signal ready dir={('LONG' if signal.direction == 1 else 'SHORT')} "
                    f"conf={signal.confidence:.3f} but shadow entry was not accepted)"
                )
        elif long_signal == 0 and short_signal == 0:
            # Explain which side was closest and what's blocking it.
            closer = "bull" if bull_score >= bear_score else "bear"
            closer_score = bull_score if closer == "bull" else bear_score
            htf_block = (closer == "bull" and htf_trend == -1) or (closer == "bear" and htf_trend == 1)
            if htf_block:
                outcome_code = "HTF_BLOCK"
                outcome = f"skip: HTF_BLOCK ({htf_gate_tf}_smc_trend={htf_str} opposes {closer} 1m setup)"
            else:
                outcome_code = f"LOW:{closer[0]}{closer_score}"
                outcome = f"skip: SCORE_LOW ({closer}_score={closer_score}/11, need 6)"
        else:
            outcome_code = f"REJECT:{rejection_reason}"
            outcome = f"skip: MODEL_REJECT ({rejection_reason})"

        # 11. Compute projected SL/TP distances for BOTH directions so the display
        #     can always show where a long or short entry would land at current price.
        #     Uses structure-aware multipliers when OB/FVG is present; falls back to
        #     the signal handler's default multipliers otherwise.
        def _proj_dists(direction: int) -> tuple[float, float]:
            if atr_15m <= 0:
                return 0.0, 0.0
            is_rev = (
                (direction == 1 and int(last.get("reversal_long_signal", 0) or 0) == 1)
                or (direction == -1 and int(last.get("reversal_short_signal", 0) or 0) == 1)
            )
            mults = self.signal_handler._get_sl_tp_mults(last, is_rev, direction)
            if mults is not None:
                return float(atr_15m * mults[0]), float(atr_15m * mults[1])
            return (
                float(atr_15m * self.signal_handler.SL_ATR_MULT),
                float(atr_15m * self.signal_handler.TP_ATR_MULT),
            )

        long_sl_dist,  long_tp_dist  = _proj_dists(1)
        short_sl_dist, short_tp_dist = _proj_dists(-1)

        self._last_eval_summary = outcome_code
        self._last_eval_data = {
            "bull_score": bull_score,
            "bear_score": bear_score,
            "rsi_1m": rsi_1m,
            "rsi_15m": rsi_15m,
            "rsi_45m": rsi_45m,
            "atr_15m": atr_15m,
            "htf_15m": htf_trend_15m,
            "htf_45m": htf_trend_45m,
            "htf_gate": htf_trend,
            "htf_gate_tf": htf_gate_tf,
            "ema_fast": int(self.signal_handler.ema_fast),
            "ema_slow": int(self.signal_handler.ema_slow),
            "long_signal": long_signal,
            "short_signal": short_signal,
            "override_candidate": int(override_candidate),
            "outcome_code": outcome_code,
            "regime": self.signal_handler.last_regime or "?",
            "open_trades": open_trades,
            "long_sl_dist": long_sl_dist,
            "long_tp_dist": long_tp_dist,
            "short_sl_dist": short_sl_dist,
            "short_tp_dist": short_tp_dist,
        }

    def _check_sl_tp_on_closed_candle(self, candle_1m: pd.Series) -> None:
        """Fallback defense: if tick path misses, use closed 1m high/low to enforce exits."""
        if not self.shadow_mode.has_open_trades():
            return
        candle_high = float(candle_1m.get("high", 0.0) or 0.0)
        candle_low = float(candle_1m.get("low", 0.0) or 0.0)
        if candle_high <= 0 or candle_low <= 0:
            return

        atr = self._latest_atr if self._latest_atr > 0 else self.shadow_mode.get_any_open_atr()
        now = datetime.now(UTC)
        mid_price = (candle_high + candle_low) / 2.0
        closed_ids = self.shadow_mode.tick(
            current_price=float(mid_price),
            high=float(candle_high),
            low=float(candle_low),
            atr=float(atr),
            current_time=now,
        )
        if closed_ids:
            self._sl_tp_candle_fallback_count += len(closed_ids)
            logger.warning("sl_tp_candle_fallback_fired ids=%s total=%d", closed_ids, self._sl_tp_candle_fallback_count)
            self._send_exit_alerts(closed_ids)

    def _bootstrap_from_history(self) -> None:
        """Warm up features from historical candles and evaluate immediately on startup."""
        if not self._poll_lock.acquire(timeout=30):
            logger.warning("bootstrap_skipped: could not acquire poll lock")
            return
        try:
            self._poll()
            # Avoid duplicate immediate poll on the first tick in the same minute.
            self._last_candle_minute = int(datetime.now(UTC).timestamp()) // 60
            logger.info("bootstrap_ready: historical warmup complete")
        except Exception as e:
            logger.exception("bootstrap_failed: %s", e)
        finally:
            self._poll_lock.release()

    def _maybe_send_hourly_heartbeat(self, now: datetime):
        if now.minute != 0:
            return

        key = now.strftime("%Y-%m-%d-%H")
        if key == self._last_hourly_key:
            return

        btc_price = float(self.delta_client.get_btc_price() or 0.0)
        open_trades = int(len(self.journal.load_open_trades()))

        all_trades = self.journal.load_all()
        if all_trades.empty:
            signals_today = 0
        else:
            ts = pd.to_datetime(all_trades["timestamp_entry"], errors="coerce", utc=True)
            signals_today = int((ts.dt.date == now.date()).sum())

        started = self._started_at or now
        uptime_minutes = max(0, int((now - started).total_seconds() // 60))

        self.reporter.send_hourly_live_summary(
            btc_price=btc_price,
            open_trades=open_trades,
            signals_today=signals_today,
            capital_inr=float(self.capital_tracker.current_capital),
            uptime_minutes=uptime_minutes,
        )
        self._last_hourly_key = key

    def _maybe_send_daily_summary(self, now: datetime):
        if not (now.hour == 18 and now.minute == 30):
            return
        if self._last_daily_summary_date == now.date():
            return

        self.reporter.send_daily_summary(
            journal=self.journal,
            capital_inr=float(self.capital_tracker.current_capital),
        )
        self._last_daily_summary_date = now.date()

    def _send_exit_alerts(self, closed_ids: list[str]) -> None:
        for trade_id in closed_ids:
            override_event = self.journal.get_closed_override_event(trade_id)
            if override_event is not None:
                self.signal_handler.ingest_override_realized_r(override_event[0], override_event[1])
            record = self._get_record(trade_id)
            if record is not None:
                self.reporter.send_exit_alert(record, capital_inr=float(self.capital_tracker.current_capital))

    def _update_live_bar_bounds(self, now: datetime, price: float) -> None:
        key = f"{now.strftime('%Y-%m-%d-%H')}-{now.minute // 1}"
        if key != self._live_bar_key:
            self._live_bar_key = key
            self._live_bar_high = float(price)
            self._live_bar_low = float(price)
            return

        self._live_bar_high = max(float(self._live_bar_high), float(price)) if self._live_bar_high is not None else float(price)
        self._live_bar_low = min(float(self._live_bar_low), float(price)) if self._live_bar_low is not None else float(price)

    def _install_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, _frame):
        logger.info("shutdown signal received")
        self._running = False
        if self.close_on_shutdown:
            logger.info("close_on_shutdown enabled; force-closing open trades")
            self.shadow_mode.force_close_all(self.delta_client.get_btc_price() or 0.0, "SHUTDOWN")
        else:
            logger.info("preserving open trades on shutdown (BTC_CLOSE_ON_SHUTDOWN=false)")
        self.delta_client.stop_ws()
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def _merge_htf_context(df_15m: pd.DataFrame, df_htf: pd.DataFrame, tf: str) -> pd.DataFrame:
        if df_htf.empty:
            return df_15m

        cols = [c for c in HTF_FEATURES if c in df_htf.columns]
        if not cols:
            return df_15m

        htf = df_htf[cols].copy()
        htf.columns = [f"{tf}_{c}" for c in cols]

        merged = pd.merge_asof(
            df_15m.sort_index(),
            htf.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        prefixed = [f"{tf}_{c}" for c in cols]
        merged[prefixed] = merged[prefixed].ffill()
        return merged

    def _get_record(self, trade_id: str) -> BtcTradeRecord | None:
        df = self.journal.load_all()
        if df.empty:
            return None

        row_df = df[df["trade_id"].astype(str) == str(trade_id)]
        if row_df.empty:
            return None

        r = row_df.iloc[-1]
        entry_ts = pd.to_datetime(r.get("timestamp_entry"), errors="coerce")
        exit_ts = pd.to_datetime(r.get("timestamp_exit"), errors="coerce")

        return BtcTradeRecord(
            trade_id=str(r.get("trade_id")),
            symbol=str(r.get("symbol", "BTCUSDT")),
            timestamp_entry=entry_ts.to_pydatetime() if pd.notna(entry_ts) else datetime.now(UTC),
            timestamp_exit=exit_ts.to_pydatetime() if pd.notna(exit_ts) else None,
            direction=int(r.get("direction", 0)),
            entry_price=float(r.get("entry_price", 0.0) or 0.0),
            exit_price=float(r.get("exit_price", 0.0)) if pd.notna(r.get("exit_price")) else None,
            sl_price=float(r.get("sl_price", 0.0) or 0.0),
            target_price=float(r.get("target_price", 0.0) or 0.0),
            contracts=float(r.get("contracts", 0.0) or 0.0),
            confidence=float(r.get("confidence", 0.0) or 0.0),
            direction_prob=float(r.get("direction_prob", 0.0) or 0.0),
            atr_at_entry=float(r.get("atr_at_entry", 0.0) or 0.0),
            exit_reason=str(r.get("exit_reason")) if pd.notna(r.get("exit_reason")) else None,
            pnl_usd=float(r.get("pnl_usd", 0.0)) if pd.notna(r.get("pnl_usd")) else None,
            pnl_inr=float(r.get("pnl_inr", 0.0)) if pd.notna(r.get("pnl_inr")) else None,
            charges_usd=float(r.get("charges_usd", 0.0)) if pd.notna(r.get("charges_usd")) else None,
            model_version=str(r.get("model_version", self.model_version)),
            override=bool(r.get("override", False) or False),
            charges_inr=float(r.get("charges_inr")) if pd.notna(r.get("charges_inr")) else (float(r.get("charges_usd", 0.0)) * USD_TO_INR if pd.notna(r.get("charges_usd")) else None),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    BtcEngine(capital_inr=50000.0, model_version="v1.0").run()
