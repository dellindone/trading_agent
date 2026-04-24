"""Main Phase 5 shadow-trading event loop for Nifty options agent."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "trading_agent"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from config.instruments import FYERS_SYMBOL
from core.data.fyers_client import fyers_client
from core.data.option_chain import option_chain_service
from core.features.engineering import build_feature_frame
from core.model.predict import nifty_predictor
from core.risk.capital_tracker import CapitalTracker
from core.utils.market_calendar import is_trading_day
from model_improver.journal import TradeRecord, TradeJournal
from model_improver.reporter import Reporter
from model_improver.shadow_mode import ShadowModeExecutor
from model_improver.signal_handler import SignalHandler

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class Engine:
    def __init__(self, instrument: str, artifacts_dir: str | Path) -> None:
        self.instrument = instrument.upper()
        self.artifacts_dir = self._resolve_path(artifacts_dir)
        self.data_dir = ROOT / "model_improver" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        nifty_predictor.load(self.artifacts_dir, self.instrument)
        self.predictor = nifty_predictor
        self.signal_handler = SignalHandler()
        self.journal = TradeJournal(self.data_dir)
        self.capital_tracker = CapitalTracker(data_dir=self.data_dir)
        self.shadow_mode = ShadowModeExecutor(
            journal=self.journal,
            capital_tracker=self.capital_tracker,
        )
        self.reporter = Reporter(
            journal=self.journal,
            capital_tracker=self.capital_tracker,
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        self._running = True
        self._eod_closed_on: str | None = None
        self._summary_sent_on: str | None = None
        self._started_at_ist: datetime | None = None
        self._last_hourly_heartbeat_key: str | None = None

    def run(self) -> None:
        self._install_signal_handlers()
        logger.info("Engine started for %s in SHADOW mode only", self.instrument)
        self._started_at_ist = datetime.now(IST)
        self.reporter.send_engine_start_alert(self.instrument, self._started_at_ist)

        while self._running:
            now_ist = datetime.now(IST)
            self._handle_schedule_tasks(now_ist)
            self._maybe_send_hourly_heartbeat(now_ist)

            if self._is_within_run_window(now_ist):
                if self._market_open(now_ist):
                    self._run_single_poll(now_ist)
                else:
                    self._log_poll(now_ist, "MARKET_CLOSED")

            self._sleep_until_next_five_minute_mark()

        logger.info("Engine stopped gracefully")

    def _run_single_poll(self, now_ist: datetime) -> None:
        try:
            frames = self._fetch_live_frames(now_ist)
            feature_frame = build_feature_frame(frames, instrument=self.instrument)
            if feature_frame.empty:
                self._log_poll(now_ist, "NO_FEATURE_ROW")
                return
            feature_row = feature_frame.iloc[[-1]].copy()

            prediction = self.predictor.predict(feature_row)
            trade_signal = self.signal_handler.process(
                prediction=prediction,
                feature_row=feature_row,
                instrument=self.instrument,
            )

            signal_label = "NONE"
            if trade_signal is not None:
                signal_label = trade_signal.option_type
                trade = self.shadow_mode.enter_trade(
                    trade_signal,
                )
                if trade is not None:
                    self.reporter.send_signal_alert(trade_signal)

            closed_ids: list[str] = []
            open_trades_df = self.journal.load_open_trades()
            if not open_trades_df.empty:
                for instrument_key in (
                    open_trades_df["instrument"].astype(str).str.upper().dropna().unique().tolist()
                ):
                    current_price = self._latest_option_price_fallback(instrument_key)
                    closed_ids.extend(
                        self.shadow_mode.tick(
                            instrument=instrument_key,
                            current_premium=float(current_price),
                            current_time=now_ist.astimezone(pytz.utc).replace(tzinfo=None),
                        )
                    )
            if closed_ids:
                all_trades = self.journal.load_all()
                for trade_id in closed_ids:
                    row = all_trades[all_trades["trade_id"].astype(str) == str(trade_id)]
                    if row.empty:
                        continue
                    r = row.iloc[-1].to_dict()

                    record = TradeRecord(
                        trade_id=str(r["trade_id"]),
                        instrument=str(r["instrument"]),
                        timestamp_entry=pd.to_datetime(r["timestamp_entry"]).to_pydatetime(),
                        timestamp_exit=pd.to_datetime(r["timestamp_exit"]).to_pydatetime()
                        if pd.notna(r["timestamp_exit"])
                        else None,
                        direction=int(r["direction"]),
                        strike=int(r["strike"]),
                        expiry_date=pd.to_datetime(r["expiry_date"]).date(),
                        option_type=str(r["option_type"]),
                        entry_premium=float(r["entry_premium"]),
                        exit_premium=float(r["exit_premium"]) if pd.notna(r["exit_premium"]) else None,
                        lot_size=int(r["lot_size"]),
                        sl_price=float(r["sl_price"]),
                        target_price=float(r["target_price"]),
                        trail_bin=str(r["trail_bin"]),
                        trail_tf=str(r["trail_tf"]),
                        confidence=float(r["confidence"]),
                        direction_prob=float(r["direction_prob"]),
                        exit_reason=str(r["exit_reason"]) if pd.notna(r["exit_reason"]) else None,
                        pnl_gross=float(r["pnl_gross"]) if pd.notna(r["pnl_gross"]) else None,
                        pnl_net=float(r["pnl_net"]) if pd.notna(r["pnl_net"]) else None,
                        charges=float(r["charges"]) if pd.notna(r["charges"]) else None,
                        vix_at_entry=float(r["vix_at_entry"]),
                        atr_at_entry=float(r["atr_at_entry"]),
                        model_version=str(r["model_version"]),
                    )
                    self.reporter.send_exit_alert(record)

            open_trades_count = len(self.journal.open_trades())
            self._log_poll(now_ist, signal_label, open_trades_count)
        except Exception as exc:
            logger.exception("poll_failed timestamp=%s error=%s", now_ist.isoformat(), exc)

    def _fetch_live_frames(self, now_ist: datetime) -> dict[str, pd.DataFrame]:
        frame_5m  = self._fetch_history(FYERS_SYMBOL[self.instrument], "5",  bars=200, days_back=30)
        frame_15m = self._fetch_history(FYERS_SYMBOL[self.instrument], "15", bars=200, days_back=60)
        frame_60m = self._fetch_history(FYERS_SYMBOL[self.instrument], "60", bars=200, days_back=99)
        frame_D   = self._fetch_history(FYERS_SYMBOL[self.instrument], "D",  bars=200, days_back=365)

        vix_5m = self._fetch_history("NSE:INDIAVIX-INDEX", "5", bars=400, days_back=30)
        if not vix_5m.empty and "close" in vix_5m.columns:
            left = frame_5m.sort_index().reset_index().rename(columns={"index": "timestamp"})
            right = (
                vix_5m[["close"]]
                .rename(columns={"close": "vix"})
                .sort_index()
                .reset_index()
                .rename(columns={"index": "timestamp"})
            )
            merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
            frame_5m = merged.set_index("timestamp").sort_index()

        frames = {
            "5":   frame_5m,
            "15":  frame_15m,
            "60":  frame_60m,
            "D":   frame_D,
            "5m":  frame_5m,
            "15m": frame_15m,
            "60m": frame_60m,
        }
        return frames

    def _fetch_history(self, symbol: str, resolution: str, *, bars: int, days_back: int) -> pd.DataFrame:
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days_back)
        candles = fyers_client.get_historical(
            symbol=symbol,
            resolution=resolution,
            date_from=start_date.strftime("%Y-%m-%d"),
            date_to=end_date.strftime("%Y-%m-%d"),
        )
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df.tail(bars).copy()

    def _handle_schedule_tasks(self, now_ist: datetime) -> None:
        date_key = now_ist.date().isoformat()

        if is_trading_day(now_ist.date()) and now_ist.hour == 15 and now_ist.minute >= 30 and self._eod_closed_on != date_key:
            try:
                current_premiums = self._current_premiums_for_open_trades()
                self.shadow_mode.force_close_all(
                    current_premiums=current_premiums,
                    reason="EOD",
                )
                self._eod_closed_on = date_key
            except Exception as exc:
                logger.exception("eod_close_failed timestamp=%s error=%s", now_ist.isoformat(), exc)

        if now_ist.hour == 15 and now_ist.minute >= 35 and self._summary_sent_on != date_key:
            try:
                self.reporter.send_daily_summary()
                self._summary_sent_on = date_key
            except Exception as exc:
                logger.exception("daily_summary_failed timestamp=%s error=%s", now_ist.isoformat(), exc)

    def _maybe_send_hourly_heartbeat(self, now_ist: datetime) -> None:
        if now_ist.minute != 0:
            return
        key = now_ist.strftime("%Y-%m-%d-%H")
        if self._last_hourly_heartbeat_key == key:
            return

        index_price = 0.0
        try:
            quote = fyers_client.get_quotes([FYERS_SYMBOL[self.instrument]])
            if quote:
                index_price = float(quote[0]["v"]["lp"])
        except Exception:
            pass

        total_signals_today = 0
        all_trades = self.journal.load_all()
        if not all_trades.empty and "timestamp_entry" in all_trades.columns:
            timestamps = pd.to_datetime(all_trades["timestamp_entry"], errors="coerce", utc=True)
            local_dates = timestamps.dt.tz_convert(IST).dt.date
            total_signals_today = int((local_dates == now_ist.date()).sum())

        open_trades = len(self.journal.load_open_trades())
        started = self._started_at_ist or now_ist
        uptime_minutes = max(0, int((now_ist - started).total_seconds() // 60))
        self.reporter.send_hourly_live_summary(
            instrument=self.instrument,
            open_trades=open_trades,
            total_signals_today=total_signals_today,
            capital=float(self.capital_tracker.current_capital),
            uptime_minutes=uptime_minutes,
            index_price=index_price,
        )
        self._last_hourly_heartbeat_key = key

    def _latest_option_price_fallback(self, instrument: str) -> float:
        open_trades = self.journal.load_open_trades()
        if not open_trades.empty:
            filtered = open_trades[open_trades["instrument"].astype(str).str.upper() == instrument.upper()]
            if not filtered.empty:
                trade = filtered.iloc[-1]
                live = self._latest_open_trade_ltp(trade)
                if live is not None and live > 0:
                    return float(live)
                return float(trade["entry_premium"])
        quote = fyers_client.get_quotes([FYERS_SYMBOL[self.instrument]])
        if quote:
            return float(quote[0]["v"]["lp"])
        return 0.0

    def _current_premiums_for_open_trades(self) -> dict[str, float]:
        premiums: dict[str, float] = {}
        open_trades_df = self.journal.load_open_trades()
        if open_trades_df.empty:
            premiums[self.instrument] = self._latest_option_price_fallback(self.instrument)
            return premiums
        for instrument_key in open_trades_df["instrument"].astype(str).str.upper().dropna().unique().tolist():
            premiums[instrument_key] = self._latest_option_price_fallback(instrument_key)
        return premiums

    def _latest_open_trade_ltp(self, trade_row: pd.Series) -> float | None:
        option_symbol = self._resolve_option_symbol(trade_row)
        if not option_symbol:
            return None
        return fyers_client.get_ltp(option_symbol)

    def _resolve_option_symbol(self, trade_row: pd.Series) -> str | None:
        instrument = str(trade_row.get("instrument", self.instrument)).upper()
        strike = int(float(trade_row.get("strike", 0)))
        option_type = str(trade_row.get("option_type", "")).upper()
        expiry_raw = trade_row.get("expiry_date")
        expiry_ts = pd.to_datetime(expiry_raw, errors="coerce")
        if pd.isna(expiry_ts) or strike <= 0 or option_type not in {"CE", "PE"}:
            return None
        expiry_date = expiry_ts.date()

        csv_df = option_chain_service._fetch_csv(instrument)
        if csv_df is None or csv_df.empty:
            return None

        # Column 9 = Fyers option symbol (e.g. "NSE:NIFTY2622APR25000CE")
        # Column 15 = strike price (numeric)
        try:
            strike_col = pd.to_numeric(csv_df[15], errors="coerce")
            strike_mask = strike_col == float(strike)
            type_mask = csv_df[9].astype(str).str.upper().str.endswith(option_type)

            # Match expiry: parse the date encoded in column 9 symbol string
            # Fyers format: NSE:NIFTY{YYDDMMM}{STRIKE}{TYPE}
            # Extract expiry by matching the expiry_date against what we can parse
            def _symbol_expiry_matches(sym: str) -> bool:
                try:
                    # Strip exchange prefix and underlying name then parse date
                    core = sym.split(":")[-1]  # e.g. "NIFTY2622APR25000CE"
                    name_len = len(instrument)
                    rest = core[name_len:]  # e.g. "2622APR25000CE"
                    # Try YYDDMMM format (e.g. "2622APR")
                    candidate = datetime.strptime(rest[:7], "%y%d%b").date()
                    return candidate == expiry_date
                except Exception:
                    return False

            expiry_mask = csv_df[9].astype(str).apply(_symbol_expiry_matches)
            matched = csv_df[strike_mask & type_mask & expiry_mask]
        except Exception:
            return None

        if matched.empty:
            return None
        symbol = str(matched.iloc[0][9])
        return symbol if symbol else None

    def _latest_vix_fallback(self) -> float:
        vix = fyers_client.get_vix()
        return float(vix) if vix is not None else 0.0

    def _market_open(self, now_ist: datetime) -> bool:
        if now_ist.weekday() >= 5:
            return False
        if not is_trading_day(now_ist.date()):
            return False
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now_ist <= market_close

    def _is_within_run_window(self, now_ist: datetime) -> bool:
        start = now_ist.replace(hour=9, minute=10, second=0, microsecond=0)
        end = now_ist.replace(hour=15, minute=35, second=59, microsecond=0)
        return start <= now_ist <= end and now_ist.weekday() < 5

    def _sleep_until_next_five_minute_mark(self) -> None:
        now_ist = datetime.now(IST)
        minute = now_ist.minute
        next_minute = ((minute // 5) + 1) * 5
        target = now_ist.replace(second=0, microsecond=0)
        if next_minute >= 60:
            target = (target + timedelta(hours=1)).replace(minute=0)
        else:
            target = target.replace(minute=next_minute)

        sleep_seconds = max(0.1, (target - now_ist).total_seconds())
        time.sleep(sleep_seconds)

    def _log_poll(self, now_ist: datetime, signal_label: str, open_trades_count: int | None = None) -> None:
        count = len(self.journal.open_trades()) if open_trades_count is None else int(open_trades_count)
        logger.info(
            "poll timestamp=%s signal=%s open_trades=%d",
            now_ist.isoformat(),
            signal_label,
            count,
        )

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum, _frame) -> None:
        logger.info("shutdown_signal_received signal=%s", signum)
        self._running = False

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (ROOT / path).resolve()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = Engine(instrument="NIFTY", artifacts_dir="trading_agent/core/model/artifacts")
    engine.run()
