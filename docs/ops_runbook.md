# Ops Runbook: Alerts, Restart, Incident Handling, Logs

Last updated: April 25, 2026

---

## 1. Start / Stop Commands

### NIFTY shadow engine

```bash
cd /Users/aditya/Desktop/chartflix/trading_agent
export PYTHONPATH=.
./.venv/bin/python run_shadow.py --instrument NIFTY --log-level INFO
```

The engine self-gates on Indian market hours. Starting it outside market hours is safe — it will sleep until the next market window opens.

### BTC shadow engine

```bash
cd /Users/aditya/Desktop/chartflix/trading_agent
./.venv/bin/python run_shadow_btc.py --capital 20000 --model-version v1.0
```

`--capital` is the starting paper capital in INR (used only on first run; subsequent runs resume from persisted capital state).  
`--model-version` is a label string attached to every journal row.

### Stop

`Ctrl+C` on either process triggers the signal handler for graceful shutdown.

- NIFTY: `Engine._handle_shutdown_signal()` — completes the current poll cycle, runs end-of-day close if needed.
- BTC: `BtcEngine` shutdown — if `BTC_CLOSE_ON_SHUTDOWN=true`, force-closes all open positions before exit; otherwise leaves them for restoration on next start.

---

## 2. Alerting (Telegram)

Both agents use `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`. If either is absent, the reporter logs a warning and continues.

### NIFTY Reporter events

| Event | Trigger |
|---|---|
| Engine start | On `run_shadow.py` boot |
| Signal alert | New shadow trade entered |
| Exit alert | Shadow trade closed (SL / target / trailing / EOD) |
| Hourly heartbeat | Every clock hour while market is open |
| Daily summary | End of trading day |

### BTC Reporter events

| Event | Trigger |
|---|---|
| Engine start | `BtcEngine.run()` initialisation |
| Entry alert | `BtcShadowMode.enter_trade()` |
| Exit alert | SL/TP/force-close triggered |
| Hourly heartbeat | Every UTC clock hour |
| Daily summary | UTC midnight rollover |

---

## 3. Logging Surfaces

### Console

Both agents log structured lines to stdout/stderr. Key patterns to watch:

```bash
# NIFTY: look for signal events, poll summaries, errors
grep -E "signal|poll|ERROR|Exception|Traceback" /path/to/nifty_console.log

# BTC: look for tick-work results, eval summaries, errors
grep -E "eval_summary|signal|ERROR|Exception|Traceback" /path/to/btc_console.log
```

### Fyers API logs (NIFTY only)

```bash
cd /Users/aditya/Desktop/chartflix/trading_agent
rg -n "ERROR|Exception|failed|Traceback" fyersApi.log fyersRequests.log
```

### Journal files

Open/closed trade records are persisted as parquet:

| Agent | Journal path |
|---|---|
| NIFTY | `model_improver/data/trades.parquet` |
| BTC | `data/btc/btc_journal.parquet` |

Inspect with pandas:

```python
import pandas as pd
df = pd.read_parquet("data/btc/btc_journal.parquet")
print(df[df["exit_reason"].isna()])   # open trades
```

### Capital history

| Agent | Capital path |
|---|---|
| NIFTY | `model_improver/data/capital.parquet` |
| BTC | `data/btc/capital.parquet` |

`CapitalTracker.load_history()` returns a DataFrame with columns: `timestamp`, `capital`, `daily_pnl`, `cumulative_pnl`, `open_margin_used`, `event`.

---

## 4. Common Incidents + First Response

### A) NIFTY engine starts but no signals ever fire

1. Confirm it is a trading day and within market hours — check `is_trading_day()` logic in `core/utils/market_calendar.py`.
2. Confirm Fyers env vars are set and the access token is valid (check `fyersApi.log` for auth errors).
3. Verify all artifact files exist under `core/model/artifacts/NIFTY_*` (see `docs/model_lifecycle.md`).
4. Check logs for any of these rejection reasons in `SignalHandler.process()`:
   - `VIX > 30` — macro volatility block
   - `confidence < 0.60`
   - `RR < 1.5`
   - option chain or strike selection returning `None`

### B) NIFTY retrain fails with FileNotFoundError on dataset path

Known behaviour: if `--dataset` is omitted, the script uses a relative default that may not resolve in all shell environments. Always pass explicit absolute `--dataset` and `--output` paths.

### C) BTC engine running but no entries for extended periods

1. Check `last_rejection_reason` in log lines from `btc_signal_handler.py` — the most recent rejection is always logged at eval time.
2. Common rejections: `NO_DIRECTIONAL_CONFLUENCE` (no SMC signal fired), `CONFIDENCE_LOW`, `NO_STRUCTURE_CONFLUENCE` (no OB/FVG alignment), `FEE_UNVIABLE`.
3. Confirm Delta WebSocket is connected — check `btc_agent/delta_client.py` logs for reconnect loops or REST fallback warnings.
4. Confirm the engine is emitting hourly heartbeat Telegram messages or console heartbeat lines.

### D) BTC engine: stale ticker / no ticks arriving

1. Check Delta WebSocket status in logs — look for `ws_connect`, `ws_disconnect`, `ws_reconnect` events.
2. If REST fallback is the only source, confirm `delta_client.get_ohlcv()` returns non-empty frames.
3. Restart the engine — Delta client auto-reconnects on restart.

### E) Model load failure on startup

1. Confirm required files exist (see artifact contracts in `docs/model_lifecycle.md`).
2. `NiftyPredictor._validate_loaded()` raises `FileNotFoundError` listing the missing file.
3. `BtcSignalHandler.__init__` raises on missing model or metadata files.
4. Restore the last known-good backup snapshot, then restart.

### F) Capital mismatch or unexpected open trades after restart

Both agents persist state to parquet on every trade event — not just at shutdown.

- NIFTY: `ShadowMode._restore_open_trades()` replays open rows from `model_improver/data/trades.parquet`.
- BTC: `BtcShadowMode` restores from `data/btc/btc_journal.parquet`.
- `CapitalTracker` resumes from the last row of `capital.parquet` — it does **not** reset to `initial_capital` on restart.

If a mismatch occurs:
1. Inspect journal parquet for rows where `exit_reason` is null (still open).
2. Inspect capital history for the last snapshot event.
3. If state is corrupt, manually set the open rows to closed in the parquet and restart with a reconciled capital value.

### G) Drift alerts in BTC logs

`DriftMonitor` reports features with |z| > 4.0 vs training distribution. These are warnings only — they do not block trades. Persistent drift on many features signals that market conditions have shifted significantly and retraining should be considered.

---

## 5. Restart Procedure

1. Stop process gracefully with `Ctrl+C`.
2. Optionally backup current artifacts and journal (see `docs/model_lifecycle.md` rollback section).
3. Restart with the same command.
4. Verify:
   - No startup traceback in the first 10 seconds.
   - Telegram start alert received (or "credentials not configured" in logs — not a crash).
   - First heartbeat or eval summary line appears.
   - Capital tracker shows resumed balance, not initial capital.

---

## 6. Safe Rollback Procedure

1. Stop the runtime.
2. Restore previous artifact snapshot (NIFTY or BTC — see `docs/model_lifecycle.md`).
3. Restart the runtime.
4. Confirm: model loads, first inference tick/poll completes, no artifact-related errors.

---

## 7. Performance Spot-Check

Quick queries to assess current shadow performance:

```python
import pandas as pd, numpy as np

# BTC
df = pd.read_parquet("data/btc/btc_journal.parquet")
closed = df[df["exit_reason"].notna()]
print("Trades:", len(closed))
print("Win rate:", (closed["pnl_net"] > 0).mean().round(3))
print("Total PnL:", closed["pnl_net"].sum().round(2))

# NIFTY
df = pd.read_parquet("model_improver/data/trades.parquet")
closed = df[df["exit_premium"].notna()]
print("Trades:", len(closed))
print("Win rate:", (closed["pnl_net"] > 0).mean().round(3))
```

For promotion eligibility, run `ModelPromoter.evaluate()` from `core/model/promoter.py` against these DataFrames and the capital history (see `docs/model_lifecycle.md` section 1.4 for thresholds).

---

## 8. Escalation Checklist

Escalate (investigate root cause, do not just restart) when any of the following occur:

- Repeated Fyers auth failures across multiple sessions (likely token expiry or TOTP clock drift).
- Repeated process crashes on startup across multiple restart attempts.
- No heartbeat messages for > 1 hour while the process is confirmed running (possible silent deadlock in tick-work thread).
- Persistent model artifact load failures after artifact restoration (possible file corruption or Python version mismatch).
- Abnormal PnL pattern that doesn't match expected shadow trades (journal replay issue or capital tracker state divergence).
- `DriftMonitor` alerts on the majority of features for multiple consecutive candles (regime shift — retrain needed).
