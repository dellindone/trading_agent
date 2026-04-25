# Model Lifecycle: Train, Promote, Rollback, Artifact Contracts

Last updated: April 25, 2026

---

## 1. NIFTY Lifecycle

### 1.1 Train (one-shot)

```bash
cd /Users/aditya/Desktop/chartflix/trading_agent
export PYTHONPATH=.

./.venv/bin/python scripts/train_model.py \
  --instrument NIFTY \
  --dataset /Users/aditya/Desktop/chartflix/trading_agent/data/nifty/datasets/NIFTY_features.parquet \
  --output /Users/aditya/Desktop/chartflix/trading_agent/core/model/artifacts
```

What happens internally (`NiftyTrainer.train()`):
1. Drop rows with NaN in critical target columns.
2. SHAP-based feature selection (`_select_top_features`) — prunes features by mean |SHAP| importance.
3. Build one pipeline per target head (direction classifier, three multiclass heads, one regressor) via `_build_direction_pipeline` / `_build_multiclass_pipeline` / `_build_regressor_pipeline`.
4. Walk-forward validation across N folds; metrics written per fold and aggregated.
5. Final model fit on full dataset.
6. Artifacts saved via `_save_artifacts`.

### 1.2 Research (multi-candidate)

```bash
./.venv/bin/python scripts/research_model.py \
  --instrument NIFTY \
  --dataset .../NIFTY_features.parquet \
  --output .../core/model/artifacts/research
```

`run_research()` trains multiple `ResearchCandidate` variants (different hyperparameters or feature sets) in parallel and writes `ResearchResult` summaries to the research output directory. Used to compare candidates before promoting one to production.

### 1.3 Weekly Retrain + Promotion Gate

```bash
./.venv/bin/python scripts/weekly_retrain.py \
  --instrument NIFTY \
  --dataset .../NIFTY_features.parquet \
  --output .../core/model/artifacts \
  --promote
```

Promotion logic in `weekly_retrain.py::main()`:
1. Load the current weighted F1 from `NIFTY_train_metrics.json` (`_load_previous_f1`).
2. Train a new model in a temp directory.
3. If new weighted F1 > old F1 + **0.005**, copy new artifacts to production via `_copy_tree`.
4. Otherwise, discard new artifacts and log "promotion gate not met."

Without `--promote`, training runs but artifacts are never written to the production directory.

### 1.4 ModelPromoter (shadow-performance gate)

`core/model/promoter.ModelPromoter` evaluates accumulated shadow-trade performance.  
Call via any script that loads journal data. Thresholds:

| Criterion | Threshold | Reason for failure label |
|---|---|---|
| `weeks_of_data` | ≥ 6 weeks | Insufficient history |
| `total_trades` | ≥ 50 closed | Too few trades for statistics |
| `win_rate` | ≥ 0.52 | Below break-even edge |
| `sharpe_ratio` (annualised) | ≥ 0.8 | Risk-adjusted return too low |
| `max_drawdown_pct` | ≤ 0.15 (15%) | Excessive capital drawdown |

Recommendation output:

| State | Recommendation |
|---|---|
| All pass | `PROMOTE` |
| Insufficient history or trades | `CONTINUE_SHADOW` |
| Enough data but poor metrics | `RETRAIN` |

`save_report()` writes a timestamped JSON to the output directory for audit.

### 1.5 Rollback (manual)

No automated rollback script. Recommended pattern:

```bash
# Backup before training
mkdir -p backups/nifty_artifacts_$(date +%Y%m%d_%H%M%S)
cp core/model/artifacts/NIFTY_* backups/nifty_artifacts_$(date +%Y%m%d_%H%M%S)/

# Restore
cp backups/<snapshot_dir>/NIFTY_* core/model/artifacts/
# Then restart the shadow engine.
```

---

## 2. BTC Lifecycle

### 2.1 Train

```bash
cd /Users/aditya/Desktop/chartflix/trading_agent
./.venv/bin/python run_train_btc.py              # full pipeline
./.venv/bin/python run_train_btc.py --skip-download   # reuse raw
./.venv/bin/python run_train_btc.py --skip-features   # reuse features
```

Training pipeline:
1. Collect Binance historical candles (multi-TF).
2. Build features per TF via `btc_agent/features.py::build_features()`.
3. Merge HTF features (15m, 45m, 1h) onto 1m spine (`HTF_FEATURES` list from `btc_agent/train.py`).
4. Label with `btc_agent/labeler.py::compute_entry_signals()`.
5. EMA pair search (fast/slow) to find highest-signal EMA combination.
6. Train base LightGBM binary classifier on all data.
7. Threshold search — find `best_threshold` maximising precision-recall balance on validation set.
8. Train per-regime models if regime has ≥ 500 samples (`MIN_REGIME_SAMPLES`).
9. Write artifacts.

### 2.2 Promotion

BTC currently promotes by overwrite: every successful `run_train_btc.py` writes directly to `data/btc/models/`.  
There is no promotion gate script — the shadow journal performance should be reviewed manually before retraining.

### 2.3 Rollback (manual)

```bash
mkdir -p backups/btc_models_$(date +%Y%m%d_%H%M%S)
cp data/btc/models/* backups/btc_models_$(date +%Y%m%d_%H%M%S)/

# Restore
cp backups/<snapshot_dir>/* data/btc/models/
# Then restart run_shadow_btc.py.
```

---

## 3. Artifact Contracts

### NIFTY artifacts (`core/model/artifacts/`)

Required for inference (all must exist):

| File | Purpose |
|---|---|
| `NIFTY_direction.joblib` | Direction classifier (XGBoost pipeline) |
| `NIFTY_sl_bin.joblib` | SL bucket classifier |
| `NIFTY_sl_bin_encoder.joblib` | LabelEncoder for sl_bin |
| `NIFTY_trail_bin.joblib` | Trail bucket classifier |
| `NIFTY_trail_bin_encoder.joblib` | LabelEncoder for trail_bin |
| `NIFTY_trail_tf.joblib` | Trail timeframe classifier |
| `NIFTY_trail_tf_encoder.joblib` | LabelEncoder for trail_tf |
| `NIFTY_phase1_target.joblib` | Phase-1 target regressor |
| `NIFTY_selected_features.json` | Ordered list of feature columns consumed by NiftyPredictor |

Metrics and audit:

| File | Purpose |
|---|---|
| `NIFTY_train_metrics.json` | Walk-forward accuracy, F1 per fold + aggregate; read by weekly_retrain |

Replace `NIFTY_` with `SENSEX_` for SENSEX instrument.  
`NiftyPredictor._validate_loaded()` raises `FileNotFoundError` on startup if any required file is absent.

### BTC artifacts (`data/btc/models/`)

Required:

| File | Purpose |
|---|---|
| `lgbm_signal_model.pkl` | Base LightGBM binary classifier |
| `model_meta.json` | `feature_cols`, `ema_fast`, `ema_slow`, `best_threshold`, `reverse_map`, `feature_stats` |

Optional regime overlays (used when regime has ≥ 500 training samples):

| File | Regime |
|---|---|
| `lgbm_signal_model_bull_normal.pkl` | Close > SMA-200, ATR normal |
| `lgbm_signal_model_bull_high_vol.pkl` | Close > SMA-200, ATR recent > 1.5× historical |
| `lgbm_signal_model_bear_normal.pkl` | Close ≤ SMA-200, ATR normal |
| `lgbm_signal_model_bear_high_vol.pkl` | Close ≤ SMA-200, ATR recent > 1.5× historical |

If a regime overlay is absent, `BtcSignalHandler` falls back to the base model for that regime.

---

## 4. Validation Checklist (both agents)

Run after any model promotion before returning to normal operation:

1. **File existence** — all required artifact files are present and have a recent modification timestamp.
2. **Boot check** — start the runtime; confirm no `FileNotFoundError` on model load.
3. **First inference** — wait for the first poll/candle boundary; confirm a heartbeat or eval summary is logged (not a crash).
4. **Alerts pipeline** — confirm Telegram start alert fires, or that the log shows "Telegram credentials not configured" (not a silent failure).
5. **Capital state** — confirm `CapitalTracker` initialises from existing history (not reset to `initial_capital`).
6. **Open trade restore** — if open trades existed before restart, confirm they are restored in shadow mode logs.
