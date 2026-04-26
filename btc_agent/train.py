"""BTC LightGBM training pipeline with walk-forward validation."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from btc_agent.labeler import apply_ema_pair_features
from btc_agent.regime_classifier import REGIMES, MIN_REGIME_SAMPLES, add_regime_cols

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "btc"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMA_PAIR_CANDIDATES = [
    (5, 15),
    (5, 13),
    (8, 21),
    (9, 15),
    (9, 21),
    (10, 30),
    (12, 26),
]

BASE_FEATURES = [
    "close_vs_ema8", "close_vs_ema21", "close_vs_sma50", "close_vs_sma200",
    "ema8_vs_ema21", "rsi_14", "macd", "macd_signal", "macd_hist",
    "atr_14", "body_size", "upper_wick", "lower_wick", "volume_ratio",
    "dist_to_resistance", "dist_to_support",
    "bull_bos", "bear_bos", "tl_bull_break", "tl_bear_break",
    "bull_choch", "bear_choch", "swing_bull_bos", "swing_bear_bos", "smc_trend",
    "in_bull_ob", "in_bear_ob", "bull_fvg", "bear_fvg",
    "bull_score", "bear_score", "is_bullish_candle",
    # candlestick patterns
    "body_pct", "range_pct",
    "is_doji", "is_hammer", "is_shooting_star",
    "is_engulfing_bull", "is_engulfing_bear", "is_pin_bar",
    "gap_up", "gap_down", "momentum_3", "momentum_5",
    # market microstructure
    "vwap_dev", "vol_delta_norm",
    # session context
    "session_asia", "session_london", "session_ny",
    "session_overlap", "session_open_impulse",
    # funding & OI (zero-filled when unavailable)
    "funding_rate_norm", "funding_rate_extreme",
    "oi_change_pct", "oi_impulse",
]

HTF_FEATURES = [
    "rsi_14", "macd_hist", "smc_trend", "ema8_vs_ema21",
    "close_vs_sma50", "close_vs_sma200", "bull_bos", "bear_bos",
    "bull_choch", "bear_choch", "tl_bull_break", "tl_bear_break",
    "volume_ratio", "atr_14",
]


def merge_htf(df_15m, tf) -> pd.DataFrame:
    """Merge higher-timeframe features into 15m frame without lookahead."""
    path = PROC_DIR / f"BTCUSDT_{tf}_labeled.parquet"
    if not path.exists():
        return df_15m.copy()

    df_htf = pd.read_parquet(path)
    cols = [c for c in HTF_FEATURES if c in df_htf.columns]
    if not cols:
        return df_15m.copy()

    df_htf = df_htf[cols].copy()
    df_htf.columns = [f"{tf}_{c}" for c in cols]

    # Drop pre-existing prefixed columns (e.g. added by Step 3.5) to avoid
    # merge_asof creating _x/_y suffixes when both sides share column names.
    prefixed = [f"{tf}_{c}" for c in cols]
    base = df_15m.drop(columns=[c for c in prefixed if c in df_15m.columns])

    merged = pd.merge_asof(
        base.sort_index(),
        df_htf.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    merged[prefixed] = merged[prefixed].ffill()
    return merged


def build_feature_matrix(df, include_htf=True) -> tuple:
    """Return X, y_encoded, feature_cols, label_map for binary win/loss."""
    feature_cols = BASE_FEATURES.copy()

    if include_htf:
        for tf in ["15m", "45m", "1h"]:
            feature_cols += [f"{tf}_{c}" for c in HTF_FEATURES]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = pd.to_numeric(df["label"], errors="coerce")

    # Binary target:
    #   +1 (TP-first win) -> 1
    #   -1/0 (SL/timeout loss) -> 0
    label_map = {-1: 0, 0: 0, 1: 1}  # timeout (0) treated as loss — conservative; no silent NaN drops
    y_encoded = y.map(label_map)

    return X, y_encoded, feature_cols, label_map


def walk_forward_train(df, n_splits=5, forward_bars: int = 360) -> dict:
    """Run walk-forward LightGBM training and return best model + metrics."""
    df = df.sort_index()
    total_bars = len(df)
    fold_size = total_bars // (n_splits + 1)

    X, y, feature_cols, label_map = build_feature_matrix(df, include_htf=True)
    reverse_map = {v: k for k, v in label_map.items()}

    reports = []
    fold_accuracies: list[float] = []
    fold_models: list[lgb.LGBMClassifier] = []
    fold_thresholds: list[float] = []
    best_model = None
    best_score = -1.0
    best_threshold = 0.5

    # Embargo is measured in labeled rows, not wall-clock bars; with sparse signals
    # this can span much longer calendar time than `forward_bars` suggests.
    EMBARGO = forward_bars  # default 360 to match labeler.py FORWARD_BARS

    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end + EMBARGO
        test_end = test_start + fold_size

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end].copy()
        X_test = X.iloc[test_start:test_end].copy()
        y_test = y.iloc[test_start:test_end].copy()

        train_mask = X_train.notna().all(axis=1) & y_train.notna()
        test_mask = X_test.notna().all(axis=1) & y_test.notna()
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]

        if X_train.empty or X_test.empty:
            reports.append(
                {
                    "fold": fold + 1,
                    "dir_accuracy": 0.0,
                    "best_threshold": 0.5,
                    "train_size": int(len(X_train)),
                    "test_size": int(len(X_test)),
                }
            )
            continue

        val_size = int(len(X_train) * 0.20)
        if val_size > 0:
            X_val = X_train.iloc[-val_size:].copy()
            y_val = y_train.iloc[-val_size:].copy()
            X_fit = X_train.iloc[:-val_size].copy()
            y_fit = y_train.iloc[:-val_size].copy()
        else:
            X_val = X_train.iloc[0:0].copy()
            y_val = y_train.iloc[0:0].copy()
            X_fit = X_train.copy()
            y_fit = y_train.copy()

        if X_fit.empty:
            X_fit = X_train.copy()
            y_fit = y_train.copy()
            X_val = X_train.iloc[0:0].copy()
            y_val = y_train.iloc[0:0].copy()

        pos = int((y_fit == 1).sum())
        neg = int((y_fit == 0).sum())
        scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=48,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            scale_pos_weight=float(scale_pos_weight),
        )

        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)] if not X_val.empty else [(X_fit, y_fit)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        fold_best_threshold = 0.5
        if not X_val.empty:
            val_prob = model.predict_proba(X_val)[:, 1]
            y_val_np = y_val.to_numpy()
            fold_best_acc = -1.0
            for threshold in np.linspace(0.40, 0.69, 30):
                y_pred_try = (val_prob >= float(threshold)).astype(int)
                acc_try = float(accuracy_score(y_val_np, y_pred_try))
                if acc_try > fold_best_acc:
                    fold_best_acc = acc_try
                    fold_best_threshold = float(round(float(threshold), 2))

        win_prob = model.predict_proba(X_test)[:, 1]
        y_test_np = y_test.to_numpy()
        y_pred = (win_prob >= fold_best_threshold).astype(int)
        # Requested metric: correct predictions / total signaled bars.
        total_tr = int(len(y_test))
        correct = int((y_pred == y_test_np).sum())
        dir_acc = (correct / total_tr) if total_tr > 0 else 0.0

        reports.append(
            {
                "fold": fold + 1,
                "dir_accuracy": float(dir_acc),
                "best_threshold": float(fold_best_threshold),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
            }
        )

        fold_accuracies.append(float(dir_acc))
        fold_models.append(model)
        fold_thresholds.append(float(fold_best_threshold))

    if fold_accuracies:
        acc_arr = np.asarray(fold_accuracies, dtype=float)
        mean_acc = float(np.mean(acc_arr))
        std_acc = float(np.std(acc_arr))
        rep_idx = int(np.argmin(np.abs(acc_arr - mean_acc)))
        best_model = fold_models[rep_idx]
        best_score = float(acc_arr[rep_idx])
        best_threshold = float(fold_thresholds[rep_idx])
        stable = bool((mean_acc - std_acc) > 0.50)
        if not stable:
            logging.warning(
                "Walk-forward stability gate failed: mean_acc=%.4f std_acc=%.4f (mean-std=%.4f <= 0.50). Returning best-available model.",
                mean_acc,
                std_acc,
                mean_acc - std_acc,
            )
    else:
        mean_acc = 0.0
        std_acc = 0.0
        stable = False
        logging.warning("No valid walk-forward folds produced a train/test model. Returning empty model result.")

    return {
        "model": best_model,
        "feature_cols": feature_cols,
        "label_map": label_map,
        "reverse_map": reverse_map,
        "reports": reports,
        "best_dir_accuracy": float(best_score if best_score >= 0 else 0.0),
        "best_threshold": float(best_threshold),
        "mean_dir_accuracy": float(mean_acc),
        "std_dir_accuracy": float(std_acc),
        "stable": bool(stable),
    }


def select_best_ema_pair(df: pd.DataFrame, n_splits: int = 5, forward_bars: int = 360) -> tuple[tuple[int, int], list[dict], dict]:
    """Pick EMA fast/slow pair by walk-forward directional accuracy."""
    search_reports: list[dict] = []
    best_pair = (8, 21)
    best_score = -1.0
    best_result: dict | None = None

    holdout_start = int(len(df) * 0.80)
    df_search = df.iloc[:holdout_start]
    df_holdout = df.iloc[holdout_start:]

    for fast, slow in EMA_PAIR_CANDIDATES:
        tuned = apply_ema_pair_features(df_search.copy(), ema_fast=fast, ema_slow=slow)
        result = walk_forward_train(tuned, n_splits=n_splits, forward_bars=forward_bars)
        score = float(result.get("mean_dir_accuracy", 0.0) or 0.0) - float(result.get("std_dir_accuracy", 0.0) or 0.0)
        search_reports.append(
            {
                "ema_fast": int(fast),
                "ema_slow": int(slow),
                "best_dir_accuracy": float(result.get("best_dir_accuracy", 0.0) or 0.0),
                "ema_selection_score": float(score),
                "holdout_dir_accuracy": None,
            }
        )
        if score > best_score:
            best_score = score
            best_pair = (int(fast), int(slow))
            best_result = result

    if best_result is None:
        tuned = apply_ema_pair_features(df_search.copy(), ema_fast=best_pair[0], ema_slow=best_pair[1])
        best_result = walk_forward_train(tuned, n_splits=n_splits, forward_bars=forward_bars)

    holdout_dir_accuracy = None
    best_model = best_result.get("model")
    if best_model is not None and not df_holdout.empty:
        tuned_holdout = apply_ema_pair_features(df_holdout.copy(), ema_fast=best_pair[0], ema_slow=best_pair[1])
        X_holdout, y_holdout, _, _ = build_feature_matrix(tuned_holdout, include_htf=True)
        mask = X_holdout.notna().all(axis=1) & y_holdout.notna()
        X_holdout = X_holdout.loc[mask]
        y_holdout = y_holdout.loc[mask]
        if not X_holdout.empty:
            win_prob = best_model.predict_proba(X_holdout)[:, 1]
            threshold = float(best_result.get("best_threshold", 0.5) or 0.5)
            y_pred = (win_prob >= threshold).astype(int)
            holdout_dir_accuracy = float(accuracy_score(y_holdout.to_numpy(), y_pred))

    best_result["holdout_dir_accuracy"] = holdout_dir_accuracy
    for row in search_reports:
        if row["ema_fast"] == best_pair[0] and row["ema_slow"] == best_pair[1]:
            row["holdout_dir_accuracy"] = holdout_dir_accuracy

    return best_pair, search_reports, best_result


def train_final_model(df, feature_cols, label_map) -> lgb.LGBMClassifier:
    """Train LightGBM on full dataset with n_estimators=1000."""
    X = df[feature_cols].copy()
    y = pd.to_numeric(df["label"], errors="coerce").map(label_map)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=48,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=float(scale_pos_weight),
    )

    model.fit(X, y)
    return model


def save_model(result, final_model, ema_fast: int = 8, ema_slow: int = 21, ema_search_reports: list[dict] | None = None):
    """Save model pickle and metadata json to MODEL_DIR."""
    model_path = MODEL_DIR / "lgbm_signal_model.pkl"
    meta_path = MODEL_DIR / "model_meta.json"

    with model_path.open("wb") as f:
        pickle.dump(final_model, f)

    meta = {
        "feature_cols": result["feature_cols"],
        "label_map": {str(k): v for k, v in result["label_map"].items()},
        "reverse_map": {str(k): v for k, v in result["reverse_map"].items()},
        "best_dir_accuracy": result["best_dir_accuracy"],
        "mean_dir_accuracy": float(result.get("mean_dir_accuracy", 0.0) or 0.0),
        "std_dir_accuracy": float(result.get("std_dir_accuracy", 0.0) or 0.0),
        "stable": bool(result.get("stable", False)),
        "holdout_dir_accuracy": result.get("holdout_dir_accuracy"),
        "best_threshold": float(result.get("best_threshold", 0.5) or 0.5),
        "walk_forward_reports": result["reports"],
        "ema_fast": int(ema_fast),
        "ema_slow": int(ema_slow),
        "ema_search_reports": ema_search_reports or [],
        "regime_models": result.get("regime_models", {}),
        "feature_stats": result.get("feature_stats", {}),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def run_training(n_splits: int = 5, force_ema_search: bool = False, forward_bars: int = 360) -> None:
    """Convenience full training flow using local BTC processed parquet."""
    path_1m = PROC_DIR / "BTCUSDT_1m_labeled.parquet"
    if not path_1m.exists():
        raise FileNotFoundError(f"Missing input: {path_1m}")

    df = pd.read_parquet(path_1m)
    df = merge_htf(df, "15m")
    df = merge_htf(df, "45m")
    df = merge_htf(df, "1h")
    meta_path = MODEL_DIR / "model_meta.json"
    best_pair: tuple[int, int] | None = None
    if not force_ema_search and meta_path.exists():
        try:
            with meta_path.open("r") as f:
                cached = json.load(f)
            if "ema_fast" in cached and "ema_slow" in cached:
                best_pair = (int(cached["ema_fast"]), int(cached["ema_slow"]))
        except Exception:
            best_pair = None

    if best_pair is None:
        best_pair, ema_search_reports, best_result = select_best_ema_pair(df, n_splits=n_splits, forward_bars=forward_bars)
    else:
        ema_search_reports = []
        tuned = apply_ema_pair_features(df.copy(), ema_fast=best_pair[0], ema_slow=best_pair[1])
        best_result = walk_forward_train(tuned, n_splits=n_splits, forward_bars=forward_bars)

    ema_fast, ema_slow = best_pair
    tuned_df = apply_ema_pair_features(df.copy(), ema_fast=ema_fast, ema_slow=ema_slow)
    tuned_df = add_regime_cols(tuned_df)

    result = best_result

    feat_cols_trained = [
        c for c in result.get("feature_cols", [])
        if c in tuned_df.columns
    ]
    if feat_cols_trained:
        X_stats = tuned_df[feat_cols_trained]
        feature_stats = {
            col: {
                "mean": float(X_stats[col].mean()),
                "std":  float(X_stats[col].std()),
            }
            for col in feat_cols_trained
            if X_stats[col].notna().any()
        }
    else:
        feature_stats = {}
        logging.warning(
            "feature_stats: no trained feature_cols found in tuned_df — "
            "drift monitor will be blind. Check labeled parquet schema."
        )

    if feature_stats:
        logging.info("feature_stats: computed stats for %d features", len(feature_stats))

    regime_model_meta: dict[str, dict[str, object]] = {}
    for regime in REGIMES:
        regime_df = tuned_df[tuned_df["regime"] == regime]
        sample_count = int(len(regime_df))
        if sample_count >= MIN_REGIME_SAMPLES:
            regime_model = train_final_model(regime_df, result["feature_cols"], result["label_map"])
            regime_model_path = MODEL_DIR / f"lgbm_signal_model_{regime}.pkl"
            with regime_model_path.open("wb") as f:
                pickle.dump(regime_model, f)
            regime_model_meta[regime] = {"available": True, "samples": sample_count}
        else:
            regime_model_meta[regime] = {"available": False, "samples": sample_count}

    result["regime_models"] = regime_model_meta
    result["feature_stats"] = feature_stats

    holdout_acc = float(best_result.get("holdout_dir_accuracy") or 0.0)
    mean_wf_acc = float(best_result.get("mean_dir_accuracy") or 0.0)
    if holdout_acc > 0.0:
        degradation = mean_wf_acc - holdout_acc
        if degradation > 0.05:
            logging.error(
                "holdout_gate: walk_forward_mean=%.4f holdout=%.4f "
                "degradation=%.4f exceeds 5%% threshold — model may overfit. "
                "Blocking save.",
                mean_wf_acc,
                holdout_acc,
                degradation,
            )
            return
        else:
            logging.info(
                "holdout_gate: PASSED walk_forward_mean=%.4f holdout=%.4f "
                "degradation=%.4f",
                mean_wf_acc,
                holdout_acc,
                degradation,
            )
    else:
        logging.warning("holdout_gate: holdout_dir_accuracy unavailable — skipping gate.")

    final_model = train_final_model(tuned_df, result["feature_cols"], result["label_map"])
    save_model(result, final_model, ema_fast=ema_fast, ema_slow=ema_slow, ema_search_reports=ema_search_reports)


def run_holdout_precision_check(threshold: float = 0.62) -> None:
    """Load saved model + holdout slice, report precision on class-1 (win).

    Uses the last 20% of the labeled 1m parquet as holdout (same split
    as select_best_ema_pair). Applies `threshold` to win_probability.
    """
    from sklearn.metrics import precision_score, classification_report

    path_1m = PROC_DIR / "BTCUSDT_1m_labeled.parquet"
    if not path_1m.exists():
        raise FileNotFoundError(f"Missing: {path_1m}")

    meta_path = MODEL_DIR / "model_meta.json"
    model_path = MODEL_DIR / "lgbm_signal_model.pkl"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError("model_meta.json or pkl missing — retrain first")

    with meta_path.open("r") as f:
        meta = json.load(f)
    with model_path.open("rb") as f:
        model = pickle.load(f)

    ema_fast = int(meta.get("ema_fast", 8) or 8)
    ema_slow = int(meta.get("ema_slow", 21) or 21)
    feature_cols = [str(c) for c in meta.get("feature_cols", [])]
    threshold = float(meta.get("best_threshold", threshold) or threshold)
    label_map = {-1: 0, 0: 0, 1: 1}

    df = pd.read_parquet(path_1m)
    df = merge_htf(df, "15m")
    df = merge_htf(df, "45m")
    df = merge_htf(df, "1h")

    holdout_start = int(len(df) * 0.80)
    df_holdout = df.iloc[holdout_start:].copy()
    df_holdout = apply_ema_pair_features(df_holdout, ema_fast=ema_fast, ema_slow=ema_slow)

    y = pd.to_numeric(df_holdout["label"], errors="coerce").map(label_map)
    X = df_holdout[[c for c in feature_cols if c in df_holdout.columns]].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    if X.empty:
        logging.warning("holdout_precision_check: no valid rows after NaN filter")
        return

    win_prob = model.predict_proba(X)[:, 1]
    y_pred = (win_prob >= threshold).astype(int)
    y_true = y.to_numpy().astype(int)

    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    signal_rate = float(y_pred.mean())

    print(f"\n=== Holdout Precision Check ===")
    print(f"Threshold : {threshold:.3f}")
    print(f"Samples   : {len(y_true):,}")
    print(f"Signal %  : {signal_rate * 100:.1f}%  ({int(y_pred.sum())} signals)")
    print(f"Precision (class-1 win rate): {precision * 100:.1f}%")
    print()
    print(classification_report(y_true, y_pred, target_names=["loss", "win"]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-holdout", action="store_true",
                        help="Run holdout precision check instead of training")
    parser.add_argument("--threshold", type=float, default=0.62)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_holdout:
        run_holdout_precision_check(threshold=args.threshold)
    else:
        run_training(n_splits=5)
