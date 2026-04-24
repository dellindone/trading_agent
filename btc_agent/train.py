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
        for tf in ["15m", "1h", "4h"]:
            feature_cols += [f"{tf}_{c}" for c in HTF_FEATURES]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = pd.to_numeric(df["label"], errors="coerce")

    # Binary target:
    #   +1 (TP-first win) -> 1
    #   -1 (SL/timeout loss) -> 0
    label_map = {-1: 0, 1: 1}
    y_encoded = y.map(label_map)

    return X, y_encoded, feature_cols, label_map


def walk_forward_train(df, n_splits=5) -> dict:
    """Run walk-forward LightGBM training and return best model + metrics."""
    df = df.sort_index()
    total_bars = len(df)
    fold_size = total_bars // (n_splits + 1)

    X, y, feature_cols, label_map = build_feature_matrix(df, include_htf=True)
    reverse_map = {v: k for k, v in label_map.items()}

    reports = []
    best_model = None
    best_score = -1.0
    best_threshold = 0.5

    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end
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

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
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
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        win_prob = model.predict_proba(X_test)[:, 1]
        fold_best_threshold = 0.5
        fold_best_acc = -1.0
        val_size = max(1, int(len(X_train) * 0.2))
        if val_size >= len(X_train):
            val_size = max(1, len(X_train) - 1)
        X_thresh_val = X_train.iloc[-val_size:]
        y_thresh_val = y_train.iloc[-val_size:]
        thresh_probs = model.predict_proba(X_thresh_val)[:, 1]

        y_thresh_val_np = y_thresh_val.to_numpy()
        for threshold in np.linspace(0.40, 0.69, 30):
            y_pred_try = (thresh_probs >= float(threshold)).astype(int)
            acc_try = float(accuracy_score(y_thresh_val_np, y_pred_try))
            if acc_try > fold_best_acc:
                fold_best_acc = acc_try
                fold_best_threshold = float(round(float(threshold), 2))

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

        if dir_acc > best_score:
            best_score = float(dir_acc)
            best_model = model
            best_threshold = float(fold_best_threshold)

    return {
        "model": best_model,
        "feature_cols": feature_cols,
        "label_map": label_map,
        "reverse_map": reverse_map,
        "reports": reports,
        "best_dir_accuracy": float(best_score if best_score >= 0 else 0.0),
        "best_threshold": float(best_threshold),
    }


def select_best_ema_pair(df: pd.DataFrame, n_splits: int = 5) -> tuple[tuple[int, int], list[dict], dict]:
    """Pick EMA fast/slow pair by walk-forward directional accuracy."""
    search_reports: list[dict] = []
    best_pair = (8, 21)
    best_score = -1.0
    best_result: dict | None = None

    for fast, slow in EMA_PAIR_CANDIDATES:
        tuned = apply_ema_pair_features(df.copy(), ema_fast=fast, ema_slow=slow)
        result = walk_forward_train(tuned, n_splits=n_splits)
        score = float(result.get("best_dir_accuracy", 0.0) or 0.0)
        search_reports.append({"ema_fast": int(fast), "ema_slow": int(slow), "best_dir_accuracy": score})
        if score > best_score:
            best_score = score
            best_pair = (int(fast), int(slow))
            best_result = result

    if best_result is None:
        tuned = apply_ema_pair_features(df.copy(), ema_fast=best_pair[0], ema_slow=best_pair[1])
        best_result = walk_forward_train(tuned, n_splits=n_splits)
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
        "best_threshold": float(result.get("best_threshold", 0.5) or 0.5),
        "walk_forward_reports": result["reports"],
        "ema_fast": int(ema_fast),
        "ema_slow": int(ema_slow),
        "ema_search_reports": ema_search_reports or [],
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def run_training(n_splits: int = 5, force_ema_search: bool = False) -> None:
    """Convenience full training flow using local BTC processed parquet."""
    path_1m = PROC_DIR / "BTCUSDT_1m_labeled.parquet"
    if not path_1m.exists():
        raise FileNotFoundError(f"Missing input: {path_1m}")

    df = pd.read_parquet(path_1m)
    df = merge_htf(df, "15m")
    df = merge_htf(df, "45m")
    df = merge_htf(df, "1h")
    df = merge_htf(df, "4h")
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
        best_pair, ema_search_reports, best_result = select_best_ema_pair(df, n_splits=n_splits)
    else:
        ema_search_reports = []
        tuned = apply_ema_pair_features(df.copy(), ema_fast=best_pair[0], ema_slow=best_pair[1])
        best_result = walk_forward_train(tuned, n_splits=n_splits)

    ema_fast, ema_slow = best_pair
    tuned_df = apply_ema_pair_features(df.copy(), ema_fast=ema_fast, ema_slow=ema_slow)

    result = best_result
    final_model = train_final_model(tuned_df, result["feature_cols"], result["label_map"])
    save_model(result, final_model, ema_fast=ema_fast, ema_slow=ema_slow, ema_search_reports=ema_search_reports)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(n_splits=5)
