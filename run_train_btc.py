# python run_train_btc.py                  — full pipeline (download + train)
# python run_train_btc.py --skip-download  — skip steps 1-2, re-label + retrain only
# python run_train_btc.py --skip-features  — skip steps 1-3, only re-label + retrain
"""Run full BTC training pipeline end-to-end."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from btc_agent.data_collector import TIMEFRAMES, collect_all_timeframes, sync_delta_candles
from btc_agent.features import build_features
from btc_agent.labeler import label_trades, print_label_stats
from btc_agent.train import (
    DATA_DIR,
    HTF_FEATURES,
    PROC_DIR,
    save_model,
    train_final_model,
    walk_forward_train,
)


def _merge_htf_into_features(df_1m: pd.DataFrame, tf: str, processed_dir: Path) -> pd.DataFrame:
    """Merge higher-TF feature columns into the 1m frame (backward fill, no lookahead)."""
    htf_path = processed_dir / f"BTCUSDT_{tf}_features.parquet"
    if not htf_path.exists():
        print(f"  [skip HTF merge] missing {htf_path}")
        return df_1m

    df_htf = pd.read_parquet(htf_path)
    cols = [c for c in HTF_FEATURES if c in df_htf.columns]
    if not cols:
        return df_1m

    df_htf = df_htf[cols].copy()
    df_htf.columns = [f"{tf}_{c}" for c in cols]
    prefixed = [f"{tf}_{c}" for c in cols]

    # Drop any pre-existing prefixed columns to prevent merge_asof from
    # creating _x/_y suffixes when both sides share the same column names.
    base = df_1m.drop(columns=[c for c in prefixed if c in df_1m.columns])

    merged = pd.merge_asof(
        base.sort_index(),
        df_htf.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    merged[prefixed] = merged[prefixed].ffill()
    return merged


def run_pipeline(skip_download: bool = False, skip_features: bool = False) -> None:
    raw_dir = DATA_DIR / "raw"
    processed_dir = PROC_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("BTC Training Pipeline")
    print("=" * 68)

    if skip_features or skip_download:
        print("\n[Step 1/5] SKIPPED — using existing raw parquet files")
        print("[Step 2/5] SKIPPED — using existing raw parquet files")
    else:
        # Step 1
        print("\n[Step 1/5] collect_all_timeframes() — downloading ~4 years from Binance...")
        collect_all_timeframes(years_back=4)
        print(f"  done: raw parquet files in {raw_dir}")

        # Step 2
        print("\n[Step 2/5] sync_delta_candles() — syncing recent candles from Delta...")
        sync_delta_candles()
        print("  done: recent Delta candles merged into raw parquet files")

    if skip_features:
        print("\n[Step 3/5] SKIPPED — using existing feature parquet files")
    else:
        # Step 3
        print("\n[Step 3/5] build_features() for each timeframe...")
        for tf in TIMEFRAMES:
            in_path = raw_dir / f"BTCUSDT_{tf}.parquet"
            if not in_path.exists():
                print(f"  [skip] missing raw file: {in_path}")
                continue

            df = pd.read_parquet(in_path)
            feat_df = build_features(df)
            out_path = processed_dir / f"BTCUSDT_{tf}_features.parquet"
            feat_df.to_parquet(out_path)
            print(f"  [{tf}] features saved -> {out_path}")

    # Step 3.5: Merge 15m + 1h context into 1m BEFORE labeling so the labeler
    # can apply the HTF direction filter (long only when HTF bullish, short only when HTF bearish).
    print("\n[Step 3.5/5] Merging 15m + 1h context into 1m features for HTF direction filter...")
    path_1m_feat = processed_dir / "BTCUSDT_1m_features.parquet"
    if path_1m_feat.exists():
        df_1m_feat = pd.read_parquet(path_1m_feat)
        for htf in ["15m", "1h"]:
            df_1m_feat = _merge_htf_into_features(df_1m_feat, htf, processed_dir)
            print(f"  merged {htf} → 1m features (cols: {[c for c in df_1m_feat.columns if c.startswith(htf + '_')]})")
        df_1m_feat.to_parquet(path_1m_feat)
        print(f"  updated: {path_1m_feat}")
    else:
        print(f"  [skip] missing 1m features file: {path_1m_feat}")

    # Step 4
    print("\n[Step 4/5] label_trades() for each timeframe...")
    for tf in TIMEFRAMES:
        feat_path = processed_dir / f"BTCUSDT_{tf}_features.parquet"
        if not feat_path.exists():
            print(f"  [skip] missing features file: {feat_path}")
            continue

        feat_df = pd.read_parquet(feat_path)
        labeled_df = label_trades(feat_df)
        out_path = processed_dir / f"BTCUSDT_{tf}_labeled.parquet"
        labeled_df.to_parquet(out_path)
        print(f"  [{tf}] labels saved -> {out_path}")
        print_label_stats(labeled_df)

    # Step 5: 1m labeled already has correct 15m/1h columns from step 3.5.
    print("\n[Step 5/5] walk_forward_train() + train_final_model()...")
    path_1m = processed_dir / "BTCUSDT_1m_labeled.parquet"
    if not path_1m.exists():
        raise FileNotFoundError(f"Missing 1m labeled input: {path_1m}")

    df_1m = pd.read_parquet(path_1m)
    if len(df_1m) > 60:
        df_1m = df_1m.iloc[:-60]  # drop unlabeled forward-sim tail

    result = walk_forward_train(df_1m, n_splits=5)
    final_model = train_final_model(df_1m, result["feature_cols"], result["label_map"])
    save_model(result, final_model)

    print("\nTraining complete.")
    print(f"Model artifacts saved under: {DATA_DIR / 'models'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download steps (use existing raw parquet files)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip steps 1-3 entirely; only re-label + retrain (fastest iteration)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline(skip_download=args.skip_download, skip_features=args.skip_features)
