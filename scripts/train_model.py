"""CLI for Phase 4 model training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model.train import nifty_trainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 4 models for Nifty options agent")
    parser.add_argument("--instrument", required=True, help="Instrument name, e.g. NIFTY")
    parser.add_argument("--dataset", default=None, help="Optional dataset parquet path")
    parser.add_argument("--output", default=None, help="Optional artifacts output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--n-top-features", type=int, default=90, help="Top features to retain")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    instrument = args.instrument.upper()

    dataset_path = Path(args.dataset) if args.dataset else ROOT / "data" / "nifty" / "datasets" / f"{instrument}_features.parquet"
    output_dir = (
        Path(args.output)
        if args.output
        else ROOT / "trading_agent" / "core" / "model" / "artifacts"
    )

    result = nifty_trainer.train(
        dataset_path=dataset_path,
        instrument=instrument,
        output_dir=output_dir,
        n_folds=args.n_folds,
        n_top_features=args.n_top_features,
    )

    summary = {
        "instrument": result.instrument,
        "n_rows": result.n_rows,
        "n_features": result.n_features,
        "direction_accuracy": result.direction_metrics["accuracy"],
        "direction_f1": result.direction_metrics["f1_weighted"],
        "wf_mean_accuracy": result.walk_forward.mean_accuracy,
        "wf_mean_f1": result.walk_forward.mean_f1,
        "phase1_target_mae": result.direction_metrics["phase1_target_mae"],
        "artifacts_dir": result.artifacts_dir,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

