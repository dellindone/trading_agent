"""CLI for weekly retrain and optional promotion of Phase 4 model artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model.train import nifty_trainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly retrain for Phase 4 models")
    parser.add_argument("--instrument", required=True, help="Instrument name, e.g. NIFTY")
    parser.add_argument("--dataset", default=None, help="Optional dataset parquet path")
    parser.add_argument("--output", default=None, help="Optional artifacts output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--n-top-features", type=int, default=90, help="Top features to retain")
    parser.add_argument("--promote", action="store_true", help="Promote if F1 improves by > 0.005")
    return parser.parse_args()


def _load_previous_f1(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    value = data.get("direction_f1")
    return float(value) if value is not None else None


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


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

    run_dir = output_dir
    if args.promote:
        run_dir = output_dir / f"_candidate_{instrument}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result = nifty_trainer.train(
        dataset_path=dataset_path,
        instrument=instrument,
        output_dir=run_dir,
        n_folds=args.n_folds,
        n_top_features=args.n_top_features,
    )
    new_f1 = float(result.direction_metrics["f1_weighted"])

    decision = "trained_no_promotion"
    promoted = False
    previous_f1 = None

    if args.promote:
        metrics_path = output_dir / f"{instrument}_train_metrics.json"
        previous_f1 = _load_previous_f1(metrics_path)
        threshold = (previous_f1 if previous_f1 is not None else float("-inf")) + 0.005
        if new_f1 > threshold:
            _copy_tree(run_dir, output_dir)
            decision = "promoted"
            promoted = True
            logger.info(
                "Promoted new model for %s (new_f1=%.6f, previous_f1=%s)",
                instrument,
                new_f1,
                f"{previous_f1:.6f}" if previous_f1 is not None else "None",
            )
        else:
            decision = "skipped_promotion"
            logger.info(
                "Skipped promotion for %s (new_f1=%.6f, previous_f1=%s)",
                instrument,
                new_f1,
                f"{previous_f1:.6f}" if previous_f1 is not None else "None",
            )

    summary = {
        "instrument": result.instrument,
        "n_rows": result.n_rows,
        "n_features": result.n_features,
        "direction_accuracy": result.direction_metrics["accuracy"],
        "direction_f1": new_f1,
        "wf_mean_accuracy": result.walk_forward.mean_accuracy,
        "wf_mean_f1": result.walk_forward.mean_f1,
        "phase1_target_mae": result.direction_metrics["phase1_target_mae"],
        "artifacts_dir": result.artifacts_dir,
        "decision": decision,
        "promoted": promoted,
        "previous_direction_f1": previous_f1,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

