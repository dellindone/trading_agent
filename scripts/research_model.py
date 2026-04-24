"""CLI for Phase 4 model research candidate runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model.research import run_research

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 research candidates")
    parser.add_argument("--instrument", required=True, help="Instrument name, e.g. NIFTY")
    parser.add_argument("--dataset", default=None, help="Optional dataset parquet path")
    parser.add_argument("--output", default=None, help="Optional research output directory")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    instrument = args.instrument.upper()

    dataset_path = Path(args.dataset) if args.dataset else ROOT / "data" / "nifty" / "datasets" / f"{instrument}_features.parquet"
    output_dir = (
        Path(args.output)
        if args.output
        else ROOT / "trading_agent" / "core" / "model" / "artifacts" / "research"
    )

    ranked_results = run_research(
        dataset_path=dataset_path,
        instrument=instrument,
        output_base_dir=output_dir,
    )

    print(json.dumps([asdict(item) for item in ranked_results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

