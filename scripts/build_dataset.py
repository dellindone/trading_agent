import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "nifty" / "datasets"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.features.engineering import feature_engineer
from core.features.labels import build_labels

logger = logging.getLogger(__name__)

MODEL_LABEL_COLUMNS = [
    "direction",
    "sl_bin",
    "phase1_target",
    "trail_bin",
    "trail_tf",
]

# Analysis-only diagnostics (not model targets). Keep separate to avoid
# accidental leakage into feature matrices during training.
ANALYSIS_LABEL_COLUMNS = [
    "adverse_excursion",
    "favorable_excursion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML-ready feature dataset for the trading agent")
    parser.add_argument("--instrument", required=True, help="Instrument name, e.g. NIFTY")
    parser.add_argument("--start", dest="start_date", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", dest="end_date", default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--output", default=None, help="Optional output parquet path")
    return parser.parse_args()


def resolve_output_path(instrument: str, output: str | None) -> Path:
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        return output_path
    return DATASET_DIR / f"{instrument.upper()}_features.parquet"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    instrument = args.instrument.upper()
    output_path = resolve_output_path(instrument, args.output)
    sample_path = output_path.with_name(f"{instrument}_features_sample.parquet")
    analysis_path = output_path.with_name(f"{output_path.stem}_label_analysis.parquet")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df = feature_engineer.build(
        instrument=instrument,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    label_df = build_labels(feature_df, horizon=3)

    model_dataset = feature_df.loc[label_df.index].join(label_df[MODEL_LABEL_COLUMNS])
    analysis_dataset = label_df.loc[model_dataset.index, ANALYSIS_LABEL_COLUMNS].copy()

    null_fraction = model_dataset.isna().mean(axis=1)
    model_dataset = model_dataset.loc[null_fraction <= 0.20].copy()
    analysis_dataset = analysis_dataset.loc[model_dataset.index]

    direction_distribution = model_dataset["direction"].value_counts(dropna=False).sort_index().to_dict()
    null_counts = model_dataset.isna().sum().sort_values(ascending=False)

    logger.info("Total rows: %s", len(model_dataset))
    logger.info("Direction label distribution: %s", direction_distribution)
    logger.info("Null counts per column:\n%s", null_counts.to_string())

    model_dataset.to_parquet(output_path)
    model_dataset.tail(500).to_parquet(sample_path)
    analysis_dataset.to_parquet(analysis_path)

    high_null_columns = model_dataset.columns[model_dataset.isna().mean() > 0.05].tolist()

    print(f"Rows: {len(model_dataset):,}")
    print(f"Features: {model_dataset.shape[1]}")
    print(f"Label distribution: direction -> {direction_distribution}")
    print(f"Nulls > 5%: {high_null_columns}")
    print(f"Saved to: {output_path}")
    print(f"Analysis labels saved to: {analysis_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
