import logging

import pandas as pd

from core.data.multi_tf_builder import multi_tf_builder
from core.features.engineering import build_feature_frame
from core.features.labels import build_labels

logger = logging.getLogger(__name__)


def build_dataset(
    instrument: str = "NIFTY",
    days_to_expiry: int = 7,
    risk_free_rate: float = 6.5,
    dropna: bool = True,
) -> pd.DataFrame:
    frames = multi_tf_builder.build(instrument)
    if not frames:
        raise ValueError(f"No raw timeframe data available for instrument={instrument}")

    feature_frame = build_feature_frame(
        frames=frames,
        instrument=instrument,
        days_to_expiry=days_to_expiry,
        risk_free_rate=risk_free_rate,
    )
    label_frame = build_labels(feature_frame)

    dataset = feature_frame.join(label_frame)
    if dropna:
        dataset = dataset.dropna()

    logger.info("Built dataset for instrument=%s rows=%s columns=%s", instrument, len(dataset), len(dataset.columns))
    return dataset
