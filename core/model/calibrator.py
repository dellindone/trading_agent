"""Probability calibration utilities for the direction model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def calibrate_direction(
    pipeline,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(estimator=pipeline, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def save_calibrated(calibrated_model: CalibratedClassifierCV, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, path)


def load_calibrated(path: Path) -> CalibratedClassifierCV:
    loaded = joblib.load(path)
    if not isinstance(loaded, CalibratedClassifierCV):
        raise TypeError(f"Expected CalibratedClassifierCV at {path}, got {type(loaded).__name__}")
    return loaded

