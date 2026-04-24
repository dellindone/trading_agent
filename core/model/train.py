"""Phase 4 model training for the Nifty options trading agent."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    import shap
except ImportError:  # optional dependency
    shap = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # optional dependency
    XGBClassifier = None
    XGBRegressor = None

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMNS = [
    "direction",
    "sl_bin",
    "trail_bin",
    "phase1_target",
    "trail_tf",
    "adverse_excursion",
    "favorable_excursion",
    "instrument",
]


@dataclass
class WalkForwardFold:
    fold: int
    train_size: int
    val_size: int
    direction_accuracy: float
    direction_f1: float


@dataclass
class WalkForwardResult:
    n_folds: int
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    folds: list[WalkForwardFold] = field(default_factory=list)


@dataclass
class TrainResult:
    instrument: str
    n_rows: int
    n_features: int
    selected_features: list[str]
    walk_forward: WalkForwardResult
    direction_metrics: dict
    artifacts_dir: str


class NiftyTrainer:
    def __init__(
        self,
        *,
        xgb_n_estimators: int = 500,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.03,
    ) -> None:
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate

    def train(
        self,
        dataset_path: str | Path,
        instrument: str,
        output_dir: str | Path,
        n_folds: int = 5,
        n_top_features: int = 90,
        test_fraction: float = 0.15,
    ) -> TrainResult:
        self._validate_dependencies()

        instrument_key = instrument.upper()
        dataset = pd.read_parquet(Path(dataset_path)).sort_index()
        dataset = self._drop_invalid_rows(dataset)

        all_features = self._feature_columns(dataset)
        if len(all_features) == 0:
            raise ValueError("No numeric feature columns available after exclusions.")

        selected_features = self._select_top_features(
            dataset=dataset,
            feature_columns=all_features,
            n_top_features=n_top_features,
        )

        walk_forward = self._walk_forward_validate(
            dataset=dataset,
            feature_columns=selected_features,
            n_folds=n_folds,
            min_train=500,
        )

        split_idx = int(len(dataset) * (1.0 - test_fraction))
        if split_idx <= 0 or split_idx >= len(dataset):
            raise ValueError("Invalid test_fraction for dataset size.")

        train_df = dataset.iloc[:split_idx].copy()
        test_df = dataset.iloc[split_idx:].copy()

        x_train = train_df[selected_features]
        x_test = test_df[selected_features]

        direction_pipeline = self._build_direction_pipeline()
        direction_pipeline.fit(x_train, train_df["direction"].astype(int))
        direction_pred = direction_pipeline.predict(x_test)
        direction_acc = float(accuracy_score(test_df["direction"].astype(int), direction_pred))
        direction_f1 = float(f1_score(test_df["direction"].astype(int), direction_pred, average="weighted"))

        sl_encoder = LabelEncoder()
        y_sl = sl_encoder.fit_transform(train_df["sl_bin"].astype(str))
        sl_pipeline = self._build_multiclass_pipeline(num_class=len(sl_encoder.classes_))
        sl_pipeline.fit(x_train, y_sl)

        trail_bin_encoder = LabelEncoder()
        y_trail_bin = trail_bin_encoder.fit_transform(train_df["trail_bin"].astype(str))
        trail_bin_pipeline = self._build_multiclass_pipeline(num_class=len(trail_bin_encoder.classes_))
        trail_bin_pipeline.fit(x_train, y_trail_bin)

        trail_tf_encoder = LabelEncoder()
        y_trail_tf = trail_tf_encoder.fit_transform(train_df["trail_tf"].astype(str))
        trail_tf_pipeline = self._build_multiclass_pipeline(num_class=len(trail_tf_encoder.classes_))
        trail_tf_pipeline.fit(x_train, y_trail_tf)

        phase1_pipeline = self._build_regressor_pipeline()
        phase1_pipeline.fit(x_train, pd.to_numeric(train_df["phase1_target"], errors="coerce"))
        phase1_pred = phase1_pipeline.predict(x_test)
        phase1_mae = float(
            mean_absolute_error(pd.to_numeric(test_df["phase1_target"], errors="coerce"), phase1_pred)
        )

        artifacts_dir = Path(output_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._save_artifacts(
            artifacts_dir=artifacts_dir,
            instrument=instrument_key,
            direction_pipeline=direction_pipeline,
            sl_pipeline=sl_pipeline,
            sl_encoder=sl_encoder,
            trail_bin_pipeline=trail_bin_pipeline,
            trail_bin_encoder=trail_bin_encoder,
            trail_tf_pipeline=trail_tf_pipeline,
            trail_tf_encoder=trail_tf_encoder,
            phase1_pipeline=phase1_pipeline,
            selected_features=selected_features,
            direction_acc=direction_acc,
            direction_f1=direction_f1,
            phase1_mae=phase1_mae,
            walk_forward=walk_forward,
        )

        direction_metrics = {
            "accuracy": direction_acc,
            "f1_weighted": direction_f1,
            "phase1_target_mae": phase1_mae,
        }

        return TrainResult(
            instrument=instrument_key,
            n_rows=int(len(dataset)),
            n_features=int(len(selected_features)),
            selected_features=selected_features,
            walk_forward=walk_forward,
            direction_metrics=direction_metrics,
            artifacts_dir=str(artifacts_dir),
        )

    def _validate_dependencies(self) -> None:
        if XGBClassifier is None or XGBRegressor is None:
            raise RuntimeError(
                "xgboost is required for training. Install with: pip install xgboost"
            )

    def _drop_invalid_rows(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required_targets = ["direction", "sl_bin", "trail_bin", "phase1_target", "trail_tf"]
        clean = dataset.dropna(subset=required_targets).copy()
        return clean.sort_index()

    def _feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        exclude = set(TARGET_COLUMNS)
        return [column for column in numeric_cols if column not in exclude]

    def _build_direction_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        n_estimators=self.xgb_n_estimators,
                        max_depth=self.xgb_max_depth,
                        learning_rate=self.xgb_learning_rate,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def _build_multiclass_pipeline(self, num_class: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=num_class,
                        n_estimators=max(100, int(self.xgb_n_estimators * 0.8)),
                        max_depth=max(3, self.xgb_max_depth - 1),
                        learning_rate=self.xgb_learning_rate,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def _build_regressor_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=max(100, int(self.xgb_n_estimators * 0.8)),
                        max_depth=max(3, self.xgb_max_depth - 1),
                        learning_rate=self.xgb_learning_rate,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def _select_top_features(
        self,
        dataset: pd.DataFrame,
        feature_columns: list[str],
        n_top_features: int,
    ) -> list[str]:
        x = dataset[feature_columns]
        y = dataset["direction"].astype(int)

        imputer = SimpleImputer(strategy="median")
        x_imputed = imputer.fit_transform(x)

        quick_model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        quick_model.fit(x_imputed, y)

        if shap is not None:
            try:
                explainer = shap.TreeExplainer(quick_model)
                sample_size = min(len(x_imputed), 3000)
                sample_idx = np.random.RandomState(42).choice(
                    len(x_imputed), size=sample_size, replace=False
                )
                shap_values = explainer.shap_values(x_imputed[sample_idx])
                if isinstance(shap_values, list):
                    shap_matrix = np.asarray(shap_values[-1])
                else:
                    shap_matrix = np.asarray(shap_values)
                importances = np.mean(np.abs(shap_matrix), axis=0)
            except Exception:
                importances = quick_model.feature_importances_
        else:
            importances = quick_model.feature_importances_

        ranking = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
        top_n = min(n_top_features, len(ranking))
        return ranking.head(top_n).index.tolist()

    def _walk_forward_validate(
        self,
        dataset: pd.DataFrame,
        feature_columns: list[str],
        n_folds: int,
        min_train: int,
    ) -> WalkForwardResult:
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if len(dataset) <= min_train + n_folds:
            raise ValueError("Dataset is too small for requested walk-forward settings.")

        total_rows = len(dataset)
        usable = total_rows - min_train
        fold_size = max(1, usable // n_folds)

        folds: list[WalkForwardFold] = []
        for fold_idx in range(n_folds):
            train_end = min_train + (fold_idx * fold_size)
            val_end = min(min_train + ((fold_idx + 1) * fold_size), total_rows)
            if fold_idx == n_folds - 1:
                val_end = total_rows
            if train_end >= val_end:
                continue

            train_df = dataset.iloc[:train_end]
            val_df = dataset.iloc[train_end:val_end]

            x_train = train_df[feature_columns]
            y_train = train_df["direction"].astype(int)
            x_val = val_df[feature_columns]
            y_val = val_df["direction"].astype(int)

            pipeline = self._build_direction_pipeline()
            pipeline.fit(x_train, y_train)
            preds = pipeline.predict(x_val)

            folds.append(
                WalkForwardFold(
                    fold=fold_idx + 1,
                    train_size=int(len(train_df)),
                    val_size=int(len(val_df)),
                    direction_accuracy=float(accuracy_score(y_val, preds)),
                    direction_f1=float(f1_score(y_val, preds, average="weighted")),
                )
            )

        if not folds:
            raise ValueError("No walk-forward folds could be generated.")

        acc_values = np.array([f.direction_accuracy for f in folds], dtype=float)
        f1_values = np.array([f.direction_f1 for f in folds], dtype=float)

        return WalkForwardResult(
            n_folds=int(len(folds)),
            mean_accuracy=float(acc_values.mean()),
            std_accuracy=float(acc_values.std(ddof=0)),
            mean_f1=float(f1_values.mean()),
            std_f1=float(f1_values.std(ddof=0)),
            folds=folds,
        )

    def _save_artifacts(
        self,
        *,
        artifacts_dir: Path,
        instrument: str,
        direction_pipeline: Pipeline,
        sl_pipeline: Pipeline,
        sl_encoder: LabelEncoder,
        trail_bin_pipeline: Pipeline,
        trail_bin_encoder: LabelEncoder,
        trail_tf_pipeline: Pipeline,
        trail_tf_encoder: LabelEncoder,
        phase1_pipeline: Pipeline,
        selected_features: list[str],
        direction_acc: float,
        direction_f1: float,
        phase1_mae: float,
        walk_forward: WalkForwardResult,
    ) -> None:
        joblib.dump(direction_pipeline, artifacts_dir / f"{instrument}_direction.joblib")
        joblib.dump(sl_pipeline, artifacts_dir / f"{instrument}_sl_bin.joblib")
        joblib.dump(sl_encoder, artifacts_dir / f"{instrument}_sl_bin_encoder.joblib")
        joblib.dump(trail_bin_pipeline, artifacts_dir / f"{instrument}_trail_bin.joblib")
        joblib.dump(trail_bin_encoder, artifacts_dir / f"{instrument}_trail_bin_encoder.joblib")
        joblib.dump(trail_tf_pipeline, artifacts_dir / f"{instrument}_trail_tf.joblib")
        joblib.dump(trail_tf_encoder, artifacts_dir / f"{instrument}_trail_tf_encoder.joblib")
        joblib.dump(phase1_pipeline, artifacts_dir / f"{instrument}_phase1_target.joblib")

        (artifacts_dir / f"{instrument}_selected_features.json").write_text(
            json.dumps(selected_features, indent=2),
            encoding="utf-8",
        )

        metrics = {
            "instrument": instrument,
            "direction_accuracy": direction_acc,
            "direction_f1": direction_f1,
            "phase1_target_mae": phase1_mae,
            "walk_forward": asdict(walk_forward),
        }
        (artifacts_dir / f"{instrument}_train_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        logger.info("Saved training artifacts to %s", artifacts_dir)


nifty_trainer = NiftyTrainer()
