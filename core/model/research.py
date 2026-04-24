"""Research runner for comparing multiple Phase 4 model-training candidates."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from core.model.train import NiftyTrainer

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


@dataclass
class ResearchCandidate:
    n_top_features: int
    n_folds: int
    test_fraction: float
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    description: str


@dataclass
class ResearchResult:
    candidate: ResearchCandidate
    direction_accuracy: float
    direction_f1: float
    wf_mean_accuracy: float
    wf_mean_f1: float
    artifacts_dir: str


def _default_candidates() -> list[ResearchCandidate]:
    return [
        ResearchCandidate(90, 5, 0.15, 500, 6, 0.03, "V1"),
        ResearchCandidate(80, 5, 0.15, 400, 5, 0.05, "V2"),
        ResearchCandidate(100, 5, 0.15, 600, 7, 0.02, "V3"),
        ResearchCandidate(70, 5, 0.15, 500, 5, 0.03, "V4"),
        ResearchCandidate(90, 7, 0.15, 500, 6, 0.03, "V5"),
        ResearchCandidate(90, 5, 0.15, 800, 8, 0.01, "V6"),
    ]


def run_research(
    dataset_path: str | Path,
    instrument: str,
    output_base_dir: str | Path,
    candidates: list[ResearchCandidate] | None = None,
) -> list[ResearchResult]:
    candidate_list = candidates if candidates is not None else _default_candidates()
    output_root = Path(output_base_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[ResearchResult] = []
    for candidate in candidate_list:
        trainer = NiftyTrainer(
            xgb_n_estimators=candidate.xgb_n_estimators,
            xgb_max_depth=candidate.xgb_max_depth,
            xgb_learning_rate=candidate.xgb_learning_rate,
        )
        candidate_output = output_root / candidate.description
        try:
            train_result = trainer.train(
                dataset_path=dataset_path,
                instrument=instrument,
                output_dir=candidate_output,
                n_folds=candidate.n_folds,
                n_top_features=candidate.n_top_features,
                test_fraction=candidate.test_fraction,
            )
        except Exception as exc:
            logger.warning("Candidate %s failed: %s", candidate.description, exc)
            continue

        results.append(
            ResearchResult(
                candidate=candidate,
                direction_accuracy=float(train_result.direction_metrics["accuracy"]),
                direction_f1=float(train_result.direction_metrics["f1_weighted"]),
                wf_mean_accuracy=float(train_result.walk_forward.mean_accuracy),
                wf_mean_f1=float(train_result.walk_forward.mean_f1),
                artifacts_dir=train_result.artifacts_dir,
            )
        )

    ranked = sorted(results, key=lambda item: item.direction_f1, reverse=True)
    report = [asdict(item) for item in ranked]
    (output_root / "research_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ranked
