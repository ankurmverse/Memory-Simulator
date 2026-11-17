"""Random-forest reuse-distance training and learned replacement policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from page_replacement.features import (
    CausalFeatureHistory,
    CandidateSample,
    feature_vector,
)
from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy
from page_replacement.traces import SyntheticTrace

MODEL_FORMAT_VERSION = 1
FEATURE_NAMES = (
    "access_count",
    "references_since_last_access",
    "last_reuse_distance",
    "mean_reuse_distance",
    "max_reuse_distance",
    "residency_age",
)


class Regressor(Protocol):
    feature_importances_: Sequence[float]

    def predict(self, values: Sequence[Sequence[float]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class TrainingSummary:
    sample_count: int
    eviction_decisions: int
    never_reused_fraction: float
    feature_importances: tuple[tuple[str, float], ...]


def _require_ml_dependencies():
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as error:
        raise ImportError(
            "ML training and evaluation require the optional dependencies; "
            "install them with `python -m pip install -e '.[ml]'`"
        ) from error
    return joblib, RandomForestRegressor


def _training_rows(
    traces: Sequence[SyntheticTrace], frame_capacity: int
) -> tuple[list[CandidateSample], list[float]]:
    from page_replacement.features import build_candidate_samples

    samples: list[CandidateSample] = []
    targets: list[float] = []
    for trace in traces:
        trace_samples = build_candidate_samples(
            trace.pages, frame_capacity, FIFOPolicy()
        )
        samples.extend(trace_samples)
        never_use_target = float(len(trace.pages) + 1)
        targets.extend(
            float(sample.next_use_distance)
            if sample.next_use_distance is not None
            else never_use_target
            for sample in trace_samples
        )
    return samples, targets


def train_model(
    traces: Sequence[SyntheticTrace],
    frame_capacity: int,
    model_path: str | Path,
    *,
    random_state: int = 0,
    n_estimators: int = 100,
) -> TrainingSummary:
    """Train and persist a next-use-distance regressor from training traces."""
    if not traces:
        raise ValueError("at least one training trace is required")
    if (
        not isinstance(frame_capacity, int)
        or isinstance(frame_capacity, bool)
        or frame_capacity <= 0
    ):
        raise ValueError("frame capacity must be a positive integer")
    if (
        not isinstance(random_state, int)
        or isinstance(random_state, bool)
    ):
        raise ValueError("random_state must be an integer")
    if (
        not isinstance(n_estimators, int)
        or isinstance(n_estimators, bool)
        or n_estimators <= 0
    ):
        raise ValueError("n_estimators must be a positive integer")

    joblib, random_forest_regressor = _require_ml_dependencies()
    samples, targets = _training_rows(traces, frame_capacity)
    if not samples:
        raise ValueError(
            "training traces produced no eviction candidates; use longer traces "
            "or a smaller frame capacity"
        )

    estimator = random_forest_regressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    estimator.fit(
        [feature_vector(sample.features) for sample in samples],
        targets,
    )

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": MODEL_FORMAT_VERSION,
        "feature_names": FEATURE_NAMES,
        "frame_capacity": frame_capacity,
        "training_workloads": sorted({trace.workload for trace in traces}),
        "training_seeds": [trace.seed for trace in traces],
        "training_references_per_trace": sorted({len(trace.pages) for trace in traces}),
        "training_sample_count": len(samples),
        "never_use_target": max(len(trace.pages) + 1 for trace in traces),
        "estimator": estimator,
    }
    joblib.dump(artifact, path)

    importances = tuple(
        sorted(
            zip(FEATURE_NAMES, (float(value) for value in estimator.feature_importances_)),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    return TrainingSummary(
        sample_count=len(samples),
        eviction_decisions=len(samples) // frame_capacity,
        never_reused_fraction=sum(
            sample.next_use_distance is None for sample in samples
        )
        / len(samples),
        feature_importances=importances,
    )


class LearnedReplacementPolicy:
    """Choose the resident page with the greatest predicted next-use distance."""

    def __init__(self, estimator: Regressor) -> None:
        self._estimator = estimator
        self._history = CausalFeatureHistory()
        self._resident_since: dict[int, int] = {}
        self._reference_index = 0

    def on_access(self, page: int) -> None:
        self._history.record_access(page, self._reference_index)
        self._reference_index += 1

    def on_load(self, page: int) -> None:
        if page in self._resident_since:
            raise RuntimeError(f"learned policy page {page} is already resident")
        self._resident_since[page] = self._reference_index - 1

    def on_evict(self, page: int) -> None:
        if page not in self._resident_since:
            raise RuntimeError(f"learned policy page {page} is not resident")
        del self._resident_since[page]

    def choose_victim(self, resident_pages: Sequence[int]) -> int:
        if set(self._resident_since) != set(resident_pages):
            raise RuntimeError(
                "learned policy state is inconsistent with resident pages"
            )
        decision_index = self._reference_index - 1
        features = [
            self._history.features_for(
                page, decision_index, self._resident_since[page]
            )
            for page in resident_pages
        ]
        predictions = self._estimator.predict(
            [feature_vector(candidate_features) for candidate_features in features]
        )
        if len(predictions) != len(resident_pages):
            raise RuntimeError("model returned an unexpected number of predictions")
        scores = [float(prediction) for prediction in predictions]
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("model returned a non-finite next-use prediction")
        return resident_pages[max(range(len(scores)), key=scores.__getitem__)]


def load_model(model_path: str | Path) -> LearnedReplacementPolicy:
    """Load a persisted model artifact and validate its schema."""
    joblib, _ = _require_ml_dependencies()
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("model artifact must contain a mapping")
    version = artifact.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != MODEL_FORMAT_VERSION
    ):
        raise ValueError(f"unsupported model format version: {version!r}")
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("model artifact has an incompatible feature schema")
    estimator = artifact.get("estimator")
    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("model artifact does not contain a prediction model")
    return LearnedReplacementPolicy(estimator)


def evaluate_learned_policy(
    trace: Sequence[int],
    frame_capacity: int,
    model_path: str | Path,
) -> dict[str, object]:
    """Evaluate the learned model against FIFO, LRU, and CLOCK on one trace."""
    from page_replacement.simulator import simulate

    policies = {
        "fifo": FIFOPolicy,
        "lru": LRUPolicy,
        "clock": CLOCKPolicy,
        "ml": lambda: load_model(model_path),
    }
    return {
        name: simulate(trace, frame_capacity, factory(), record_events=False)
        for name, factory in policies.items()
    }
