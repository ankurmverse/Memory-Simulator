"""Causal eviction-candidate features and offline next-use labels."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from page_replacement.policies import ReplacementPolicy
from page_replacement.simulator import simulate


@dataclass(frozen=True)
class CandidateFeatures:
    """Numeric candidate features computed only from pre-decision history."""

    access_count: int
    references_since_last_access: int
    last_reuse_distance: int
    mean_reuse_distance: float
    max_reuse_distance: int
    residency_age: int


@dataclass(frozen=True)
class CandidateSample:
    """One resident eviction candidate, its causal features, and offline target."""

    reference_index: int
    incoming_page: int
    candidate_page: int
    features: CandidateFeatures
    next_use_distance: int | None


class CausalFeatureHistory:
    """Track past accesses and expose features available at a decision point."""

    def __init__(self) -> None:
        self._access_count: dict[int, int] = defaultdict(int)
        self._last_access: dict[int, int] = {}
        self._last_reuse_distance: dict[int, int] = defaultdict(int)
        self._reuse_distance_sum: dict[int, int] = defaultdict(int)
        self._max_reuse_distance: dict[int, int] = defaultdict(int)

    def record_access(self, page: int, index: int) -> None:
        if page in self._last_access:
            gap = index - self._last_access[page]
            self._last_reuse_distance[page] = gap
            self._reuse_distance_sum[page] += gap
            self._max_reuse_distance[page] = max(
                self._max_reuse_distance[page], gap
            )
        self._access_count[page] += 1
        self._last_access[page] = index

    def features_for(
        self, page: int, decision_index: int, residency_start: int
    ) -> CandidateFeatures:
        if page not in self._last_access:
            raise RuntimeError(f"missing access history for resident page {page}")
        count = self._access_count[page]
        return CandidateFeatures(
            access_count=count,
            references_since_last_access=decision_index - self._last_access[page],
            last_reuse_distance=self._last_reuse_distance[page],
            mean_reuse_distance=(
                self._reuse_distance_sum[page] / (count - 1) if count > 1 else 0.0
            ),
            max_reuse_distance=self._max_reuse_distance[page],
            residency_age=decision_index - residency_start,
        )


def feature_vector(features: CandidateFeatures) -> tuple[float, ...]:
    """Return candidate features in the stable model input order."""
    return (
        float(features.access_count),
        float(features.references_since_last_access),
        float(features.last_reuse_distance),
        features.mean_reuse_distance,
        float(features.max_reuse_distance),
        float(features.residency_age),
    )


def build_candidate_samples(
    trace: Sequence[int],
    frame_capacity: int,
    policy: ReplacementPolicy,
) -> tuple[CandidateSample, ...]:
    """Create one labeled sample per candidate at every full-memory fault.

    Features use only references strictly before the fault at ``reference_index``.
    The target is the distance from that decision to the candidate's next use;
    ``None`` means the candidate does not appear again in the trace.
    """
    result = simulate(trace, frame_capacity, policy)
    positions: dict[int, list[int]] = defaultdict(list)
    for index, page in enumerate(trace):
        positions[page].append(index)

    samples: list[CandidateSample] = []
    history = CausalFeatureHistory()
    resident_since: dict[int, int] = {}

    for event in result.events:
        if not event.hit and event.evicted_page is not None:
            candidates = (set(event.resident_pages) - {event.page}) | {
                event.evicted_page
            }
            if len(candidates) != frame_capacity:
                raise RuntimeError(
                    f"expected {frame_capacity} eviction candidates at reference "
                    f"{event.index}, found {len(candidates)}"
                )
            for candidate in sorted(candidates):
                if candidate not in resident_since:
                    raise RuntimeError(
                        f"missing history for resident page {candidate} at "
                        f"reference {event.index}"
                    )
                future_position = bisect_right(positions[candidate], event.index)
                next_use_distance = (
                    positions[candidate][future_position] - event.index
                    if future_position < len(positions[candidate])
                    else None
                )
                samples.append(
                    CandidateSample(
                        reference_index=event.index,
                        incoming_page=event.page,
                        candidate_page=candidate,
                        features=history.features_for(
                            candidate, event.index, resident_since[candidate]
                        ),
                        next_use_distance=next_use_distance,
                    )
                )

        if not event.hit:
            if event.evicted_page is not None:
                del resident_since[event.evicted_page]
            resident_since[event.page] = event.index
        history.record_access(event.page, event.index)

    return tuple(samples)
