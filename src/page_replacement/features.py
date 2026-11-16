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
    access_count: dict[int, int] = defaultdict(int)
    last_access: dict[int, int] = {}
    last_reuse_distance: dict[int, int] = defaultdict(int)
    reuse_distance_sum: dict[int, int] = defaultdict(int)
    max_reuse_distance: dict[int, int] = defaultdict(int)
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
                if candidate not in resident_since or candidate not in last_access:
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
                        features=CandidateFeatures(
                            access_count=access_count[candidate],
                            references_since_last_access=event.index
                            - last_access[candidate],
                            last_reuse_distance=last_reuse_distance[candidate],
                            mean_reuse_distance=(
                                reuse_distance_sum[candidate]
                                / (access_count[candidate] - 1)
                                if access_count[candidate] > 1
                                else 0.0
                            ),
                            max_reuse_distance=max_reuse_distance[candidate],
                            residency_age=event.index - resident_since[candidate],
                        ),
                        next_use_distance=next_use_distance,
                    )
                )

        if not event.hit:
            if event.evicted_page is not None:
                del resident_since[event.evicted_page]
            resident_since[event.page] = event.index
        if event.page in last_access:
            gap = event.index - last_access[event.page]
            last_reuse_distance[event.page] = gap
            reuse_distance_sum[event.page] += gap
            max_reuse_distance[event.page] = max(max_reuse_distance[event.page], gap)
        access_count[event.page] += 1
        last_access[event.page] = event.index

    return tuple(samples)
