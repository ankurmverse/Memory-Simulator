"""Policy-independent trace simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from page_replacement.policies import ReplacementPolicy


@dataclass(frozen=True)
class ReferenceEvent:
    """Outcome and post-reference memory state for one page request."""

    index: int
    page: int
    hit: bool
    evicted_page: int | None
    resident_pages: tuple[int, ...]


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate counters and optional per-reference events."""

    references: int
    hits: int
    faults: int
    evictions: int
    events: tuple[ReferenceEvent, ...]

    @property
    def fault_rate(self) -> float:
        return self.faults / self.references if self.references else 0.0


def simulate(
    trace: Sequence[int],
    frame_capacity: int,
    policy: ReplacementPolicy,
    record_events: bool = True,
) -> SimulationResult:
    """Run a page-reference trace using the supplied replacement policy."""
    if (
        not isinstance(frame_capacity, int)
        or isinstance(frame_capacity, bool)
        or frame_capacity <= 0
    ):
        raise ValueError("frame capacity must be a positive integer")
    if any(not isinstance(page, int) or isinstance(page, bool) for page in trace):
        raise TypeError("trace page IDs must be integers (booleans are not page IDs)")

    residents: list[int] = []
    resident_set: set[int] = set()
    events: list[ReferenceEvent] = []
    hits = 0
    faults = 0
    evictions = 0

    for index, page in enumerate(trace):
        policy.on_access(page)
        evicted_page: int | None = None
        hit = page in resident_set
        if hit:
            hits += 1
        else:
            faults += 1
            if len(residents) == frame_capacity:
                evicted_page = policy.choose_victim(tuple(residents))
                if evicted_page not in resident_set:
                    raise RuntimeError(
                        f"policy selected non-resident page {evicted_page} at reference {index}"
                    )
                resident_set.remove(evicted_page)
                residents.remove(evicted_page)
                policy.on_evict(evicted_page)
                evictions += 1
            residents.append(page)
            resident_set.add(page)
            policy.on_load(page)

        if record_events:
            events.append(
                ReferenceEvent(index, page, hit, evicted_page, tuple(residents))
            )

    return SimulationResult(
        references=len(trace),
        hits=hits,
        faults=faults,
        evictions=evictions,
        events=tuple(events),
    )

