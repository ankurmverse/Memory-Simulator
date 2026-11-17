"""Reusable workload analysis helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from page_replacement.policies import ReplacementPolicy
from page_replacement.simulator import simulate


@dataclass(frozen=True)
class CapacitySweepPoint:
    """Fault metrics for one policy at one memory capacity."""

    policy: str
    frame_capacity: int
    faults: int
    fault_rate: float


PolicyFactory = Callable[[], ReplacementPolicy]


def sweep_frame_capacities(
    trace: Sequence[int],
    max_frame_capacity: int,
    policies: Mapping[str, PolicyFactory],
) -> tuple[CapacitySweepPoint, ...]:
    """Compare fresh policy instances at capacities up to the configured maximum."""
    if (
        not isinstance(max_frame_capacity, int)
        or isinstance(max_frame_capacity, bool)
        or max_frame_capacity <= 0
    ):
        raise ValueError("maximum frame capacity must be a positive integer")
    if not policies:
        raise ValueError("at least one policy is required")

    points = []
    for capacity in range(1, max_frame_capacity + 1):
        for name, policy_factory in policies.items():
            result = simulate(
                trace,
                capacity,
                policy_factory(),
                record_events=False,
            )
            points.append(
                CapacitySweepPoint(
                    policy=name,
                    frame_capacity=capacity,
                    faults=result.faults,
                    fault_rate=result.fault_rate,
                )
            )
    return tuple(points)
