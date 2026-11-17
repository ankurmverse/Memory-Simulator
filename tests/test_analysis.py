import pytest

from page_replacement.analysis import sweep_frame_capacities
from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy


def test_capacity_sweep_reports_fifo_beladys_anomaly():
    trace = (1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5)
    points = sweep_frame_capacities(trace, 4, {"fifo": FIFOPolicy})
    faults_by_capacity = {point.frame_capacity: point.faults for point in points}

    assert faults_by_capacity == {1: 12, 2: 12, 3: 9, 4: 10}


def test_lru_faults_do_not_increase_with_capacity():
    trace = (1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5)
    points = sweep_frame_capacities(trace, 6, {"lru": LRUPolicy})
    faults = [point.faults for point in points]

    assert faults == sorted(faults, reverse=True)


def test_capacity_sweep_includes_each_configured_policy_and_frame_count():
    trace = (1, 2, 3, 4, 1, 2, 5)
    points = sweep_frame_capacities(
        trace, 3, {"lru": LRUPolicy, "clock": CLOCKPolicy}
    )

    assert len(points) == 6
    assert {(point.policy, point.frame_capacity) for point in points} == {
        (policy, frames)
        for policy in ("lru", "clock")
        for frames in (1, 2, 3)
    }


@pytest.mark.parametrize("max_frames", (0, -1, True, 1.5))
def test_capacity_sweep_requires_positive_integer(max_frames):
    with pytest.raises(ValueError, match="positive integer"):
        sweep_frame_capacities((1, 2), max_frames, {"fifo": FIFOPolicy})


def test_capacity_sweep_requires_at_least_one_policy():
    with pytest.raises(ValueError, match="at least one policy"):
        sweep_frame_capacities((1,), 2, {})
