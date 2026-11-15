import pytest

from page_replacement.policies import FIFOPolicy
from page_replacement.simulator import simulate


def test_fifo_hand_checked_fault_count_and_resident_state():
    result = simulate([1, 2, 3, 1, 4, 2], 3, FIFOPolicy())

    assert result.references == 6
    assert result.hits == 2
    assert result.faults == 4
    assert result.evictions == 1
    assert result.fault_rate == pytest.approx(4 / 6)
    assert result.events[4].evicted_page == 1
    assert result.events[4].resident_pages == (2, 3, 4)


def test_empty_and_single_reference_traces():
    empty = simulate([], 2, FIFOPolicy())
    single = simulate([7], 1, FIFOPolicy())

    assert (empty.hits, empty.faults, empty.fault_rate) == (0, 0, 0.0)
    assert (single.hits, single.faults, single.evictions) == (0, 1, 0)
    assert single.events[0].hit is False


@pytest.mark.parametrize("capacity", (0, -1, 1.5, True))
def test_frame_capacity_must_be_a_positive_integer(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        simulate([1], capacity, FIFOPolicy())


def test_trace_page_ids_must_be_integers():
    with pytest.raises(TypeError, match="integers"):
        simulate(["page"], 1, FIFOPolicy())


def test_event_recording_can_be_disabled_for_batch_runs():
    result = simulate([1, 2, 1, 3], 2, FIFOPolicy(), record_events=False)

    assert result.faults == 3
    assert result.hits == 1
    assert result.events == ()

