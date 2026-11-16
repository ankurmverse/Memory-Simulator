import pytest

from page_replacement.features import build_candidate_samples
from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy
from page_replacement.simulator import simulate


def test_candidate_features_and_next_use_labels_are_hand_checkable():
    samples = build_candidate_samples(
        [1, 2, 1, 3, 2, 4], frame_capacity=2, policy=FIFOPolicy()
    )
    at_first_eviction = [sample for sample in samples if sample.reference_index == 3]
    by_page = {sample.candidate_page: sample for sample in at_first_eviction}

    assert set(by_page) == {1, 2}
    assert by_page[1].incoming_page == 3
    assert by_page[1].features.access_count == 2
    assert by_page[1].features.references_since_last_access == 1
    assert by_page[1].features.last_reuse_distance == 2
    assert by_page[1].features.mean_reuse_distance == 2.0
    assert by_page[1].features.max_reuse_distance == 2
    assert by_page[1].features.residency_age == 3
    assert by_page[1].next_use_distance is None

    assert by_page[2].features.access_count == 1
    assert by_page[2].features.references_since_last_access == 2
    assert by_page[2].features.residency_age == 2
    assert by_page[2].next_use_distance == 1


def test_samples_include_each_candidate_at_each_eviction_decision():
    trace = [1, 2, 3, 4, 1, 2, 5]
    frame_capacity = 3
    samples = build_candidate_samples(trace, frame_capacity, LRUPolicy())
    decisions = {sample.reference_index for sample in samples}

    assert decisions == {3, 4, 5, 6}
    assert all(
        sum(sample.reference_index == index for sample in samples) == frame_capacity
        for index in decisions
    )


@pytest.mark.parametrize("policy_factory", (FIFOPolicy, LRUPolicy, CLOCKPolicy))
def test_candidate_sample_count_matches_evictions_for_each_policy(policy_factory):
    trace = [1, 2, 3, 1, 4, 2, 5, 1, 2]
    capacity = 3
    expected = simulate(trace, capacity, policy_factory())
    samples = build_candidate_samples(trace, capacity, policy_factory())

    assert len(samples) == expected.evictions * capacity


def test_features_at_a_decision_do_not_depend_on_future_references():
    prefix = [1, 2, 1, 3]
    first = build_candidate_samples(
        prefix + [2, 2], frame_capacity=2, policy=FIFOPolicy()
    )
    second = build_candidate_samples(
        prefix + [1, 1], frame_capacity=2, policy=FIFOPolicy()
    )
    first_page_one = next(
        sample for sample in first
        if sample.reference_index == 3 and sample.candidate_page == 1
    )
    second_page_one = next(
        sample for sample in second
        if sample.reference_index == 3 and sample.candidate_page == 1
    )

    assert first_page_one.features == second_page_one.features
    assert first_page_one.next_use_distance is None
    assert second_page_one.next_use_distance == 1


def test_trace_without_evictions_has_no_candidate_samples():
    assert build_candidate_samples(
        [1, 2, 1], frame_capacity=2, policy=FIFOPolicy()
    ) == ()


@pytest.mark.parametrize("capacity", (0, -1, 1.5, True))
def test_candidate_builder_validates_frame_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        build_candidate_samples([1, 2], capacity, FIFOPolicy())
