import pytest

from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy


def test_fifo_evicts_oldest_loaded_page_and_ignores_hits():
    policy = FIFOPolicy()
    policy.on_load(1)
    policy.on_load(2)
    policy.on_access(1)

    assert policy.choose_victim((1, 2)) == 1


def test_fifo_state_advances_after_eviction():
    policy = FIFOPolicy()
    policy.on_load(1)
    policy.on_load(2)
    policy.on_evict(1)

    assert policy.choose_victim((2,)) == 2


def test_fifo_rejects_inconsistent_eviction():
    policy = FIFOPolicy()
    policy.on_load(1)
    policy.on_load(2)

    with pytest.raises(RuntimeError, match="inconsistent"):
        policy.on_evict(2)


def test_lru_hit_updates_recency_before_victim_selection():
    policy = LRUPolicy()
    for page in (1, 2, 3):
        policy.on_load(page)
    policy.on_access(1)

    assert policy.choose_victim((1, 2, 3)) == 2


def test_lru_state_advances_after_eviction():
    policy = LRUPolicy()
    for page in (1, 2, 3):
        policy.on_load(page)
    policy.on_access(1)
    policy.on_evict(2)
    policy.on_load(4)

    assert policy.choose_victim((1, 3, 4)) == 3


def test_lru_rejects_inconsistent_load_or_eviction():
    policy = LRUPolicy()
    policy.on_load(1)

    with pytest.raises(RuntimeError, match="already resident"):
        policy.on_load(1)
    with pytest.raises(RuntimeError, match="not resident"):
        policy.on_evict(2)


def test_clock_gives_referenced_pages_a_second_chance():
    policy = CLOCKPolicy()
    for page in (1, 2, 3):
        policy.on_load(page)

    assert policy.choose_victim((1, 2, 3)) == 1
    policy.on_evict(1)
    policy.on_load(4)

    assert policy.choose_victim((2, 3, 4)) == 2


def test_clock_hit_sets_reference_bit_and_hand_advances_after_eviction():
    policy = CLOCKPolicy()
    for page in (1, 2, 3):
        policy.on_load(page)
    assert policy.choose_victim((1, 2, 3)) == 1
    policy.on_evict(1)
    policy.on_load(4)
    policy.on_access(2)

    assert policy.choose_victim((2, 3, 4)) == 3


def test_policies_expose_complexity_and_storage_tradeoffs():
    fifo = FIFOPolicy().complexity
    lru = LRUPolicy().complexity
    clock = CLOCKPolicy().complexity

    assert (fifo.hit, fifo.fault_with_eviction, fifo.extra_space) == (
        "O(1)",
        "O(1)",
        "O(f)",
    )
    assert lru.hit == "O(1) average"
    assert lru.data_structure.startswith("OrderedDict")
    assert clock.hit == "O(1) average"
    assert clock.fault_with_eviction == "O(f) worst case"
    assert "hash map" in clock.data_structure


def test_clock_rejects_eviction_that_does_not_match_hand():
    policy = CLOCKPolicy()
    policy.on_load(1)
    policy.on_load(2)

    with pytest.raises(RuntimeError, match="hand position"):
        policy.on_evict(2)
