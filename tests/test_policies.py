import pytest

from page_replacement.policies import FIFOPolicy


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

