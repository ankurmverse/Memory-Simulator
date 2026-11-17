"""Page replacement policy contracts and the FIFO baseline."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class PolicyComplexity:
    """Asymptotic policy costs for a trace request and resident capacity."""

    hit: str
    fault_without_eviction: str
    fault_with_eviction: str
    extra_space: str
    data_structure: str
    tradeoff: str


class ReplacementPolicy(Protocol):
    """Policy hooks called by the simulator as pages enter and leave memory."""

    def on_access(self, page: int) -> None: ...

    def on_load(self, page: int) -> None: ...

    def on_evict(self, page: int) -> None: ...

    def choose_victim(self, resident_pages: Sequence[int]) -> int: ...

    @property
    def complexity(self) -> PolicyComplexity: ...


class FIFOPolicy:
    """Evict the resident page that entered memory earliest."""

    def __init__(self) -> None:
        self._queue: deque[int] = deque()

    def on_access(self, page: int) -> None:
        del page

    def on_load(self, page: int) -> None:
        self._queue.append(page)

    def on_evict(self, page: int) -> None:
        if not self._queue or self._queue[0] != page:
            raise RuntimeError("FIFO policy state is inconsistent with resident pages")
        self._queue.popleft()

    def choose_victim(self, resident_pages: Sequence[int]) -> int:
        del resident_pages
        if not self._queue:
            raise RuntimeError("FIFO policy has no valid resident victim")
        return self._queue[0]

    @property
    def complexity(self) -> PolicyComplexity:
        return PolicyComplexity(
            hit="O(1)",
            fault_without_eviction="O(1)",
            fault_with_eviction="O(1)",
            extra_space="O(f)",
            data_structure="Deque",
            tradeoff="Fast queue operations; no recency adaptation.",
        )


class LRUPolicy:
    """Evict the resident page that has not been used for the longest time."""

    def __init__(self) -> None:
        self._recency: OrderedDict[int, None] = OrderedDict()

    def on_access(self, page: int) -> None:
        if page in self._recency:
            self._recency.move_to_end(page)

    def on_load(self, page: int) -> None:
        if page in self._recency:
            raise RuntimeError(f"LRU page {page} is already resident")
        self._recency[page] = None

    def on_evict(self, page: int) -> None:
        if page not in self._recency:
            raise RuntimeError(f"LRU page {page} is not resident")
        del self._recency[page]

    def choose_victim(self, resident_pages: Sequence[int]) -> int:
        del resident_pages
        if not self._recency:
            raise RuntimeError("LRU policy has no valid resident victim")
        return next(iter(self._recency))

    @property
    def complexity(self) -> PolicyComplexity:
        return PolicyComplexity(
            hit="O(1) average",
            fault_without_eviction="O(1) average",
            fault_with_eviction="O(1) average",
            extra_space="O(f)",
            data_structure="OrderedDict (hash table + linked order)",
            tradeoff="Constant-time recency updates, with higher bookkeeping than FIFO.",
        )


class CLOCKPolicy:
    """Approximate LRU using reference bits and a circular second-chance hand."""

    def __init__(self) -> None:
        self._pages: list[int] = []
        self._referenced: list[bool] = []
        self._page_indices: dict[int, int] = {}
        self._hand = 0

    def on_access(self, page: int) -> None:
        index = self._page_indices.get(page)
        if index is not None:
            self._referenced[index] = True

    def on_load(self, page: int) -> None:
        if page in self._page_indices:
            raise RuntimeError(f"CLOCK page {page} is already resident")
        self._page_indices[page] = len(self._pages)
        self._pages.append(page)
        self._referenced.append(True)
        if len(self._pages) == 1:
            self._hand = 0

    def on_evict(self, page: int) -> None:
        try:
            index = self._page_indices[page]
        except KeyError as error:
            raise RuntimeError(f"CLOCK page {page} is not resident") from error
        if index != self._hand:
            raise RuntimeError("CLOCK eviction does not match the hand position")
        self._pages.pop(index)
        self._referenced.pop(index)
        del self._page_indices[page]
        for moved_index in range(index, len(self._pages)):
            self._page_indices[self._pages[moved_index]] = moved_index
        if self._pages:
            self._hand %= len(self._pages)
        else:
            self._hand = 0

    def choose_victim(self, resident_pages: Sequence[int]) -> int:
        if not self._pages:
            raise RuntimeError("CLOCK policy has no resident victim")
        if set(self._pages) != set(resident_pages):
            raise RuntimeError("CLOCK policy state is inconsistent with resident pages")

        while True:
            if self._referenced[self._hand]:
                self._referenced[self._hand] = False
                self._hand = (self._hand + 1) % len(self._pages)
            else:
                return self._pages[self._hand]

    @property
    def complexity(self) -> PolicyComplexity:
        return PolicyComplexity(
            hit="O(1) average",
            fault_without_eviction="O(1) average",
            fault_with_eviction="O(f) worst case",
            extra_space="O(f)",
            data_structure="Circular arrays + page-to-slot hash map",
            tradeoff=(
                "Fast page lookup; a full clock scan and array compaction can be "
                "linear."
            ),
        )
