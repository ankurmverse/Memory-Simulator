"""Page replacement policy contracts and the FIFO baseline."""

from __future__ import annotations

from collections import deque
from typing import Protocol, Sequence


class ReplacementPolicy(Protocol):
    """Policy hooks called by the simulator as pages enter and leave memory."""

    def on_access(self, page: int) -> None: ...

    def on_load(self, page: int) -> None: ...

    def on_evict(self, page: int) -> None: ...

    def choose_victim(self, resident_pages: Sequence[int]) -> int: ...


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
        if not self._queue or self._queue[0] not in resident_pages:
            raise RuntimeError("FIFO policy has no valid resident victim")
        return self._queue[0]

