import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

SECONDS_PER_MINUTE = 60.0


@dataclass
class RateLimiter:
    """Spaces requests evenly: at most `requests_per_minute`, no bursts.

    Each caller reserves the next free slot under a lock, then sleeps outside it, so
    concurrent callers queue up one interval apart instead of all waking at once.
    Scope is one process: N worker processes each get their own limiter.
    """

    requests_per_minute: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    next_slot: float = field(default=float("-inf"), init=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.requests_per_minute > 0:
            raise ValueError(f"requests_per_minute must be > 0, got {self.requests_per_minute}")

    @property
    def interval_s(self) -> float:
        return SECONDS_PER_MINUTE / self.requests_per_minute

    async def acquire(self) -> float:
        """Wait for a slot; return how long this caller waited."""
        async with self.lock:
            now = self.clock()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval_s
        wait = slot - now
        if wait > 0:
            await self.sleep(wait)
        return wait
