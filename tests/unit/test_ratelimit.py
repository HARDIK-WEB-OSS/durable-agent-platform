import asyncio

import pytest

from dap.llm.ratelimit import RateLimiter


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def limiter(rpm: float, t: FakeTime) -> RateLimiter:
    return RateLimiter(rpm, clock=t.clock, sleep=t.sleep)


async def test_first_request_does_not_wait() -> None:
    t = FakeTime()
    assert await limiter(30, t).acquire() == 0.0
    assert t.sleeps == []


async def test_back_to_back_requests_are_spaced_by_interval() -> None:
    t = FakeTime()
    rl = limiter(30, t)  # one request every 2 s
    waits = [await rl.acquire() for _ in range(3)]
    assert waits == pytest.approx([0.0, 2.0, 4.0])


async def test_idle_time_is_not_banked_into_a_burst() -> None:
    t = FakeTime()
    rl = limiter(60, t)
    await rl.acquire()
    t.now += 10.0
    assert await rl.acquire() == 0.0
    assert await rl.acquire() == pytest.approx(1.0)


async def test_request_after_interval_has_passed_does_not_wait() -> None:
    t = FakeTime()
    rl = limiter(60, t)
    await rl.acquire()
    t.now += 1.5
    assert await rl.acquire() == 0.0


async def test_concurrent_callers_get_distinct_slots() -> None:
    t = FakeTime()
    rl = limiter(120, t)  # 0.5 s apart
    waits = await asyncio.gather(*(rl.acquire() for _ in range(4)))
    assert sorted(waits) == pytest.approx([0.0, 0.5, 1.0, 1.5])


async def test_real_clock_actually_waits() -> None:
    rl = RateLimiter(600)  # 0.1 s apart
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(3):
        await rl.acquire()
    assert loop.time() - start >= 0.15  # 2 waits of 0.1 s, minus Windows timer slack


@pytest.mark.parametrize("rpm", [0, -5])
def test_non_positive_rate_is_rejected(rpm: float) -> None:
    with pytest.raises(ValueError, match="requests_per_minute"):
        RateLimiter(rpm)
