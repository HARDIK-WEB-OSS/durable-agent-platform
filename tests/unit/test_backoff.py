import random

import pytest

from dap.backoff import backoff_seconds
from dap.engine.runner import StepResult, StepSpec


@pytest.mark.parametrize("attempt", range(1, 12))
def test_backoff_stays_within_exponential_ceiling(attempt: int) -> None:
    rng = random.Random(attempt)
    ceiling = min(30.0, 0.5 * 2 ** (attempt - 1))
    delays = [backoff_seconds(attempt, 0.5, 30.0, rng) for _ in range(200)]
    assert all(0.0 <= d <= ceiling for d in delays)


def test_backoff_is_jittered_not_constant() -> None:
    rng = random.Random(0)
    delays = {round(backoff_seconds(4, 0.5, 30.0, rng), 6) for _ in range(50)}
    assert len(delays) > 40


def test_backoff_is_capped() -> None:
    rng = random.Random(0)
    assert max(backoff_seconds(50, 0.5, 30.0, rng) for _ in range(500)) <= 30.0


def test_step_result_rejects_next_step_and_final_result_together() -> None:
    with pytest.raises(ValueError, match="either"):
        StepResult({}, next_step=StepSpec("x"), run_result={"done": True})
