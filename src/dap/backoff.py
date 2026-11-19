import random


def backoff_seconds(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    return rng.uniform(0.0, min(cap, base * 2 ** (attempt - 1)))
