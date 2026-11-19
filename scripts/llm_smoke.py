"""Phase 3 acceptance check: every arm must return schema-valid JSON twice (cold, warm).

Local arms always run. Hosted (paid) arms run only with --allow-paid; without it the
script prints each hosted arm's price and worst-case cost and counts the arm as not passed.

    uv run python scripts/llm_smoke.py               # local arms only, no spend
    uv run python scripts/llm_smoke.py --allow-paid  # all arms, spends the printed amount
"""

import argparse
import asyncio
import sys

from pydantic import BaseModel

from dap.config import get_settings
from dap.db import repo
from dap.db.session import make_engine, make_sessionmaker
from dap.llm.client import (
    ARM_NAMES,
    LLMCallError,
    LLMClient,
    LLMResult,
    ModelConfig,
    configured_arms,
    estimate_prompt_tokens,
    record_llm_call,
    record_llm_failure,
)
from dap.llm.pricing import call_cost, price_basis, price_per_million
from dap.states import RunStatus

EXPECTED = 8
MAX_TOKENS = 512  # room for a reasoning model's hidden thinking tokens plus the JSON answer
LABELS = ("cold", "warm")
REPAIR_ATTEMPTS = 1
REPAIR_PROMPT_TOKENS = 200  # generous bound on the error text a repair turn adds
MESSAGES = [
    {"role": "system", "content": "Solve the problem. Reply only with JSON."},
    {"role": "user", "content": "Pens cost $2 for 3 pens. How many dollars do 12 pens cost?"},
]

HEADER = (
    f"{'arm':<7}{'call':<6}{'model':<30}{'ans':>4} chk {'in':>6}{'out':>6}"
    f"{'paid $':>10}{'shadow $':>12} {'basis':<9}{'latency':>8}{'wall':>9}{'n':>4} trunc"
)


class Answer(BaseModel):
    answer: int


def short_name(model: str) -> str:
    name = model.removeprefix("ollama_chat/")
    return name if len(name) <= 29 else name[:28] + "~"


def row(label: str, parsed: Answer, r: LLMResult) -> str:
    mark = "ok " if parsed.answer == EXPECTED else "BAD"
    return (
        f"{r.arm:<7}{label:<6}{short_name(r.model):<30}{parsed.answer:>4} {mark}"
        f"{r.tokens_in:>7}{r.tokens_out:>6}{r.cost_usd:>10.6f}{r.shadow_cost_usd:>12.8f}"
        f" {r.price_basis:<9}{r.latency_s:>7.2f}s{r.wall_s:>8.2f}s{r.calls:>4}"
        f" {'YES' if r.truncated else 'no'}"
    )


def worst_case_cost(
    model: ModelConfig, overrides: dict[str, tuple[float, float]], max_retries: int
) -> float:
    """Upper bound for one arm: every call needs a repair turn and every attempt is billed."""
    calls = len(LABELS) * (REPAIR_ATTEMPTS + 1) * (max_retries + 1)
    prompt = estimate_prompt_tokens(MESSAGES) + MAX_TOKENS + REPAIR_PROMPT_TOKENS
    return calls * call_cost(model.model, prompt, MAX_TOKENS, overrides)


def describe_hosted(
    model: ModelConfig, overrides: dict[str, tuple[float, float]], max_retries: int
) -> str:
    price_in, price_out = price_per_million(model.model, overrides)
    basis = price_basis(model.model, overrides)
    rpm = model.requests_per_minute
    return (
        f"{model.arm:<7}hosted {model.model}: ${price_in:.4f} in / ${price_out:.4f} out "
        f"per 1M tokens ({basis}), limit {rpm:g} req/min, "
        f"worst case for this run ${worst_case_cost(model, overrides, max_retries):.6f}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--allow-paid", action="store_true", help="also call hosted arms (costs money)"
    )
    args = parser.parse_args()

    settings = get_settings()
    overrides = dict(settings.llm_price_overrides)
    arms = configured_arms(settings)
    client = LLMClient.from_settings(settings)

    for arm in arms.values():
        if not arm.local:
            print(describe_hosted(arm, overrides, settings.llm_max_retries))
    print()

    engine = make_engine(settings)
    sm = make_sessionmaker(engine)
    async with sm.begin() as s:
        run = await repo.create_run(s, "llm smoke test")
        await repo.transition_run(s, run.id, RunStatus.RUNNING)
    print(HEADER)
    failed: list[str] = []
    for name in ARM_NAMES:
        if name not in arms:
            print(f"{name:<7}NOT CONFIGURED: set {name.upper()}_MODEL in .env")
            failed.append(name)
            continue
        if not arms[name].local and not args.allow_paid:
            print(f"{name:<7}NOT RUN: hosted arm, re-run with --allow-paid to call it")
            failed.append(name)
            continue
        for label in LABELS:
            try:
                parsed, result = await client.complete_structured(
                    arms[name],
                    MESSAGES,
                    Answer,
                    max_tokens=MAX_TOKENS,
                    repair_attempts=REPAIR_ATTEMPTS,
                )
            except Exception as exc:
                print(f"{name:<7}{label:<6}FAILED {type(exc).__name__}: {exc}")
                if isinstance(exc, LLMCallError):
                    async with sm.begin() as s:
                        await record_llm_failure(s, run.id, exc)
                failed.append(name)
                break
            async with sm.begin() as s:
                await record_llm_call(s, run.id, result)
            print(row(label, parsed, result))
    async with sm.begin() as s:
        outcome = RunStatus.FAILED if failed else RunStatus.SUCCEEDED
        await repo.finish_run(s, run.id, outcome, result={"failed_arms": failed})
    await engine.dispose()
    print(f"\nevents stored under run {run.id}")
    if failed:
        print(f"NOT PASSED: {', '.join(failed)}")
        return 1
    print(f"PASSED: all {len(ARM_NAMES)} arms")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
