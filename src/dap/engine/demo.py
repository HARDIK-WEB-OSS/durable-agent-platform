import asyncio
import uuid
from collections import Counter
from functools import partial

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dap.db.models import DemoSideEffect, JsonDict
from dap.engine.queue import Claim
from dap.engine.runner import StepContext, StepResult, StepSpec

CHAIN_KIND = "demo.chain"


def chain_spec(
    total: int, sleep_s: float = 0.2, max_attempts: int = 5, fail_first_attempts: int = 0
) -> StepSpec:
    payload = {
        "i": 0,
        "total": total,
        "sleep_s": sleep_s,
        "fail_first_attempts": fail_first_attempts,
    }
    return StepSpec(CHAIN_KIND, payload, max_attempts)


async def write_ledger(claim: Claim, session: AsyncSession) -> JsonDict:
    row = DemoSideEffect(
        run_id=claim.run_id,
        step_seq=claim.input["i"],
        worker_id=claim.worker_id,
        attempt=claim.attempt,
    )
    session.add(row)
    await session.flush()
    return {"ledger_id": row.id}


async def chain_step(ctx: StepContext) -> StepResult:
    params = ctx.claim.input
    i, total, sleep_s = params["i"], params["total"], params["sleep_s"]
    if ctx.claim.attempt <= params["fail_first_attempts"]:
        raise RuntimeError(f"injected failure on attempt {ctx.claim.attempt}")
    await asyncio.sleep(sleep_s)
    effect = await ctx.once("ledger", partial(write_ledger, ctx.claim))
    await asyncio.sleep(sleep_s)
    output = {"i": i, "ledger_id": effect["ledger_id"]}
    if i + 1 < total:
        next_spec = StepSpec(CHAIN_KIND, {**params, "i": i + 1}, ctx.claim.max_attempts)
        return StepResult(output, next_step=next_spec)
    return StepResult(output, run_result={"steps": total})


async def ledger_counts(session: AsyncSession, run_id: uuid.UUID) -> Counter[int]:
    stmt = select(DemoSideEffect.step_seq).where(DemoSideEffect.run_id == run_id)
    return Counter((await session.scalars(stmt)).all())
