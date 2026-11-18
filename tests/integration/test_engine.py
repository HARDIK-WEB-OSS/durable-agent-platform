import asyncio
import random
import uuid
from functools import partial

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.db import repo
from dap.db.models import DemoSideEffect, JsonDict, Run, Step
from dap.engine import demo
from dap.engine.idempotency import run_once
from dap.engine.leases import heartbeat, reap_expired_leases, renew_lease
from dap.engine.queue import Claim, claim_next_step
from dap.engine.runner import (
    EngineConfig,
    StepContext,
    StepResult,
    StepSpec,
    record_success,
    submit_run,
)
from dap.engine.worker import Worker, default_handlers
from dap.states import AttemptStatus, RunStatus, StepStatus

pytestmark = pytest.mark.integration

SM = async_sessionmaker[AsyncSession]
CONFIG = EngineConfig(
    lease_seconds=2.0, poll_seconds=0.05, retry_base_seconds=0.01, retry_cap_seconds=0.05
)
TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


async def submit(sm: SM, spec: StepSpec) -> uuid.UUID:
    async with sm.begin() as s:
        run = await submit_run(s, "demo", spec)
    return run.id


async def claim(sm: SM, worker_id: str, lease: float = 5.0) -> Claim:
    async with sm.begin() as s:
        c = await claim_next_step(s, worker_id, lease)
    assert c is not None
    return c


async def expire_leases(sm: SM) -> None:
    async with sm.begin() as s:
        await s.execute(
            update(Step)
            .where(Step.status == StepStatus.RUNNING)
            .values(lease_expires_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 0, 1))
        )


async def reap(sm: SM) -> int:
    async with sm.begin() as s:
        return await reap_expired_leases(s)


async def load_run(sm: SM, run_id: uuid.UUID) -> Run:
    async with sm() as s:
        run = await repo.get_run(s, run_id)
    assert run is not None
    return run


async def load_steps(sm: SM, run_id: uuid.UUID) -> list[Step]:
    async with sm() as s:
        return list(await repo.list_steps(s, run_id))


async def all_terminal(sm: SM, run_ids: list[uuid.UUID]) -> bool:
    async with sm() as s:
        statuses = (await s.scalars(select(Run.status).where(Run.id.in_(run_ids)))).all()
    return all(status in TERMINAL for status in statuses)


async def drive(
    sm: SM, run_ids: list[uuid.UUID], workers: int = 1, config: EngineConfig = CONFIG
) -> None:
    pool = [
        Worker(f"w{i}", sm, default_handlers(), config, rng=random.Random(i))
        for i in range(workers)
    ]
    tasks = [asyncio.create_task(w.run()) for w in pool]
    try:
        async with asyncio.timeout(30):
            while not await all_terminal(sm, run_ids):
                await asyncio.sleep(0.05)
    finally:
        for w in pool:
            w.stop.set()
        await asyncio.gather(*tasks)


async def test_skip_locked_gives_each_claimer_a_different_step(sessionmaker: SM) -> None:
    for _ in range(2):
        await submit(sessionmaker, demo.chain_spec(1))
    async with sessionmaker.begin() as s1, sessionmaker.begin() as s2, sessionmaker.begin() as s3:
        c1 = await claim_next_step(s1, "a", 5)
        c2 = await claim_next_step(s2, "b", 5)
        c3 = await claim_next_step(s3, "c", 5)
    assert c1 is not None and c2 is not None
    assert c1.step_id != c2.step_id
    assert c3 is None


async def test_claim_waits_for_available_at(sessionmaker: SM) -> None:
    await submit(sessionmaker, demo.chain_spec(1))
    async with sessionmaker.begin() as s:
        await s.execute(update(Step).values(available_at=repo.db_time_after(60)))
    async with sessionmaker.begin() as s:
        assert await claim_next_step(s, "a", 5) is None


async def test_claim_leases_step_records_attempt_and_starts_run(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(1))
    c = await claim(sessionmaker, "a")
    [step] = await load_steps(sessionmaker, run_id)
    assert (step.status, step.lease_owner, step.attempts) == (StepStatus.RUNNING, "a", 1)
    assert c.attempt == 1
    assert (await load_run(sessionmaker, run_id)).status == RunStatus.RUNNING
    async with sessionmaker() as s:
        [attempt] = await repo.list_attempts(s, step.id)
    assert (attempt.worker_id, attempt.status) == ("a", AttemptStatus.RUNNING)


async def test_run_once_returns_stored_result_on_retry(sessionmaker: SM) -> None:
    await submit(sessionmaker, demo.chain_spec(1))
    c = await claim(sessionmaker, "a")
    calls = 0

    async def effect(session: AsyncSession) -> JsonDict:
        nonlocal calls
        calls += 1
        return {"value": calls}

    first = await run_once(sessionmaker, "k1", c.step_id, effect)
    second = await run_once(sessionmaker, "k1", c.step_id, effect)
    assert first == second == {"value": 1}
    assert calls == 1


async def test_run_once_concurrent_callers_commit_one_side_effect(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(1))
    c = await claim(sessionmaker, "a")

    async def slow_ledger(session: AsyncSession) -> JsonDict:
        row = DemoSideEffect(run_id=run_id, step_seq=0, worker_id="x", attempt=1)
        session.add(row)
        await session.flush()
        await asyncio.sleep(0.3)
        return {"ledger_id": row.id}

    results = await asyncio.gather(
        run_once(sessionmaker, "k", c.step_id, slow_ledger),
        run_once(sessionmaker, "k", c.step_id, slow_ledger),
    )
    assert results[0] == results[1]
    async with sessionmaker() as s:
        assert await demo.ledger_counts(s, run_id) == {0: 1}


async def test_renew_lease_rejects_stale_owner(sessionmaker: SM) -> None:
    await submit(sessionmaker, demo.chain_spec(1))
    c = await claim(sessionmaker, "a")
    stale = Claim(**{**c.__dict__, "attempt": c.attempt + 1})
    async with sessionmaker.begin() as s:
        assert await renew_lease(s, c, 5)
        assert not await renew_lease(s, stale, 5)


async def test_heartbeat_keeps_long_step_leased(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(1, sleep_s=1.0))
    short_lease = EngineConfig(0.6, 0.05, 0.01, 0.05)
    await drive(sessionmaker, [run_id], workers=2, config=short_lease)
    [step] = await load_steps(sessionmaker, run_id)
    assert (step.status, step.attempts) == (StepStatus.SUCCEEDED, 1)


async def test_lease_lost_is_reported_to_heartbeat(sessionmaker: SM) -> None:
    await submit(sessionmaker, demo.chain_spec(1))
    c = await claim(sessionmaker, "a")
    await expire_leases(sessionmaker)
    await reap(sessionmaker)
    lost = asyncio.Event()
    async with asyncio.timeout(2):
        await heartbeat(sessionmaker, c, 0.3, lost.set)
    assert lost.is_set()


async def test_reaper_requeues_expired_step_and_marks_attempt_lost(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(1))
    await claim(sessionmaker, "a")
    await expire_leases(sessionmaker)
    assert await reap(sessionmaker) == 1
    [step] = await load_steps(sessionmaker, run_id)
    assert (step.status, step.lease_owner) == (StepStatus.PENDING, None)
    async with sessionmaker() as s:
        [attempt] = await repo.list_attempts(s, step.id)
    assert attempt.status == AttemptStatus.LOST


async def test_reaper_dead_letters_when_attempts_exhausted(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(1, max_attempts=1))
    await claim(sessionmaker, "a")
    await expire_leases(sessionmaker)
    await reap(sessionmaker)
    [step] = await load_steps(sessionmaker, run_id)
    assert step.status == StepStatus.DEAD_LETTER
    assert (await load_run(sessionmaker, run_id)).status == RunStatus.FAILED


async def test_zombie_worker_cannot_complete_reclaimed_step(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(1))
    zombie = await claim(sessionmaker, "zombie")
    await expire_leases(sessionmaker)
    await reap(sessionmaker)
    await claim(sessionmaker, "fresh")
    async with sessionmaker.begin() as s:
        with pytest.raises(repo.LeaseLostError):
            await repo.transition_step(s, zombie.step_id, StepStatus.SUCCEEDED, fence=zombie.fence)
    await record_success(sessionmaker, zombie, StepResult({"zombie": True}))
    [step] = await load_steps(sessionmaker, run_id)
    assert (step.status, step.lease_owner, step.output) == (StepStatus.RUNNING, "fresh", None)


async def test_worker_completes_chain_with_each_side_effect_once(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(5, sleep_s=0.01))
    await drive(sessionmaker, [run_id])
    run = await load_run(sessionmaker, run_id)
    assert (run.status, run.result) == (RunStatus.SUCCEEDED, {"steps": 5})
    async with sessionmaker() as s:
        assert await demo.ledger_counts(s, run_id) == dict.fromkeys(range(5), 1)


async def test_failed_attempts_retry_with_backoff_then_succeed(sessionmaker: SM) -> None:
    spec = demo.chain_spec(2, sleep_s=0.0, max_attempts=3, fail_first_attempts=2)
    run_id = await submit(sessionmaker, spec)
    await drive(sessionmaker, [run_id])
    assert (await load_run(sessionmaker, run_id)).status == RunStatus.SUCCEEDED
    for step in await load_steps(sessionmaker, run_id):
        async with sessionmaker() as s:
            statuses = [a.status for a in await repo.list_attempts(s, step.id)]
        assert statuses == [AttemptStatus.FAILED, AttemptStatus.FAILED, AttemptStatus.SUCCEEDED]


async def test_exhausted_retries_dead_letter_and_fail_run(sessionmaker: SM) -> None:
    spec = demo.chain_spec(3, sleep_s=0.0, max_attempts=3, fail_first_attempts=3)
    run_id = await submit(sessionmaker, spec)
    await drive(sessionmaker, [run_id])
    run = await load_run(sessionmaker, run_id)
    [step] = await load_steps(sessionmaker, run_id)
    assert run.status == RunStatus.FAILED
    assert run.error is not None and "injected failure" in run.error
    assert (step.status, step.attempts) == (StepStatus.DEAD_LETTER, 3)


async def test_unknown_step_kind_dead_letters_without_retry(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, StepSpec("no.such.kind", {}, max_attempts=5))
    await drive(sessionmaker, [run_id])
    [step] = await load_steps(sessionmaker, run_id)
    assert (step.status, step.attempts) == (StepStatus.DEAD_LETTER, 1)


async def test_crash_after_side_effect_is_not_repeated(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(2, sleep_s=0.0))
    crashed = await claim(sessionmaker, "crashed")
    ctx = StepContext(crashed, sessionmaker)
    await ctx.once("ledger", partial(demo.write_ledger, crashed))
    await expire_leases(sessionmaker)
    await drive(sessionmaker, [run_id])
    assert (await load_run(sessionmaker, run_id)).status == RunStatus.SUCCEEDED
    first = (await load_steps(sessionmaker, run_id))[0]
    async with sessionmaker() as s:
        assert await demo.ledger_counts(s, run_id) == {0: 1, 1: 1}
        writer_attempt = await s.scalar(
            select(DemoSideEffect.attempt).where(DemoSideEffect.step_seq == 0)
        )
    assert (first.attempts, writer_attempt) == (2, 1)


async def test_graceful_stop_finishes_current_step(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, demo.chain_spec(50, sleep_s=0.1))
    worker = Worker("w", sessionmaker, default_handlers(), CONFIG)
    task = asyncio.create_task(worker.run())
    async with asyncio.timeout(10):
        while not any(
            s.status == StepStatus.RUNNING for s in await load_steps(sessionmaker, run_id)
        ):
            await asyncio.sleep(0.02)
    worker.stop.set()
    async with asyncio.timeout(5):
        await task
    steps = await load_steps(sessionmaker, run_id)
    assert not any(s.status == StepStatus.RUNNING for s in steps)
    assert steps[0].status == StepStatus.SUCCEEDED
    assert (await load_run(sessionmaker, run_id)).status == RunStatus.RUNNING
