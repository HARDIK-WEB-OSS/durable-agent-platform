import asyncio
import random
import subprocess
import uuid
from contextlib import ExitStack
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.config import Settings
from dap.db.models import DemoSideEffect, Run, Step, StepAttempt
from dap.engine import demo
from dap.engine.runner import submit_run
from dap.engine.worker import spawn_worker_process
from dap.states import AttemptStatus, RunStatus

pytestmark = pytest.mark.chaos

ROOT = Path(__file__).resolve().parents[2]
RUNS, STEPS, WORKERS, MAX_KILLS, MIN_KILLS = 20, 10, 3, 25, 10
STEP_SLEEP_S = 0.2
# A worker takes ~12 s to start (importing LiteLLM for the startup price check). Killing a
# worker before it logs "started" tests nothing, so only ready workers are victims.
READY_MARKER = b" started"
KILL_GAP_S = (1.0, 3.0)
DEADLINE_S = 360
SM = async_sessionmaker[AsyncSession]


async def unfinished_runs(sm: SM, run_ids: list[uuid.UUID]) -> int:
    terminal = (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)
    async with sm() as s:
        stmt = select(func.count()).where(Run.id.in_(run_ids), Run.status.not_in(terminal))
        return int(await s.scalar(stmt) or 0)


async def replayed_side_effects(s: AsyncSession) -> int:
    stmt = (
        select(func.count())
        .select_from(DemoSideEffect)
        .join(Step, (Step.run_id == DemoSideEffect.run_id) & (Step.seq == DemoSideEffect.step_seq))
        .where(DemoSideEffect.attempt < Step.attempts)
    )
    return int(await s.scalar(stmt) or 0)


@pytest.mark.timeout(DEADLINE_S + 60)
async def test_random_worker_kills_never_lose_or_duplicate_work(
    settings: Settings, sessionmaker: SM, tmp_path: Path
) -> None:
    rng = random.Random(2026)
    async with sessionmaker.begin() as s:
        runs = [
            await submit_run(
                s, "chaos", demo.chain_spec(STEPS, sleep_s=STEP_SLEEP_S, max_attempts=10)
            )
            for _ in range(RUNS)
        ]
    run_ids = [r.id for r in runs]
    env = {
        "POSTGRES_DB": settings.postgres_db,
        "WORKER_LEASE_SECONDS": "1.5",
        "WORKER_POLL_SECONDS": "0.1",
        "RETRY_BASE_SECONDS": "0.05",
    }
    procs: dict[str, subprocess.Popen[bytes]] = {}
    kills = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + DEADLINE_S

    with ExitStack() as files:

        def spawn(name: str) -> None:
            log = files.enter_context(open(tmp_path / f"{name}.log", "wb"))
            procs[name] = spawn_worker_process(name, env, ROOT, stdout=log)

        def ready(name: str) -> bool:
            return READY_MARKER in (tmp_path / f"{name}.log").read_bytes()

        for i in range(WORKERS):
            spawn(f"w{i}")
        try:
            while await unfinished_runs(sessionmaker, run_ids):
                assert loop.time() < deadline, f"runs not finished; logs in {tmp_path}"
                await asyncio.sleep(rng.uniform(*KILL_GAP_S))
                victims = sorted(name for name in procs if ready(name))
                if kills < MAX_KILLS and victims:
                    procs.pop(rng.choice(victims)).kill()
                    kills += 1
                    spawn(f"w{WORKERS + kills - 1}")
        finally:
            for proc in procs.values():
                proc.kill()
                proc.wait()

    async with sessionmaker() as s:
        statuses = (await s.scalars(select(Run.status).where(Run.id.in_(run_ids)))).all()
        counts = [await demo.ledger_counts(s, run_id) for run_id in run_ids]
        lost = await s.scalar(select(func.count()).where(StepAttempt.status == AttemptStatus.LOST))
        replays = await replayed_side_effects(s)

    print(f"\nkills={kills} lost_attempts={lost} idempotent_replays={replays}")
    assert all(status == RunStatus.SUCCEEDED for status in statuses)
    assert all(c == dict.fromkeys(range(STEPS), 1) for c in counts)
    assert kills >= MIN_KILLS
    assert lost is not None and lost > 0
