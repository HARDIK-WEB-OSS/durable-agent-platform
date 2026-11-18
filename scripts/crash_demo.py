import asyncio
import sys
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.config import get_settings
from dap.db import repo
from dap.db.models import DemoSideEffect, Step, StepAttempt
from dap.db.session import make_engine, make_sessionmaker
from dap.engine import demo
from dap.engine.runner import submit_run
from dap.engine.worker import spawn_worker_process
from dap.states import RunStatus, StepStatus

ROOT = Path(__file__).resolve().parents[1]
STEPS = 20
KILL_AFTER_DONE = 5
WORKER_ENV = {"WORKER_LEASE_SECONDS": "3", "WORKER_POLL_SECONDS": "0.2"}
SM = async_sessionmaker[AsyncSession]


async def wait_for_victim(sm: SM, run_id: uuid.UUID) -> tuple[str, int]:
    while True:
        async with sm() as s:
            steps = await repo.list_steps(s, run_id)
        done = sum(step.status == StepStatus.SUCCEEDED for step in steps)
        running = [st for st in steps if st.status == StepStatus.RUNNING and st.lease_owner]
        if done >= KILL_AFTER_DONE and running:
            return running[0].lease_owner or "", running[0].seq
        await asyncio.sleep(0.05)


async def wait_until_finished(sm: SM, run_id: uuid.UUID) -> RunStatus:
    async with asyncio.timeout(120):
        while True:
            async with sm() as s:
                run = await repo.get_run(s, run_id)
            if run is not None and run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED):
                return run.status
            await asyncio.sleep(0.2)


async def print_report(sm: SM, run_id: uuid.UUID, status: RunStatus) -> bool:
    async with sm() as s:
        counts = await demo.ledger_counts(s, run_id)
        rows = (
            await s.execute(
                select(Step.seq, StepAttempt.attempt_no, StepAttempt.worker_id, StepAttempt.status)
                .join(StepAttempt, StepAttempt.step_id == Step.id)
                .where(Step.run_id == run_id)
                .order_by(Step.seq, StepAttempt.attempt_no)
            )
        ).all()
        writers = dict(
            (
                await s.execute(
                    select(DemoSideEffect.step_seq, DemoSideEffect.attempt).where(
                        DemoSideEffect.run_id == run_id
                    )
                )
            ).all()
        )
    attempts: dict[int, list[str]] = {}
    for seq, no, worker, att_status in rows:
        attempts.setdefault(seq, []).append(f"#{no} {worker} {att_status.value}")
    print(f"\n{'seq':>3}  {'side effects':>12}  attempts")
    for seq in range(STEPS):
        replay = (
            "  <- side effect replayed from idempotency key"
            if (seq in writers and len(attempts.get(seq, [])) > writers[seq])
            else ""
        )
        print(f"{seq:>3}  {counts.get(seq, 0):>12}  {', '.join(attempts.get(seq, []))}{replay}")
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    missing = STEPS - len(counts)
    print(f"\nrun status: {status.value}")
    print(f"side-effect rows: {sum(counts.values())} / {STEPS}   duplicates: {duplicates}")
    print(f"missing: {missing}")
    return status == RunStatus.SUCCEEDED and duplicates == 0 and missing == 0


async def main() -> int:
    engine = make_engine(get_settings())
    sm = make_sessionmaker(engine)
    async with sm.begin() as s:
        run = await submit_run(s, "crash demo", demo.chain_spec(STEPS, sleep_s=0.25))
    print(f"submitted run {run.id} with {STEPS} steps; starting worker-a and worker-b")
    workers = {
        name: spawn_worker_process(name, WORKER_ENV, ROOT) for name in ("worker-a", "worker-b")
    }
    try:
        victim, seq = await wait_for_victim(sm, run.id)
        workers.pop(victim).kill()
        print(f"\n>>> HARD-KILLED {victim} while it held step {seq} (process.kill) <<<\n")
        status = await wait_until_finished(sm, run.id)
    finally:
        for proc in workers.values():
            proc.kill()
            proc.wait()
    ok = await print_report(sm, run.id, status)
    await engine.dispose()
    print("RESULT:", "PASS - no lost work, no duplicated side effects" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
