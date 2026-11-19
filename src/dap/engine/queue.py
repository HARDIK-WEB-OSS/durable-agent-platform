import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dap.db import repo
from dap.db.models import JsonDict, Run, Step
from dap.db.repo import Fence
from dap.states import RunStatus, StepStatus

LIVE_RUN_STATUSES = (RunStatus.PENDING, RunStatus.RUNNING)


@dataclass(frozen=True)
class Claim:
    step_id: uuid.UUID
    run_id: uuid.UUID
    seq: int
    kind: str
    input: JsonDict
    attempt: int
    max_attempts: int
    worker_id: str

    @property
    def fence(self) -> Fence:
        return Fence(self.worker_id, self.attempt)


async def claim_next_step(
    session: AsyncSession, worker_id: str, lease_seconds: float
) -> Claim | None:
    stmt = (
        select(Step.id)
        .join(Run, Run.id == Step.run_id)
        .where(
            Step.status == StepStatus.PENDING,
            Step.available_at <= func.now(),
            Run.status.in_(LIVE_RUN_STATUSES),
        )
        .order_by(Step.available_at)
        .limit(1)
        .with_for_update(of=Step, skip_locked=True)
    )
    step_id = await session.scalar(stmt)
    if step_id is None:
        return None
    step = await repo.transition_step(
        session,
        step_id,
        StepStatus.RUNNING,
        event_payload={"worker_id": worker_id},
        attempts=Step.attempts + 1,
        lease_owner=worker_id,
        lease_expires_at=repo.db_time_after(lease_seconds),
    )
    await repo.start_attempt(session, step.id, step.attempts, worker_id)
    await repo.start_run_if_pending(session, step.run_id)
    return Claim(
        step_id=step.id,
        run_id=step.run_id,
        seq=step.seq,
        kind=step.kind,
        input=step.input,
        attempt=step.attempts,
        max_attempts=step.max_attempts,
        worker_id=worker_id,
    )


async def requeue_or_dead_letter(
    session: AsyncSession,
    claim: Claim,
    error: str,
    delay_seconds: float,
    permanent: bool = False,
) -> StepStatus:
    released: dict[str, Any] = {"lease_owner": None, "lease_expires_at": None, "error": error}
    if permanent or claim.attempt >= claim.max_attempts:
        await repo.transition_step(
            session, claim.step_id, StepStatus.DEAD_LETTER, fence=claim.fence, **released
        )
        await repo.finish_run(
            session,
            claim.run_id,
            RunStatus.FAILED,
            error=f"step {claim.seq} dead-lettered: {error}",
        )
        return StepStatus.DEAD_LETTER
    await repo.transition_step(
        session,
        claim.step_id,
        StepStatus.PENDING,
        fence=claim.fence,
        event_payload={"retry_in_s": round(delay_seconds, 3)},
        available_at=repo.db_time_after(delay_seconds),
        **released,
    )
    return StepStatus.PENDING
