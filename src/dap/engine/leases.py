import asyncio
import logging
import time
from collections.abc import Callable

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.db import repo
from dap.db.models import Step
from dap.engine.queue import Claim, requeue_or_dead_letter
from dap.states import AttemptStatus, StepStatus

log = logging.getLogger(__name__)


async def renew_lease(session: AsyncSession, claim: Claim, lease_seconds: float) -> bool:
    stmt = (
        update(Step)
        .where(
            Step.id == claim.step_id,
            Step.status == StepStatus.RUNNING,
            Step.lease_owner == claim.worker_id,
            Step.attempts == claim.attempt,
        )
        .values(lease_expires_at=repo.db_time_after(lease_seconds))
        .returning(Step.id)
        .execution_options(synchronize_session=False)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def heartbeat(
    sessionmaker: async_sessionmaker[AsyncSession],
    claim: Claim,
    lease_seconds: float,
    on_lost: Callable[[], None],
) -> None:
    """Renew the lease every lease/4 and give up before it can expire.

    The local deadline is a lower bound on the DB-side expiry, because each renewal sets the
    expiry from DB now(), which is no earlier than the moment the renewal was sent.
    """
    interval = lease_seconds / 4
    margin = 1.5 * interval
    deadline = time.monotonic() + lease_seconds
    while True:
        await asyncio.sleep(interval)
        sent = time.monotonic()
        budget = deadline - sent - margin
        if budget <= 0:
            log.error("could not renew lease on step %s in time; abandoning it", claim.step_id)
            on_lost()
            return
        try:
            async with asyncio.timeout(budget), sessionmaker.begin() as session:
                renewed = await renew_lease(session, claim, lease_seconds)
        except Exception:
            log.exception("heartbeat failed for step %s; will retry", claim.step_id)
            continue
        if not renewed:
            on_lost()
            return
        deadline = sent + lease_seconds


def no_delay(attempt: int) -> float:
    return 0.0


async def reap_expired_leases(
    session: AsyncSession, retry_delay: Callable[[int], float] = no_delay, batch: int = 10
) -> int:
    stmt = (
        select(Step)
        .where(Step.status == StepStatus.RUNNING, Step.lease_expires_at < func.now())
        .limit(batch)
        .with_for_update(skip_locked=True)
    )
    steps = (await session.scalars(stmt)).all()
    for step in steps:
        claim = Claim(
            step_id=step.id,
            run_id=step.run_id,
            seq=step.seq,
            kind=step.kind,
            input=step.input,
            attempt=step.attempts,
            max_attempts=step.max_attempts,
            worker_id=step.lease_owner or "",
        )
        await repo.finish_attempt(session, step.id, step.attempts, AttemptStatus.LOST)
        await requeue_or_dead_letter(
            session, claim, "lease expired", delay_seconds=retry_delay(claim.attempt)
        )
        log.warning("reclaimed step %s seq=%s from %s", claim.step_id, claim.seq, claim.worker_id)
    return len(steps)
