import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, NoReturn

from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dap.db.models import Event, JsonDict, Run, Step, StepAttempt
from dap.states import (
    RUN_TRANSITIONS,
    STEP_TRANSITIONS,
    AttemptStatus,
    IllegalTransitionError,
    RunStatus,
    StepStatus,
    allowed_sources,
    is_terminal,
)


class NotFoundError(Exception):
    pass


class LeaseLostError(Exception):
    pass


@dataclass(frozen=True)
class Fence:
    owner: str
    attempt: int


def db_time_after(seconds: float) -> ColumnElement[datetime]:
    return func.now() + timedelta(seconds=seconds)


async def append_event(
    session: AsyncSession,
    run_id: uuid.UUID,
    type_: str,
    payload: JsonDict | None = None,
    step_id: uuid.UUID | None = None,
) -> Event:
    event = Event(run_id=run_id, step_id=step_id, type=type_, payload=payload or {})
    session.add(event)
    await session.flush()
    return event


async def list_events(session: AsyncSession, run_id: uuid.UUID) -> Sequence[Event]:
    stmt = select(Event).where(Event.run_id == run_id).order_by(Event.id)
    return (await session.scalars(stmt)).all()


async def create_run(
    session: AsyncSession,
    task: str,
    input_: JsonDict | None = None,
    budget_usd: Decimal | None = None,
) -> Run:
    run = Run(task=task, input=input_ or {}, budget_usd=budget_usd, status=RunStatus.PENDING)
    session.add(run)
    await session.flush()
    await append_event(session, run.id, "run.created", {"task": task})
    return run


async def get_run(session: AsyncSession, run_id: uuid.UUID) -> Run | None:
    return await session.get(Run, run_id)


async def transition_run(
    session: AsyncSession, run_id: uuid.UUID, target: RunStatus, **fields: Any
) -> Run:
    stmt = (
        update(Run)
        .where(Run.id == run_id, Run.status.in_(allowed_sources(RUN_TRANSITIONS, target)))
        .values(status=target, updated_at=func.now(), **fields)
        .returning(Run)
        .execution_options(populate_existing=True)
    )
    run = (await session.scalars(stmt)).one_or_none()
    if run is None:
        current = await session.scalar(select(Run.status).where(Run.id == run_id))
        if current is None:
            raise NotFoundError(f"run {run_id}")
        raise IllegalTransitionError(current, target)
    await append_event(session, run_id, f"run.{target.value}")
    return run


async def start_run_if_pending(session: AsyncSession, run_id: uuid.UUID) -> None:
    status = await session.scalar(select(Run.status).where(Run.id == run_id))
    if status == RunStatus.PENDING:
        await transition_run(session, run_id, RunStatus.RUNNING)


async def finish_run(
    session: AsyncSession, run_id: uuid.UUID, target: RunStatus, **fields: Any
) -> None:
    status = await session.scalar(select(Run.status).where(Run.id == run_id))
    if status is None:
        raise NotFoundError(f"run {run_id}")
    if not is_terminal(RUN_TRANSITIONS, status):
        await transition_run(session, run_id, target, **fields)


async def add_step(
    session: AsyncSession,
    run_id: uuid.UUID,
    kind: str,
    input_: JsonDict | None = None,
    max_attempts: int = 3,
) -> Step:
    next_seq = await session.scalar(
        select(func.coalesce(func.max(Step.seq), -1) + 1).where(Step.run_id == run_id)
    )
    step = Step(
        run_id=run_id,
        seq=next_seq,
        kind=kind,
        input=input_ or {},
        max_attempts=max_attempts,
        status=StepStatus.PENDING,
        attempts=0,
    )
    session.add(step)
    await session.flush()
    await append_event(session, run_id, "step.created", {"kind": kind}, step_id=step.id)
    return step


async def get_step(session: AsyncSession, step_id: uuid.UUID) -> Step | None:
    return await session.get(Step, step_id)


async def list_steps(session: AsyncSession, run_id: uuid.UUID) -> Sequence[Step]:
    stmt = select(Step).where(Step.run_id == run_id).order_by(Step.seq)
    return (await session.scalars(stmt)).all()


async def transition_step(
    session: AsyncSession,
    step_id: uuid.UUID,
    target: StepStatus,
    *,
    fence: Fence | None = None,
    event_payload: JsonDict | None = None,
    **fields: Any,
) -> Step:
    conditions = [Step.id == step_id, Step.status.in_(allowed_sources(STEP_TRANSITIONS, target))]
    if fence is not None:
        conditions += [Step.lease_owner == fence.owner, Step.attempts == fence.attempt]
    stmt = (
        update(Step)
        .where(*conditions)
        .values(status=target, updated_at=func.now(), **fields)
        .returning(Step)
        .execution_options(populate_existing=True)
    )
    step = (await session.scalars(stmt)).one_or_none()
    if step is None:
        await raise_step_rejection(session, step_id, target, fence)
    await append_event(session, step.run_id, f"step.{target.value}", event_payload, step_id=step_id)
    return step


async def raise_step_rejection(
    session: AsyncSession, step_id: uuid.UUID, target: StepStatus, fence: Fence | None
) -> NoReturn:
    row = (
        await session.execute(
            select(Step.status, Step.lease_owner, Step.attempts).where(Step.id == step_id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError(f"step {step_id}")
    status, owner, attempts = row
    if fence is not None and (owner, attempts) != (fence.owner, fence.attempt):
        raise LeaseLostError(f"step {step_id} attempt {fence.attempt} by {fence.owner}")
    raise IllegalTransitionError(status, target)


async def start_attempt(
    session: AsyncSession, step_id: uuid.UUID, attempt_no: int, worker_id: str
) -> StepAttempt:
    attempt = StepAttempt(
        step_id=step_id,
        attempt_no=attempt_no,
        worker_id=worker_id,
        status=AttemptStatus.RUNNING,
    )
    session.add(attempt)
    await session.flush()
    return attempt


async def finish_attempt(
    session: AsyncSession,
    step_id: uuid.UUID,
    attempt_no: int,
    status: AttemptStatus,
    error: str | None = None,
) -> None:
    await session.execute(
        update(StepAttempt)
        .where(
            StepAttempt.step_id == step_id,
            StepAttempt.attempt_no == attempt_no,
            StepAttempt.status == AttemptStatus.RUNNING,
        )
        .values(status=status, error=error, finished_at=func.now())
        .execution_options(synchronize_session=False)
    )


async def list_attempts(session: AsyncSession, step_id: uuid.UUID) -> Sequence[StepAttempt]:
    stmt = select(StepAttempt).where(StepAttempt.step_id == step_id).order_by(StepAttempt.id)
    return (await session.scalars(stmt)).all()
