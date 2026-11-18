import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dap.db.models import Event, JsonDict, Run, Step
from dap.states import (
    RUN_TRANSITIONS,
    STEP_TRANSITIONS,
    IllegalTransitionError,
    RunStatus,
    StepStatus,
    allowed_sources,
)


class NotFoundError(Exception):
    pass


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
    session: AsyncSession, step_id: uuid.UUID, target: StepStatus, **fields: Any
) -> Step:
    stmt = (
        update(Step)
        .where(Step.id == step_id, Step.status.in_(allowed_sources(STEP_TRANSITIONS, target)))
        .values(status=target, updated_at=func.now(), **fields)
        .returning(Step)
        .execution_options(populate_existing=True)
    )
    step = (await session.scalars(stmt)).one_or_none()
    if step is None:
        current = await session.scalar(select(Step.status).where(Step.id == step_id))
        if current is None:
            raise NotFoundError(f"step {step_id}")
        raise IllegalTransitionError(current, target)
    await append_event(session, step.run_id, f"step.{target.value}", step_id=step_id)
    return step
