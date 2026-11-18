import itertools
import uuid

import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from dap.db import repo
from dap.db.models import Event, Run, Step
from dap.states import (
    RUN_TRANSITIONS,
    STEP_TRANSITIONS,
    IllegalTransitionError,
    RunStatus,
    StepStatus,
)

pytestmark = pytest.mark.integration

RUN_PATHS = {
    RunStatus.PENDING: [],
    RunStatus.RUNNING: [RunStatus.RUNNING],
    RunStatus.SUCCEEDED: [RunStatus.RUNNING, RunStatus.SUCCEEDED],
    RunStatus.FAILED: [RunStatus.RUNNING, RunStatus.FAILED],
    RunStatus.CANCELLED: [RunStatus.CANCELLED],
}

STEP_PATHS = {
    StepStatus.PENDING: [],
    StepStatus.RUNNING: [StepStatus.RUNNING],
    StepStatus.SUCCEEDED: [StepStatus.RUNNING, StepStatus.SUCCEEDED],
    StepStatus.DEAD_LETTER: [StepStatus.RUNNING, StepStatus.DEAD_LETTER],
    StepStatus.CANCELLED: [StepStatus.CANCELLED],
}


async def run_in(session: AsyncSession, status: RunStatus) -> Run:
    run = await repo.create_run(session, "demo task")
    for target in RUN_PATHS[status]:
        run = await repo.transition_run(session, run.id, target)
    return run


async def step_in(session: AsyncSession, status: StepStatus) -> Step:
    run = await repo.create_run(session, "demo task")
    step = await repo.add_step(session, run.id, "fake")
    for target in STEP_PATHS[status]:
        step = await repo.transition_step(session, step.id, target)
    return step


@pytest.mark.parametrize(("current", "target"), list(itertools.product(RunStatus, repeat=2)))
async def test_run_transitions_in_db(
    session: AsyncSession, current: RunStatus, target: RunStatus
) -> None:
    run = await run_in(session, current)
    if target in RUN_TRANSITIONS[current]:
        updated = await repo.transition_run(session, run.id, target)
        assert updated.status == target
    else:
        with pytest.raises(IllegalTransitionError):
            await repo.transition_run(session, run.id, target)
        assert (await repo.get_run(session, run.id)).status == current  # type: ignore[union-attr]


@pytest.mark.parametrize(("current", "target"), list(itertools.product(StepStatus, repeat=2)))
async def test_step_transitions_in_db(
    session: AsyncSession, current: StepStatus, target: StepStatus
) -> None:
    step = await step_in(session, current)
    if target in STEP_TRANSITIONS[current]:
        updated = await repo.transition_step(session, step.id, target)
        assert updated.status == target
    else:
        with pytest.raises(IllegalTransitionError):
            await repo.transition_step(session, step.id, target)
        assert (await repo.get_step(session, step.id)).status == current  # type: ignore[union-attr]


async def test_unknown_run_raises_not_found(session: AsyncSession) -> None:
    with pytest.raises(repo.NotFoundError):
        await repo.transition_run(session, uuid.uuid4(), RunStatus.RUNNING)


async def test_state_change_and_event_commit_together(session: AsyncSession) -> None:
    async with session.begin():
        run = await repo.create_run(session, "demo task")
        await repo.transition_run(session, run.id, RunStatus.RUNNING)
    events = await repo.list_events(session, run.id)
    assert [e.type for e in events] == ["run.created", "run.running"]


async def test_rollback_discards_state_and_events(session: AsyncSession) -> None:
    async with session.begin():
        run = await repo.create_run(session, "demo task")
    run_id = run.id
    async with session.begin():
        await repo.transition_run(session, run_id, RunStatus.RUNNING)
        await session.rollback()
    async with session.begin():
        assert (await repo.get_run(session, run_id)).status == RunStatus.PENDING  # type: ignore[union-attr]
        assert [e.type for e in await repo.list_events(session, run_id)] == ["run.created"]


async def test_steps_get_sequential_numbers(session: AsyncSession) -> None:
    run = await repo.create_run(session, "demo task")
    for _ in range(3):
        await repo.add_step(session, run.id, "fake")
    assert [s.seq for s in await repo.list_steps(session, run.id)] == [0, 1, 2]


async def test_events_are_append_only(session: AsyncSession) -> None:
    async with session.begin():
        await repo.create_run(session, "demo task")
    for stmt in (update(Event).values(type="tampered"), delete(Event)):
        with pytest.raises(DBAPIError, match="append-only"):
            async with session.begin():
                await session.execute(stmt)
