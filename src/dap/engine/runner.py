import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.backoff import backoff_seconds
from dap.config import Settings
from dap.db import repo
from dap.db.models import JsonDict, Run
from dap.db.repo import LeaseLostError
from dap.engine.effects import ExternalEffect, record_once
from dap.engine.idempotency import Effect, run_once
from dap.engine.leases import heartbeat
from dap.engine.queue import Claim, requeue_or_dead_letter
from dap.states import AttemptStatus, RunStatus, StepStatus

log = logging.getLogger(__name__)


class PermanentError(Exception):
    pass


@dataclass(frozen=True)
class StepSpec:
    kind: str
    input: JsonDict = field(default_factory=dict)
    max_attempts: int = 3


@dataclass(frozen=True)
class StepResult:
    output: JsonDict
    next_step: StepSpec | None = None
    run_result: JsonDict | None = None
    run_error: str | None = None

    def __post_init__(self) -> None:
        endings = [self.next_step, self.run_result, self.run_error]
        if sum(e is not None for e in endings) > 1:
            raise ValueError("a step schedules a next step, finishes the run, or fails it")


@dataclass(frozen=True)
class StepContext:
    claim: Claim
    sessionmaker: async_sessionmaker[AsyncSession]

    async def once(self, name: str, effect: Effect) -> JsonDict:
        key = f"{self.claim.step_id}:{name}"
        return await run_once(self.sessionmaker, key, self.claim.step_id, effect)

    async def external(self, name: str, effect: ExternalEffect) -> JsonDict:
        """Slow or outside effect: at-least-once, recorded once (see engine/effects.py)."""
        key = f"{self.claim.step_id}:{name}"
        return await record_once(self.sessionmaker, key, self.claim.step_id, effect)


Handler = Callable[[StepContext], Awaitable[StepResult]]


@dataclass(frozen=True)
class EngineConfig:
    lease_seconds: float
    poll_seconds: float
    retry_base_seconds: float
    retry_cap_seconds: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "EngineConfig":
        return cls(
            lease_seconds=settings.worker_lease_seconds,
            poll_seconds=settings.worker_poll_seconds,
            retry_base_seconds=settings.retry_base_seconds,
            retry_cap_seconds=settings.retry_cap_seconds,
        )


async def submit_run(session: AsyncSession, task: str, first: StepSpec) -> Run:
    run = await repo.create_run(session, task, first.input)
    await repo.add_step(session, run.id, first.kind, first.input, first.max_attempts)
    return run


async def execute_claim(
    sessionmaker: async_sessionmaker[AsyncSession],
    claim: Claim,
    handlers: Mapping[str, Handler],
    config: EngineConfig,
    rng: random.Random,
) -> None:
    ctx = StepContext(claim, sessionmaker)
    work = asyncio.create_task(call_handler(handlers, ctx))
    lease_lost = asyncio.Event()

    def on_lost() -> None:
        lease_lost.set()
        work.cancel()

    beat = asyncio.create_task(heartbeat(sessionmaker, claim, config.lease_seconds, on_lost))
    try:
        result = await work
    except asyncio.CancelledError:
        if not lease_lost.is_set():
            raise
        log.warning("lease lost on step %s; abandoning attempt %s", claim.step_id, claim.attempt)
        return
    except Exception as exc:
        await record_failure(sessionmaker, claim, exc, config, rng)
        return
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
    await record_success(sessionmaker, claim, result)


async def call_handler(handlers: Mapping[str, Handler], ctx: StepContext) -> StepResult:
    handler = handlers.get(ctx.claim.kind)
    if handler is None:
        raise PermanentError(f"no handler for step kind {ctx.claim.kind!r}")
    return await handler(ctx)


async def record_success(
    sessionmaker: async_sessionmaker[AsyncSession], claim: Claim, result: StepResult
) -> None:
    try:
        async with sessionmaker.begin() as session:
            await repo.transition_step(
                session,
                claim.step_id,
                StepStatus.SUCCEEDED,
                fence=claim.fence,
                output=result.output,
                lease_owner=None,
                lease_expires_at=None,
                error=None,
            )
            await repo.finish_attempt(
                session, claim.step_id, claim.attempt, AttemptStatus.SUCCEEDED
            )
            if result.next_step is not None:
                spec = result.next_step
                await repo.add_step(session, claim.run_id, spec.kind, spec.input, spec.max_attempts)
            elif result.run_error is not None:
                await repo.finish_run(
                    session, claim.run_id, RunStatus.FAILED, error=result.run_error
                )
            else:
                final = result.run_result if result.run_result is not None else result.output
                await repo.finish_run(session, claim.run_id, RunStatus.SUCCEEDED, result=final)
    except LeaseLostError:
        log.warning(
            "discarding result of step %s attempt %s: lease lost", claim.step_id, claim.attempt
        )


async def record_failure(
    sessionmaker: async_sessionmaker[AsyncSession],
    claim: Claim,
    exc: Exception,
    config: EngineConfig,
    rng: random.Random,
) -> None:
    error = f"{type(exc).__name__}: {exc}"
    delay = backoff_seconds(claim.attempt, config.retry_base_seconds, config.retry_cap_seconds, rng)
    try:
        async with sessionmaker.begin() as session:
            await repo.finish_attempt(
                session, claim.step_id, claim.attempt, AttemptStatus.FAILED, error
            )
            outcome = await requeue_or_dead_letter(
                session, claim, error, delay, permanent=isinstance(exc, PermanentError)
            )
        log.warning(
            "step %s attempt %s failed (%s) -> %s", claim.step_id, claim.attempt, error, outcome
        )
    except LeaseLostError:
        log.warning("step %s attempt %s failed after lease was lost", claim.step_id, claim.attempt)
