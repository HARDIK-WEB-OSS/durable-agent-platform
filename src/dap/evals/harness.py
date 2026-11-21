"""Run every (task, arm) pair through the durable agent, grade it, store it in eval_results.

Resumable: runs are tagged with {eval, task_id, arm}. Re-running the same eval name skips
pairs that already have a result, picks up runs that are still in flight (they are durable
steps, so an interrupted harness loses nothing), and grades runs that finished while no
harness was watching.

Outcomes:
    correct / incorrect   the agent finished, or hit its own limits (max turns, timeout,
                          budget, context), and the answer was graded
    error                 infrastructure failed (provider rate limit or outage, dead-lettered
                          step). Not a wrong answer: excluded from accuracy, retried later
                          with retry_errors=True.

Latency is split honestly: llm_s and tool_s are time spent inside calls; elapsed_s is the
wall clock from the run's first claim to its end and includes queueing. Keep concurrency=1
for local arms, because Ollama serves one request at a time and queueing inside it would
count as model latency.
"""

import asyncio
import logging
import random
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.agent.loop import AgentRuntime, submit_agent_run
from dap.db.models import EvalResult, Event, JsonDict, Run
from dap.engine.runner import EngineConfig
from dap.engine.worker import Worker
from dap.evals.graders import Grade, GradingContext, grade
from dap.evals.split import Split, SplitTask
from dap.states import RunStatus

log = logging.getLogger(__name__)

Outcome = Literal["correct", "incorrect", "error"]
LIVE = (RunStatus.PENDING, RunStatus.RUNNING)
AGENT_STOP_PREFIX = "stopped:"


class SplitGuardError(Exception):
    pass


def check_splits(splits: Sequence[Split], final_test: bool) -> None:
    if "test" in splits and not final_test:
        raise SplitGuardError(
            "the test split is reserved for the final benchmark (Phase 8); "
            "pass final_test=True only for that run"
        )


@dataclass(frozen=True)
class EvalConfig:
    name: str
    max_turns: int = 8
    concurrency: int = 1
    retry_errors: bool = False
    stop_after_errors: int = 5  # consecutive infrastructure errors before giving up on an arm
    poll_s: float = 0.5


@dataclass(frozen=True)
class Metrics:
    paid_usd: float = 0.0
    shadow_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    llm_s: float = 0.0
    tool_s: float = 0.0
    elapsed_s: float | None = None


async def run_metrics(session: AsyncSession, run_id: uuid.UUID) -> Metrics:
    events = (
        await session.execute(
            select(Event.type, Event.payload, Event.created_at)
            .where(Event.run_id == run_id)
            .order_by(Event.id)
        )
    ).all()
    llm = [p for t, p, _ in events if t == "llm.call"]
    tools = [p for t, p, _ in events if t == "tool.call"]
    started = next((at for t, _, at in events if t == "run.running"), None)
    ended = next(
        (at for t, _, at in reversed(events) if t in ("run.succeeded", "run.failed")), None
    )
    elapsed = (ended - started).total_seconds() if started and ended else None
    return Metrics(
        paid_usd=sum(p.get("cost_usd", 0.0) for p in llm),
        shadow_usd=sum(p.get("shadow_cost_usd", 0.0) for p in llm),
        tokens_in=sum(p.get("tokens_in", 0) for p in llm),
        tokens_out=sum(p.get("tokens_out", 0) for p in llm),
        llm_calls=sum(p.get("calls", 1) for p in llm),
        tool_calls=len(tools),
        llm_s=sum(p.get("wall_s", 0.0) for p in llm),
        tool_s=sum(p.get("latency_s", 0.0) for p in tools),
        elapsed_s=elapsed,
    )


async def judge(run: Run, task: SplitTask, ctx: GradingContext) -> tuple[Outcome, Grade]:
    if run.status == RunStatus.SUCCEEDED:
        answer = (run.result or {}).get("answer")
        result = await grade(task.task, answer, ctx)
        return ("correct" if result.correct else "incorrect"), result
    error = run.error or f"run {run.status.value}"
    if run.status == RunStatus.FAILED and error.startswith(AGENT_STOP_PREFIX):
        return "incorrect", Grade(False, error)
    return "error", Grade(False, error)


async def store(
    session: AsyncSession,
    cfg: EvalConfig,
    task: SplitTask,
    arm: str,
    run: Run,
    outcome: Outcome,
    result: Grade,
    metrics: Metrics,
) -> None:
    values: JsonDict = {
        "eval_name": cfg.name,
        "task_id": task.task.id,
        "dataset": task.task.dataset,
        "split": task.split,
        "arm": arm,
        "run_id": run.id,
        "outcome": outcome,
        "answer": (run.result or {}).get("answer"),
        "detail": result.detail,
        "turns": (run.result or {}).get("turns"),
        **metrics.__dict__,
    }
    stmt = insert(EvalResult).values(**values)
    update = {k: stmt.excluded[k] for k in values if k not in ("eval_name", "task_id", "arm")}
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=["eval_name", "task_id", "arm"],
            set_={**update, "updated_at": func.now()},
        )
    )


async def existing_outcomes(
    sm: async_sessionmaker[AsyncSession], name: str, arm: str
) -> dict[str, tuple[str, uuid.UUID]]:
    async with sm() as s:
        rows = await s.execute(
            select(EvalResult.task_id, EvalResult.outcome, EvalResult.run_id).where(
                EvalResult.eval_name == name, EvalResult.arm == arm
            )
        )
    return {task_id: (outcome, run_id) for task_id, outcome, run_id in rows}


async def latest_runs(sm: async_sessionmaker[AsyncSession], name: str, arm: str) -> dict[str, Run]:
    async with sm() as s:
        runs = (
            await s.scalars(
                select(Run)
                .where(Run.input["eval"].astext == name, Run.input["arm"].astext == arm)
                .order_by(Run.created_at)
            )
        ).all()
    return {run.input["task_id"]: run for run in runs}  # later runs overwrite earlier ones


async def evaluate_arm(
    sm: async_sessionmaker[AsyncSession],
    runtime: AgentRuntime,
    ctx: GradingContext,
    tasks: Sequence[SplitTask],
    arm: str,
    cfg: EvalConfig,
    engine_config: EngineConfig,
    report: Callable[[str], None] = print,
) -> Counter[str]:
    done = await existing_outcomes(sm, cfg.name, arm)
    retry = {"error"} if cfg.retry_errors else set()
    todo = [t for t in tasks if t.task.id not in done or done[t.task.id][0] in retry]
    previous = await latest_runs(sm, cfg.name, arm)
    counts: Counter[str] = Counter()
    if not todo:
        report(f"{arm}: nothing to do ({len(tasks)} already graded)")
        return counts

    in_flight: dict[uuid.UUID, SplitTask] = {}
    queue: list[SplitTask] = []
    for t in todo:
        run = previous.get(t.task.id)
        graded_run = done[t.task.id][1] if t.task.id in done else None
        if run is not None and run.id != graded_run:
            in_flight[run.id] = t  # still running, or finished while nobody graded it
        else:
            queue.append(t)
    report(f"{arm}: {len(todo)} to do ({len(in_flight)} resumed), {len(done)} already graded")

    workers = [
        Worker(f"eval-{arm}-{i}", sm, runtime.handlers(), engine_config, rng=random.Random(i))
        for i in range(cfg.concurrency)
    ]
    worker_tasks = [asyncio.create_task(w.run()) for w in workers]
    consecutive_errors = 0
    stopped_early = False
    try:
        while in_flight or (queue and not stopped_early):
            while queue and not stopped_early and len(in_flight) < cfg.concurrency:
                t = queue.pop(0)
                async with sm.begin() as s:
                    run = await submit_agent_run(
                        s,
                        t.task.prompt,
                        arm,
                        tools=list(t.task.tools),
                        sql_db=t.task.sql_db,
                        max_turns=cfg.max_turns,
                        tags={"eval": cfg.name, "task_id": t.task.id},
                    )
                in_flight[run.id] = t
            await asyncio.sleep(cfg.poll_s)
            async with sm() as s:
                finished = (
                    await s.scalars(
                        select(Run).where(Run.id.in_(list(in_flight)), Run.status.not_in(LIVE))
                    )
                ).all()
            for run in finished:
                t = in_flight.pop(run.id)
                outcome, result = await judge(run, t, ctx)
                async with sm.begin() as s:
                    metrics = await run_metrics(s, run.id)
                    await store(s, cfg, t, arm, run, outcome, result, metrics)
                counts[outcome] += 1
                consecutive_errors = consecutive_errors + 1 if outcome == "error" else 0
                n = sum(counts.values())
                report(
                    f"{arm} [{n}/{len(todo)}] {t.task.id:<12} {outcome:<9} "
                    f"{result.detail[:70]} | llm {metrics.llm_s:.1f}s "
                    f"${metrics.paid_usd:.6f}"
                )
                if consecutive_errors >= cfg.stop_after_errors and not stopped_early:
                    stopped_early = True
                    report(
                        f"{arm}: {consecutive_errors} infrastructure errors in a row; not "
                        f"submitting more ({len(queue)} left). Re-run later with retry_errors."
                    )
    finally:
        for w in workers:
            w.stop.set()
        await asyncio.gather(*worker_tasks)
    return counts


@dataclass(frozen=True)
class Row:
    arm: str
    dataset: str
    n: int
    correct: int
    errors: int
    paid_usd: float
    shadow_usd: float
    llm_s: float

    @property
    def accuracy(self) -> float | None:
        graded = self.n - self.errors
        return self.correct / graded if graded else None


async def summarize(
    sm: async_sessionmaker[AsyncSession], name: str, splits: Sequence[str]
) -> list[Row]:
    r = EvalResult
    stmt = (
        select(
            r.arm,
            r.dataset,
            func.count(),
            func.count().filter(r.outcome == "correct"),
            func.count().filter(r.outcome == "error"),
            func.coalesce(func.avg(r.paid_usd), 0.0),
            func.coalesce(func.avg(r.shadow_usd), 0.0),
            func.coalesce(func.avg(r.llm_s), 0.0),
        )
        .where(r.eval_name == name, r.split.in_(list(splits)))
        .group_by(r.arm, r.dataset)
        .order_by(r.arm, r.dataset)
    )
    async with sm() as s:
        return [Row(*row) for row in (await s.execute(stmt)).all()]


def format_summary(rows: Sequence[Row]) -> str:
    lines = [
        f"{'arm':<8}{'dataset':<8}{'n':>5}{'acc':>8}{'errors':>8}"
        f"{'paid $/task':>13}{'shadow $/task':>15}{'llm s/task':>12}"
    ]
    for row in rows:
        acc = "n/a" if row.accuracy is None else f"{row.accuracy:.1%}"
        lines.append(
            f"{row.arm:<8}{row.dataset:<8}{row.n:>5}{acc:>8}{row.errors:>8}"
            f"{row.paid_usd:>13.6f}{row.shadow_usd:>15.8f}{row.llm_s:>12.1f}"
        )
    return "\n".join(lines)
