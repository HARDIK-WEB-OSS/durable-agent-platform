from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.agent.loop import AgentLimits, AgentRuntime, submit_agent_run
from dap.agent.tools.calculator import CalculatorTool
from dap.db.models import EvalResult, Run
from dap.evals.graders import GradingContext
from dap.evals.harness import (
    EvalConfig,
    SplitGuardError,
    check_splits,
    evaluate_arm,
    format_summary,
    summarize,
)
from dap.evals.loaders import Task
from dap.evals.split import SplitTask
from tests.integration.test_agent import CONFIG, MID, FixedTools, ScriptedLLM, action

pytestmark = pytest.mark.integration

SM = async_sessionmaker[AsyncSession]


def math_task(n: int, answer: str = "42", split: str = "train") -> SplitTask:
    task = Task(
        id=f"gsm8k:{n}",
        dataset="gsm8k",
        prompt=f"question {n}",
        tools=("calculator",),
        reference={"answer": answer},
    )
    return SplitTask(task, split)  # type: ignore[arg-type]


def runtime_for(llm: ScriptedLLM, arms: dict[str, object] | None = None) -> AgentRuntime:
    tools = FixedTools({"calculator": CalculatorTool()})
    return AgentRuntime(
        llm.client(), arms if arms is not None else {"mid": MID}, tools, AgentLimits()
    )  # type: ignore[arg-type]


async def run(
    sm: SM, rt: AgentRuntime, tasks: list[SplitTask], tmp_path: Path, **cfg: object
) -> dict[str, int]:
    config = EvalConfig(name="t", poll_s=0.05, **cfg)  # type: ignore[arg-type]
    ctx = GradingContext(None, tmp_path)
    counts = await evaluate_arm(sm, rt, ctx, tasks, "mid", config, CONFIG, report=lambda _: None)
    return dict(counts)


async def results(sm: SM) -> dict[str, EvalResult]:
    async with sm() as s:
        return {r.task_id: r for r in (await s.scalars(select(EvalResult))).all()}


async def test_outcomes_are_graded_and_stored_with_metrics(
    sessionmaker: SM, tmp_path: Path
) -> None:
    llm = ScriptedLLM(
        [
            action("final_answer", "42"),
            action("final_answer", "41"),
            action("calculator", "6*7"),  # max_turns=1: the next turn is refused
        ]
    )
    tasks = [math_task(1), math_task(2), math_task(3)]
    counts = await run(sessionmaker, runtime_for(llm), tasks, tmp_path, max_turns=1)
    assert counts == {"correct": 1, "incorrect": 2}
    rows = await results(sessionmaker)
    assert (rows["gsm8k:1"].outcome, rows["gsm8k:1"].answer, rows["gsm8k:1"].turns) == (
        "correct",
        "42",
        1,
    )
    assert rows["gsm8k:2"].detail == "41 != 42"
    assert rows["gsm8k:3"].outcome == "incorrect" and "max turns" in rows["gsm8k:3"].detail
    assert (rows["gsm8k:3"].llm_calls, rows["gsm8k:3"].tool_calls) == (1, 1)
    first = rows["gsm8k:1"]
    assert (first.tokens_in, first.tokens_out, first.paid_usd) == (50, 10, 0.0)
    assert first.shadow_usd > 0 and first.llm_s >= 0 and first.elapsed_s is not None


async def test_rerun_skips_graded_pairs(sessionmaker: SM, tmp_path: Path) -> None:
    llm = ScriptedLLM([action("final_answer", "42")])
    rt = runtime_for(llm)
    assert await run(sessionmaker, rt, [math_task(1)], tmp_path) == {"correct": 1}
    assert await run(sessionmaker, rt, [math_task(1)], tmp_path) == {}
    assert len(llm.requests) == 1


async def test_interrupted_run_is_resumed_not_resubmitted(sessionmaker: SM, tmp_path: Path) -> None:
    async with sessionmaker.begin() as s:
        await submit_agent_run(
            s, "question 1", "mid", tools=["calculator"], tags={"eval": "t", "task_id": "gsm8k:1"}
        )
    llm = ScriptedLLM([action("final_answer", "42")])
    assert await run(sessionmaker, runtime_for(llm), [math_task(1)], tmp_path) == {"correct": 1}
    async with sessionmaker() as s:
        assert await s.scalar(select(func.count()).select_from(Run)) == 1


async def test_infrastructure_errors_are_not_wrong_answers_and_can_be_retried(
    sessionmaker: SM, tmp_path: Path
) -> None:
    broken = runtime_for(ScriptedLLM([]), arms={})  # arm missing: steps dead-letter
    assert await run(sessionmaker, broken, [math_task(1)], tmp_path) == {"error": 1}
    first = (await results(sessionmaker))["gsm8k:1"]
    assert "not configured" in first.detail
    rows = await summarize(sessionmaker, "t", ["train"])
    assert rows[0].accuracy is None and rows[0].errors == 1

    llm = ScriptedLLM([action("final_answer", "42")])
    assert await run(sessionmaker, runtime_for(llm), [math_task(1)], tmp_path) == {}
    retried = await run(sessionmaker, runtime_for(llm), [math_task(1)], tmp_path, retry_errors=True)
    assert retried == {"correct": 1}
    second = (await results(sessionmaker))["gsm8k:1"]
    assert second.outcome == "correct" and second.run_id != first.run_id


async def test_stops_submitting_after_consecutive_errors(sessionmaker: SM, tmp_path: Path) -> None:
    broken = runtime_for(ScriptedLLM([]), arms={})
    tasks = [math_task(i) for i in range(4)]
    counts = await run(sessionmaker, broken, tasks, tmp_path, stop_after_errors=2)
    assert counts == {"error": 2}
    assert len(await results(sessionmaker)) == 2


async def test_summary_accuracy_excludes_errors(sessionmaker: SM, tmp_path: Path) -> None:
    llm = ScriptedLLM([action("final_answer", "42"), action("final_answer", "7")])
    await run(sessionmaker, runtime_for(llm), [math_task(1), math_task(2)], tmp_path)
    [row] = await summarize(sessionmaker, "t", ["train"])
    assert (row.n, row.correct, row.errors, row.accuracy) == (2, 1, 0, 0.5)
    assert "50.0%" in format_summary([row])
    assert await summarize(sessionmaker, "t", ["val"]) == []


def test_test_split_needs_explicit_final_flag() -> None:
    check_splits(["train", "val"], final_test=False)
    with pytest.raises(SplitGuardError, match="final benchmark"):
        check_splits(["train", "test"], final_test=False)
    check_splits(["test"], final_test=True)
