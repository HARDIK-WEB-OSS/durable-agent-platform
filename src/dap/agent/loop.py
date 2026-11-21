"""ReAct agent as durable steps: one LLM call or one tool call per step.

    agent.think (turn n) --tool--> agent.act (turn n) --> agent.think (turn n+1) ...
                         --final_answer--> run succeeded with {"answer": ...}

Every LLM call and tool run goes through StepContext.external: at-least-once, recorded once.
After a crash the retried step reuses a recorded result instead of paying for the call again;
only a crash in the window between the call returning and the record committing repeats it.

Limits are checked before every LLM turn: max turns, wall-clock timeout since the run was
created, and the run's budget_usd against the paid cost of its llm.call events so far. A turn
already in flight can overshoot the budget by one call. Hitting a limit fails the run with
the reason as its error.
"""

import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Protocol

from sqlalchemy import Numeric, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.agent.memory import ACT_KIND, THINK_KIND, build_messages, load_history
from dap.agent.prompts import FINAL_ANSWER, AgentAction
from dap.agent.tools import TOOL_NAMES, Tool, ToolBox, ToolError, UnknownToolError
from dap.agent.tools.base import truncate
from dap.agent.tools.python_exec import DockerSandbox
from dap.config import Settings
from dap.db import repo
from dap.db.models import Event, JsonDict, Run
from dap.engine.runner import Handler, PermanentError, StepContext, StepResult, StepSpec
from dap.llm.client import (
    ContextTooLongError,
    LLMCallError,
    LLMClient,
    LLMResult,
    Message,
    ModelConfig,
    StructuredOutputError,
    configured_arms,
    record_llm_call,
    record_llm_failure,
)
from dap.llm.embeddings import OllamaEmbedder

STEP_MAX_ATTEMPTS = 3

# Writes one event in its own short transaction: (session, *, run_id, step_id).
EventWriter = Callable[..., Awaitable[object]]


class ToolProvider(Protocol):
    def for_run(self, run_input: JsonDict) -> dict[str, Tool]: ...


@dataclass(frozen=True)
class AgentLimits:
    max_turns: int = 12
    timeout_s: float = 900.0
    max_tokens: int = 512
    max_observation_chars: int = 2000


@dataclass(frozen=True)
class RunInfo:
    task: str
    input: JsonDict
    budget_usd: Decimal | None
    age_s: float
    spent_usd: Decimal


async def load_run_info(session: AsyncSession, run_id: uuid.UUID) -> RunInfo:
    age = func.extract("epoch", func.now() - Run.created_at)
    row = (
        await session.execute(
            select(Run.task, Run.input, Run.budget_usd, age).where(Run.id == run_id)
        )
    ).one()
    spent = await session.scalar(
        select(func.coalesce(func.sum(cast(Event.payload["cost_usd"].astext, Numeric)), 0)).where(
            Event.run_id == run_id, Event.type == "llm.call"
        )
    )
    return RunInfo(row[0], row[1], row[2], float(row[3]), Decimal(spent or 0))


def usage(result: LLMResult) -> JsonDict:
    return result.model_dump(exclude={"content"})


async def submit_agent_run(
    session: AsyncSession,
    task: str,
    arm: str,
    *,
    tools: list[str] | None = None,
    sql_db: str | None = None,
    budget_usd: Decimal | None = None,
    max_turns: int | None = None,
    tags: JsonDict | None = None,
) -> Run:
    if tools is not None and (unknown := set(tools) - set(TOOL_NAMES)):
        raise UnknownToolError(f"unknown tools {sorted(unknown)}")
    run_input: JsonDict = {
        **(tags or {}),
        "arm": arm,
        "tools": tools,
        "sql_db": sql_db,
        "max_turns": max_turns,
    }
    run = await repo.create_run(session, task, run_input, budget_usd)
    await repo.add_step(session, run.id, THINK_KIND, {"arm": arm, "turn": 1}, STEP_MAX_ATTEMPTS)
    return run


@dataclass(frozen=True)
class AgentRuntime:
    llm: LLMClient
    arms: Mapping[str, ModelConfig]
    toolbox: ToolProvider
    limits: AgentLimits = AgentLimits()

    def handlers(self) -> dict[str, Handler]:
        return {THINK_KIND: self.think, ACT_KIND: self.act}

    def stop_reason(self, info: RunInfo, turn: int) -> str | None:
        max_turns = info.input.get("max_turns") or self.limits.max_turns
        if turn > max_turns:
            return f"stopped: reached max turns ({max_turns}) without a final answer"
        if info.age_s > self.limits.timeout_s:
            return f"stopped: run exceeded its {self.limits.timeout_s:g} s time limit"
        if info.budget_usd is not None and info.spent_usd >= info.budget_usd:
            return f"stopped: spent ${info.spent_usd} of ${info.budget_usd} budget"
        return None

    def tools_for(self, info: RunInfo) -> dict[str, Tool]:
        try:
            return self.toolbox.for_run(info.input)
        except (UnknownToolError, ValueError, FileNotFoundError) as exc:
            raise PermanentError(str(exc)) from exc

    async def think(self, ctx: StepContext) -> StepResult:
        claim = ctx.claim
        turn: int = claim.input["turn"]
        arm_name: str = claim.input["arm"]
        async with ctx.sessionmaker() as session:
            info = await load_run_info(session, claim.run_id)
            history = await load_history(session, claim.run_id, claim.seq)
        if reason := self.stop_reason(info, turn):
            return StepResult({"stopped": reason}, run_error=reason)
        model = self.arms.get(arm_name)
        if model is None:
            raise PermanentError(f"arm {arm_name!r} is not configured on this worker")
        messages = build_messages(info.task, self.tools_for(info), history)
        try:
            out = await ctx.external("llm", partial(self.call_llm, ctx, model, messages))
        except ContextTooLongError as exc:
            reason = f"stopped: transcript no longer fits the context window ({exc})"
            return StepResult({"stopped": reason}, run_error=reason)
        if "format_error" in out:
            return StepResult(out, next_step=self.think_spec(arm_name, turn + 1))
        action = out["action"]
        if action["tool"] == FINAL_ANSWER:
            answer = {"answer": action["tool_input"], "turns": turn, "arm": arm_name}
            return StepResult(out, run_result=answer)
        act_input = {"arm": arm_name, "turn": turn, **action}
        return StepResult(out, next_step=StepSpec(ACT_KIND, act_input, STEP_MAX_ATTEMPTS))

    async def call_llm(
        self, ctx: StepContext, model: ModelConfig, messages: list[Message]
    ) -> JsonDict:
        try:
            parsed, result = await self.llm.complete_structured(
                model, messages, AgentAction, max_tokens=self.limits.max_tokens
            )
        except StructuredOutputError as exc:
            await self.log(ctx, partial(record_llm_call, result=exc.result, ok=False))
            raw = exc.result.content[:1000]
            return {"format_error": str(exc), "raw": raw, "usage": usage(exc.result)}
        except LLMCallError as exc:
            await self.log(ctx, partial(record_llm_failure, error=exc))
            raise
        await self.log(ctx, partial(record_llm_call, result=result))
        action = parsed.model_dump()
        action["tool"] = action["tool"].strip()
        return {"action": action, "usage": usage(result)}

    async def act(self, ctx: StepContext) -> StepResult:
        claim = ctx.claim
        async with ctx.sessionmaker() as session:
            info = await load_run_info(session, claim.run_id)
        tools = self.tools_for(info)
        call = partial(self.call_tool, ctx, tools, claim.input["tool"], claim.input["tool_input"])
        out = await ctx.external("tool", call)
        next_turn = self.think_spec(claim.input["arm"], claim.input["turn"] + 1)
        return StepResult(out, next_step=next_turn)

    async def call_tool(
        self, ctx: StepContext, tools: Mapping[str, Tool], name: str, tool_input: str
    ) -> JsonDict:
        tool = tools.get(name)
        started = time.perf_counter()
        ok = False
        if tool is None:
            observation = f"ERROR: unknown tool {name!r}. Available: {sorted(tools)}"
        else:
            try:
                observation = await tool.run(tool_input)
                ok = True
            except ToolError as exc:
                observation = f"ERROR: {exc}"
        latency = time.perf_counter() - started
        observation = truncate(observation, self.limits.max_observation_chars)
        payload = {"tool": name, "ok": ok, "latency_s": latency, "chars": len(observation)}

        async def log_tool(session: AsyncSession, run_id: uuid.UUID, step_id: uuid.UUID) -> None:
            await repo.append_event(session, run_id, "tool.call", payload, step_id=step_id)

        await self.log(ctx, log_tool)
        return {"tool": name, "observation": observation, "ok": ok}

    async def log(self, ctx: StepContext, write: EventWriter) -> None:
        """Every real call is logged, even one a crash makes us repeat: cost stays honest."""
        async with ctx.sessionmaker.begin() as session:
            await write(session, run_id=ctx.claim.run_id, step_id=ctx.claim.step_id)

    @staticmethod
    def think_spec(arm: str, turn: int) -> StepSpec:
        return StepSpec(THINK_KIND, {"arm": arm, "turn": turn}, STEP_MAX_ATTEMPTS)


async def build_runtime(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]
) -> AgentRuntime:
    """Wire the agent from settings. Fails fast if python_exec is on but Docker is not usable."""
    sandbox = None
    if settings.python_exec_enabled:
        sandbox = DockerSandbox(settings.sandbox_image, timeout_s=settings.sandbox_timeout_seconds)
        await sandbox.check_available()
    toolbox = ToolBox(
        sessionmaker=sessionmaker,
        embedder=OllamaEmbedder(settings.embedding_model, settings.ollama_api_base),
        sandbox=sandbox,
        sql_data_dir=settings.sql_data_dir,
        sql_timeout_s=settings.sql_timeout_seconds,
        search_top_k=settings.search_top_k,
    )
    limits = AgentLimits(
        max_turns=settings.agent_max_steps,
        timeout_s=settings.agent_timeout_seconds,
        max_tokens=settings.agent_max_tokens,
        max_observation_chars=settings.agent_max_observation_chars,
    )
    return AgentRuntime(
        LLMClient.from_settings(settings), configured_arms(settings), toolbox, limits
    )
