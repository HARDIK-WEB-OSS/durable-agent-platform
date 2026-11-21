import asyncio
import hashlib
import json
import math
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from dap.agent.loop import AgentLimits, AgentRuntime, submit_agent_run
from dap.agent.memory import ACT_KIND, THINK_KIND
from dap.agent.tools import ToolBox
from dap.agent.tools.base import Tool, ToolError
from dap.agent.tools.calculator import CalculatorTool
from dap.agent.tools.search_docs import Chunk, SearchDocsTool, split_paragraphs, upsert_chunks
from dap.db import repo
from dap.db.models import Event, IdempotencyKey, JsonDict, Run, Step
from dap.engine.effects import record_once
from dap.engine.queue import Claim, claim_next_step
from dap.engine.runner import EngineConfig, StepContext
from dap.engine.worker import Worker
from dap.llm.client import LLMClient, ModelConfig
from dap.llm.embeddings import EMBEDDING_DIM
from dap.states import RunStatus, StepStatus

pytestmark = pytest.mark.integration

SM = async_sessionmaker[AsyncSession]
CONFIG = EngineConfig(2.0, 0.05, 0.01, 0.05)
ANCHOR = "deepinfra/Qwen/Qwen2.5-7B-Instruct"
MID = ModelConfig(
    arm="mid", model="ollama_chat/qwen2.5:3b", api_base="http://ollama.test", num_ctx=4096
)
TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
HANG = object()


def action(tool: str, tool_input: str, thought: str = "next") -> str:
    return json.dumps({"thought": thought, "tool": tool, "tool_input": tool_input})


class ScriptedLLM:
    """Fake Ollama /api/chat that replays scripted replies and records every request."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[JsonDict] = []
        self.seen = asyncio.Event()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.seen.set()
        reply = self.replies.pop(0)
        if reply is HANG:
            await asyncio.sleep(60)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": reply},
                "done_reason": "stop",
                "prompt_eval_count": 50,
                "eval_count": 10,
            },
        )

    def client(self) -> LLMClient:
        return LLMClient(
            timeout_s=30,
            max_retries=0,
            shadow_anchor=ANCHOR,
            retry_base_s=0,
            retry_cap_s=0,
            transport=httpx.MockTransport(self.handle),
        )

    def last_messages(self) -> list[JsonDict]:
        messages: list[JsonDict] = self.requests[-1]["messages"]
        return messages


@dataclass
class CountingTool:
    name: str = "counter"
    description: str = "Counts calls. tool_input: anything."
    calls: int = 0

    async def run(self, tool_input: str) -> str:
        self.calls += 1
        if tool_input == "fail":
            raise ToolError("asked to fail")
        return f"call {self.calls}"


@dataclass
class FixedTools:
    tools: dict[str, Tool] = field(default_factory=dict)

    def for_run(self, run_input: JsonDict) -> dict[str, Tool]:
        return self.tools


def runtime(llm: ScriptedLLM, tools: dict[str, Tool] | None = None, **limits: Any) -> AgentRuntime:
    provider = FixedTools(tools if tools is not None else {"calculator": CalculatorTool()})
    return AgentRuntime(llm.client(), {"mid": MID}, provider, AgentLimits(**limits))


async def submit(sm: SM, task: str = "What is 6 * 7?", **kwargs: Any) -> uuid.UUID:
    async with sm.begin() as s:
        run = await submit_agent_run(s, task, "mid", **kwargs)
    return run.id


async def drive(sm: SM, rt: AgentRuntime, run_id: uuid.UUID, workers: int = 1) -> Run:
    pool = [
        Worker(f"w{i}", sm, rt.handlers(), CONFIG, rng=random.Random(i)) for i in range(workers)
    ]
    tasks = [asyncio.create_task(w.run()) for w in pool]
    try:
        async with asyncio.timeout(30):
            while (run := await load_run(sm, run_id)).status not in TERMINAL:
                await asyncio.sleep(0.05)
    finally:
        for w in pool:
            w.stop.set()
        await asyncio.gather(*tasks)
    return run


async def load_run(sm: SM, run_id: uuid.UUID) -> Run:
    async with sm() as s:
        run = await repo.get_run(s, run_id)
    assert run is not None
    return run


async def steps_of(sm: SM, run_id: uuid.UUID) -> list[Step]:
    async with sm() as s:
        return list(await repo.list_steps(s, run_id))


async def event_types(sm: SM, run_id: uuid.UUID) -> list[str]:
    async with sm() as s:
        return [e.type for e in await repo.list_events(s, run_id)]


async def claim_as(sm: SM, worker_id: str) -> Claim:
    async with sm.begin() as s:
        c = await claim_next_step(s, worker_id, 30)
    assert c is not None
    return c


async def expire_leases(sm: SM) -> None:
    async with sm.begin() as s:
        await s.execute(
            update(Step)
            .where(Step.status == StepStatus.RUNNING)
            .values(lease_expires_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 0, 1))
        )


# ---- the loop ------------------------------------------------------------------------------


async def test_agent_calls_a_tool_then_answers(sessionmaker: SM) -> None:
    llm = ScriptedLLM([action("calculator", "6 * 7"), action("final_answer", "42")])
    run_id = await submit(sessionmaker)
    run = await drive(sessionmaker, runtime(llm), run_id)
    assert (run.status, run.result) == (
        RunStatus.SUCCEEDED,
        {"answer": "42", "turns": 2, "arm": "mid"},
    )
    steps = await steps_of(sessionmaker, run_id)
    assert [s.kind for s in steps] == [THINK_KIND, ACT_KIND, THINK_KIND]
    assert steps[1].output == {"tool": "calculator", "observation": "42", "ok": True}
    observation = llm.last_messages()[-1]["content"]
    assert "<observation>\n42\n</observation>" in observation
    types = await event_types(sessionmaker, run_id)
    assert (types.count("llm.call"), types.count("tool.call")) == (2, 1)


async def test_invalid_reply_is_fed_back_and_the_agent_recovers(sessionmaker: SM) -> None:
    llm = ScriptedLLM(["no json here", "still none", action("final_answer", "7")])
    run_id = await submit(sessionmaker)
    run = await drive(sessionmaker, runtime(llm), run_id)
    assert run.status == RunStatus.SUCCEEDED and run.result == {
        "answer": "7",
        "turns": 2,
        "arm": "mid",
    }
    assert "was not valid" in llm.last_messages()[-1]["content"]
    async with sessionmaker() as s:
        oks = (
            await s.scalars(
                select(Event.payload["ok"].as_boolean()).where(
                    Event.run_id == run_id, Event.type == "llm.call"
                )
            )
        ).all()
    assert oks == [False, True]


async def test_unknown_tool_and_tool_errors_become_observations(sessionmaker: SM) -> None:
    counter = CountingTool()
    llm = ScriptedLLM(
        [action("rm_rf", "/"), action("counter", "fail"), action("final_answer", "done")]
    )
    run_id = await submit(sessionmaker)
    run = await drive(sessionmaker, runtime(llm, {"counter": counter}), run_id)
    assert run.status == RunStatus.SUCCEEDED
    acts = [s.output for s in await steps_of(sessionmaker, run_id) if s.kind == ACT_KIND]
    assert acts[0] is not None and "unknown tool 'rm_rf'" in acts[0]["observation"]
    assert acts[1] == {"tool": "counter", "observation": "ERROR: asked to fail", "ok": False}


async def test_max_turns_fails_the_run_without_another_llm_call(sessionmaker: SM) -> None:
    llm = ScriptedLLM([action("calculator", "1+1"), action("calculator", "2+2")])
    run_id = await submit(sessionmaker)
    run = await drive(sessionmaker, runtime(llm, max_turns=2), run_id)
    assert run.status == RunStatus.FAILED
    assert run.error is not None and "max turns (2)" in run.error
    assert len(llm.requests) == 2


async def test_per_run_max_turns_overrides_the_default(sessionmaker: SM) -> None:
    llm = ScriptedLLM([action("calculator", "1+1")])
    run_id = await submit(sessionmaker, max_turns=1)
    run = await drive(sessionmaker, runtime(llm, max_turns=10), run_id)
    assert run.error is not None and "max turns (1)" in run.error


async def test_exhausted_budget_stops_before_calling(sessionmaker: SM) -> None:
    llm = ScriptedLLM([])
    run_id = await submit(sessionmaker, budget_usd=Decimal("0"))
    run = await drive(sessionmaker, runtime(llm), run_id)
    assert run.status == RunStatus.FAILED and run.error is not None and "budget" in run.error
    assert llm.requests == []


async def test_budget_counts_paid_cost_of_logged_calls(sessionmaker: SM) -> None:
    run_id = await submit(sessionmaker, budget_usd=Decimal("0.01"))
    async with sessionmaker.begin() as s:
        payload = {"cost_usd": 0.004, "ok": True}
        for _ in range(3):
            await repo.append_event(s, run_id, "llm.call", payload)
    llm = ScriptedLLM([])
    run = await drive(sessionmaker, runtime(llm), run_id)
    assert run.error is not None and "spent $0.012 of $0.010000 budget" in run.error


async def test_run_timeout_stops_the_agent(sessionmaker: SM) -> None:
    llm = ScriptedLLM([])
    run_id = await submit(sessionmaker)
    run = await drive(sessionmaker, runtime(llm, timeout_s=0.001), run_id)
    assert run.status == RunStatus.FAILED and run.error is not None and "time limit" in run.error


async def test_transcript_too_long_for_context_fails_cleanly(sessionmaker: SM) -> None:
    llm = ScriptedLLM([])
    run_id = await submit(sessionmaker, task="x" * 20_000)
    run = await drive(sessionmaker, runtime(llm), run_id)
    assert run.status == RunStatus.FAILED
    assert run.error is not None and "context window" in run.error
    assert llm.requests == []


async def test_unconfigured_arm_dead_letters_the_step(sessionmaker: SM) -> None:
    async with sessionmaker.begin() as s:
        run = await submit_agent_run(s, "task", "strong")
    run_out = await drive(sessionmaker, runtime(ScriptedLLM([])), run.id)
    assert run_out.status == RunStatus.FAILED
    assert run_out.error is not None and "not configured" in run_out.error


# ---- durability: crashes in the middle of the agent ----------------------------------------


async def test_crash_after_llm_call_was_recorded_does_not_pay_again(sessionmaker: SM) -> None:
    llm = ScriptedLLM([action("final_answer", "42")])  # a second call would pop an empty list
    rt = runtime(llm)
    run_id = await submit(sessionmaker)
    crashed = await claim_as(sessionmaker, "crashed")
    await rt.think(StepContext(crashed, sessionmaker))  # result never committed: "crash"
    await expire_leases(sessionmaker)
    run = await drive(sessionmaker, rt, run_id)
    assert run.status == RunStatus.SUCCEEDED and run.result is not None
    assert run.result["answer"] == "42"
    assert len(llm.requests) == 1
    [think] = await steps_of(sessionmaker, run_id)
    assert think.attempts == 2
    assert (await event_types(sessionmaker, run_id)).count("llm.call") == 1


async def test_crash_during_llm_call_repeats_it_at_least_once(sessionmaker: SM) -> None:
    llm = ScriptedLLM([HANG, action("final_answer", "42")])
    rt = runtime(llm)
    run_id = await submit(sessionmaker)
    crashed = await claim_as(sessionmaker, "crashed")
    in_flight = asyncio.create_task(rt.think(StepContext(crashed, sessionmaker)))
    await asyncio.wait_for(llm.seen.wait(), 5)
    in_flight.cancel()  # worker dies while the model is still answering
    with pytest.raises(asyncio.CancelledError):
        await in_flight
    await expire_leases(sessionmaker)
    run = await drive(sessionmaker, rt, run_id)
    assert run.status == RunStatus.SUCCEEDED
    assert len(llm.requests) == 2  # at-least-once: the unrecorded call ran again
    async with sessionmaker() as s:
        keys = (await s.scalars(select(IdempotencyKey.key))).all()
    assert len(keys) == 1


async def test_crash_after_tool_run_does_not_rerun_the_tool(sessionmaker: SM) -> None:
    counter = CountingTool()
    llm = ScriptedLLM([action("counter", "go"), action("final_answer", "ok")])
    rt = runtime(llm, {"counter": counter})
    run_id = await submit(sessionmaker)
    await rt_step(sessionmaker, rt)  # think turn 1 commits normally
    crashed = await claim_as(sessionmaker, "crashed")
    assert crashed.kind == ACT_KIND
    await rt.act(StepContext(crashed, sessionmaker))  # tool ran and was recorded, then "crash"
    await expire_leases(sessionmaker)
    run = await drive(sessionmaker, rt, run_id)
    assert run.status == RunStatus.SUCCEEDED
    assert counter.calls == 1
    act = next(s for s in await steps_of(sessionmaker, run_id) if s.kind == ACT_KIND)
    assert act.attempts == 2 and act.output is not None
    assert act.output["observation"] == "call 1"


async def rt_step(sm: SM, rt: AgentRuntime) -> None:
    worker = Worker("one-step", sm, rt.handlers(), CONFIG)
    assert await worker.tick()


async def test_transcript_survives_worker_change(sessionmaker: SM) -> None:
    llm = ScriptedLLM(
        [
            action("calculator", "2 + 2"),
            action("calculator", "4 * 10"),
            action("final_answer", "40"),
        ]
    )
    rt = runtime(llm)
    run_id = await submit(sessionmaker)
    for _ in range(4):  # each step on a fresh worker object, like separate processes
        await rt_step(sessionmaker, rt)
    run = await drive(sessionmaker, rt, run_id)
    assert run.result == {"answer": "40", "turns": 3, "arm": "mid"}
    contents = [m["content"] for m in llm.last_messages()]
    assert sum("<observation>" in c for c in contents) == 2
    assert any("<observation>\n40\n</observation>" in c for c in contents)


# ---- external-effect primitive (F02) -------------------------------------------------------


async def test_record_once_holds_no_connection_while_the_effect_runs(
    sessionmaker: SM, engine: AsyncEngine
) -> None:
    run_id = await submit(sessionmaker)
    claim = await claim_as(sessionmaker, "w")
    checked_out_during: list[int] = []

    async def effect() -> JsonDict:
        checked_out_during.append(engine.pool.checkedout())  # type: ignore[attr-defined]
        return {"value": 1}

    assert await record_once(sessionmaker, "k", claim.step_id, effect) == {"value": 1}
    assert checked_out_during == [0]
    assert run_id


async def test_record_once_concurrent_duplicates_agree_on_first_result(sessionmaker: SM) -> None:
    await submit(sessionmaker)
    claim = await claim_as(sessionmaker, "w")
    calls = 0

    async def effect() -> JsonDict:
        nonlocal calls
        calls += 1
        mine = calls
        await asyncio.sleep(0.2 if mine == 1 else 0.0)
        return {"call": mine}

    results = await asyncio.gather(
        record_once(sessionmaker, "dup", claim.step_id, effect),
        record_once(sessionmaker, "dup", claim.step_id, effect),
    )
    assert calls == 2  # both ran: at-least-once
    assert results[0] == results[1]  # but one result was recorded and both see it
    assert await record_once(sessionmaker, "dup", claim.step_id, effect) == results[0]
    assert calls == 2


# ---- search_docs on pgvector ---------------------------------------------------------------


class HashEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words point the same way."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.vector(text)

    @staticmethod
    def vector(text: str) -> list[float]:
        v = [0.0] * EMBEDDING_DIM
        for word in text.lower().split():
            v[int(hashlib.sha256(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


async def test_search_docs_returns_the_closest_passage(sessionmaker: SM) -> None:
    embedder = HashEmbedder()
    chunks = [
        Chunk("pets.md", 0, "Pets", "cats sleep most of the day"),
        Chunk("space.md", 0, "Space", "rockets need fuel to reach orbit"),
        Chunk("food.md", 0, None, "bread needs flour water and yeast"),
    ]
    async with sessionmaker.begin() as s:
        assert await upsert_chunks(s, chunks, embedder) == 3
        await upsert_chunks(s, chunks[:1], embedder)  # re-ingest is an upsert, not a duplicate
    out = await SearchDocsTool(sessionmaker, embedder, top_k=2).run("how do rockets reach orbit")
    first = out.split("</document>")[0]
    assert 'source="space.md#0 (Space)"' in first and "rockets need fuel" in first
    assert out.count("<document ") == 2


async def test_search_docs_with_no_documents(sessionmaker: SM) -> None:
    out = await SearchDocsTool(sessionmaker, HashEmbedder()).run("anything")
    assert out == "No documents found."


async def test_toolbox_builds_search_docs_when_embedder_is_present(sessionmaker: SM) -> None:
    box = ToolBox(sessionmaker=sessionmaker, embedder=HashEmbedder())
    assert set(box.for_run({})) == {"calculator", "search_docs"}


def test_split_paragraphs_packs_and_cuts() -> None:
    body = "a" * 50 + "\n\n" + "b" * 50 + "\n\n" + "c" * 300
    assert split_paragraphs(body, max_chars=120) == [
        "a" * 50 + "\n\n" + "b" * 50,
        "c" * 120,
        "c" * 120,
        "c" * 60,
    ]
