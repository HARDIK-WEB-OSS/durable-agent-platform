import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dap.db import repo
from dap.llm.client import LLMCallError, LLMResult, ModelConfig, record_llm_call, record_llm_failure

pytestmark = pytest.mark.integration


async def test_llm_call_is_stored_as_event(session: AsyncSession) -> None:
    run = await repo.create_run(session, "llm demo")
    result = LLMResult(
        arm="cheap",
        model="ollama_chat/qwen2.5:3b",
        content="42",
        tokens_in=12,
        tokens_out=3,
        cost_usd=0.0,
        shadow_cost_usd=0.000001,
        price_basis="modeled",
        latency_s=0.8,
        wall_s=0.9,
        calls=1,
    )
    await record_llm_call(session, run.id, result)
    events = await repo.list_events(session, run.id)
    assert events[-1].type == "llm.call"
    payload = events[-1].payload
    assert payload["model"] == "ollama_chat/qwen2.5:3b"
    assert (payload["tokens_in"], payload["tokens_out"], payload["cost_usd"]) == (12, 3, 0.0)
    assert (payload["latency_s"], payload["wall_s"], payload["ok"]) == (0.8, 0.9, True)
    assert payload["truncated"] is False
    assert payload["price_basis"] == "modeled"


async def test_failed_llm_call_is_stored_with_its_wall_time(session: AsyncSession) -> None:
    run = await repo.create_run(session, "llm demo")
    model = ModelConfig(arm="mid", model="ollama_chat/qwen2.5:3b")
    error = LLMCallError(model, calls=3, wall_s=181.5, cause=TimeoutError())
    await record_llm_failure(session, run.id, error)
    payload = (await repo.list_events(session, run.id))[-1].payload
    assert payload == {
        "arm": "mid",
        "model": "ollama_chat/qwen2.5:3b",
        "calls": 3,
        "wall_s": 181.5,
        "error_type": "TimeoutError",
        "ok": False,
    }
