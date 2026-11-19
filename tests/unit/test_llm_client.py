import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest
from litellm.exceptions import AuthenticationError, RateLimitError, Timeout
from pydantic import BaseModel, SecretStr

from dap.llm import client as client_module
from dap.llm.client import (
    ContextTooLongError,
    LLMCallError,
    LLMClient,
    LLMResult,
    ModelConfig,
    StructuredOutputError,
    estimate_prompt_tokens,
)
from dap.llm.pricing import UnknownPriceError

LOCAL = ModelConfig(
    arm="cheap", model="ollama_chat/qwen2.5:3b", api_base="http://ollama.test", num_ctx=4096
)
HOSTED = ModelConfig(arm="strong", model="gpt-4o-mini", api_key=SecretStr("sk-test-secret"))
ANCHOR = "deepinfra/Qwen/Qwen2.5-7B-Instruct"
ASK = [{"role": "user", "content": "What is 6 * 7?"}]

Outcome = str | int | Exception


class Answer(BaseModel):
    answer: int


def make_client(
    transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 5.0, max_retries: int = 2
) -> LLMClient:
    return LLMClient(
        timeout_s=timeout_s,
        max_retries=max_retries,
        shadow_anchor=ANCHOR,
        retry_base_s=0.0,
        retry_cap_s=0.0,
        transport=transport,
    )


class FakeOllama:
    """Scripted /api/chat: str = reply, int = HTTP error status, exception = raised."""

    def __init__(
        self, outcomes: list[Outcome], tokens: tuple[int, int] = (12, 7), done_reason: str = "stop"
    ) -> None:
        self.outcomes = outcomes
        self.tokens = tokens
        self.done_reason = done_reason
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"error": "scripted failure"})
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": outcome},
                "done": True,
                "done_reason": self.done_reason,
                "prompt_eval_count": self.tokens[0],
                "eval_count": self.tokens[1],
            },
        )

    def payload(self, index: int = 0) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(self.requests[index].content)
        return body

    def client(self, **kwargs: Any) -> LLMClient:
        return make_client(httpx.MockTransport(self.handle), **kwargs)


def reply(content: str, tokens_in: int = 12, tokens_out: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


class FakeLiteLLM:
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return reply(outcome)


FakeFactory = Callable[[list[str | Exception]], FakeLiteLLM]


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> FakeFactory:
    def install(outcomes: list[str | Exception]) -> FakeLiteLLM:
        fake = FakeLiteLLM(outcomes)
        monkeypatch.setattr(client_module, "acompletion", fake)
        return fake

    return install


def rate_limited() -> Exception:
    return RateLimitError("slow down", llm_provider="openai", model="gpt-4o-mini")


def timed_out() -> Exception:
    return Timeout("too slow", model="gpt-4o-mini", llm_provider="openai")


async def test_local_call_is_free_but_has_shadow_cost() -> None:
    result = await FakeOllama(["42"]).client().complete(LOCAL, ASK)
    assert (result.content, result.tokens_in, result.tokens_out) == ("42", 12, 7)
    assert (result.cost_usd, result.calls, result.arm, result.truncated) == (0.0, 1, "cheap", False)
    anchor_in, anchor_out = litellm.cost_per_token(
        model=ANCHOR, prompt_tokens=12, completion_tokens=7
    )
    assert result.shadow_cost_usd == pytest.approx((anchor_in + anchor_out) * 3 / 7)
    assert 0.0 <= result.latency_s <= result.wall_s


async def test_local_call_goes_straight_to_ollama_with_options() -> None:
    ollama = FakeOllama(["ok"])
    await ollama.client().complete(LOCAL, ASK, max_tokens=64)
    [request] = ollama.requests
    assert str(request.url) == "http://ollama.test/api/chat"
    body = ollama.payload()
    assert (body["model"], body["stream"], body["messages"]) == ("qwen2.5:3b", False, ASK)
    assert body["options"] == {"temperature": 0.0, "num_predict": 64, "num_ctx": 4096}
    assert "format" not in body


async def test_ollama_connection_errors_and_5xx_are_retried() -> None:
    ollama = FakeOllama([httpx.ConnectError("refused"), 503, "42"])
    result = await ollama.client().complete(LOCAL, ASK)
    assert (result.content, result.calls, len(ollama.requests)) == ("42", 3, 3)


async def test_missing_local_model_fails_without_retry() -> None:
    ollama = FakeOllama([404, "never"])
    with pytest.raises(LLMCallError, match="after 1 call:") as info:
        await ollama.client().complete(LOCAL, ASK)
    assert (info.value.calls, info.value.error_type, len(ollama.requests)) == (1, "LLMError", 1)


async def test_output_cut_at_max_tokens_is_flagged_truncated() -> None:
    result = await FakeOllama(["4"], done_reason="length").client().complete(LOCAL, ASK)
    assert result.truncated


async def test_prompt_that_filled_the_context_is_flagged_truncated() -> None:
    ollama = FakeOllama(["42"], tokens=(3700, 7))
    result = await ollama.client().complete(LOCAL, ASK, max_tokens=512)
    assert result.truncated


async def test_latency_excludes_retries_but_wall_time_includes_them() -> None:
    calls = 0

    async def slow_then_ok(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.3)
            return httpx.Response(503)
        return httpx.Response(200, json={"message": {"content": "42"}, "done_reason": "stop"})

    result = await make_client(httpx.MockTransport(slow_then_ok)).complete(LOCAL, ASK)
    assert result.calls == 2
    assert result.wall_s >= 0.3 > result.latency_s


async def test_hung_call_is_cut_by_overall_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def hang(**kwargs: Any) -> Any:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    monkeypatch.setattr(client_module, "acompletion", hang)
    client = make_client(timeout_s=0.2, max_retries=0)
    async with asyncio.timeout(2):
        with pytest.raises(LLMCallError) as info:
            await client.complete(HOSTED, ASK)
    assert info.value.error_type == "TimeoutError"
    assert 0.2 <= info.value.wall_s < 1.0


async def test_hosted_call_is_priced_from_litellm_table(fake_litellm: FakeFactory) -> None:
    fake_litellm(["42"])
    result = await make_client().complete(HOSTED, ASK)
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="gpt-4o-mini", prompt_tokens=12, completion_tokens=7
    )
    assert result.cost_usd == pytest.approx(prompt_cost + completion_cost)
    assert result.shadow_cost_usd == result.cost_usd > 0


async def test_hosted_request_disables_sdk_retries(fake_litellm: FakeFactory) -> None:
    fake = fake_litellm(["ok"])
    await make_client().complete(HOSTED, ASK, max_tokens=64)
    [sent] = fake.requests
    assert (sent["num_retries"], sent["max_retries"]) == (0, 0)
    assert (sent["timeout"], sent["temperature"], sent["max_tokens"]) == (5.0, 0.0, 64)
    assert "num_ctx" not in sent


async def test_api_key_is_sent_but_never_exposed(fake_litellm: FakeFactory) -> None:
    fake = fake_litellm(["ok"])
    result = await make_client().complete(HOSTED, ASK)
    assert fake.requests[0]["api_key"] == "sk-test-secret"
    assert "sk-test-secret" not in repr(HOSTED)
    assert "sk-test-secret" not in result.model_dump_json()


async def test_transient_errors_are_retried(fake_litellm: FakeFactory) -> None:
    fake = fake_litellm([rate_limited(), timed_out(), "42"])
    result = await make_client().complete(HOSTED, ASK)
    assert (result.content, result.calls, len(fake.requests)) == ("42", 3, 3)


async def test_gives_up_after_max_retries(fake_litellm: FakeFactory) -> None:
    fake = fake_litellm([rate_limited(), rate_limited(), rate_limited(), "never"])
    with pytest.raises(LLMCallError, match="after 3 calls") as info:
        await make_client().complete(HOSTED, ASK)
    assert (info.value.calls, info.value.error_type) == (3, "RateLimitError")
    assert len(fake.requests) == 3


async def test_auth_error_is_not_retried(fake_litellm: FakeFactory) -> None:
    error = AuthenticationError("bad key", llm_provider="openai", model="gpt-4o-mini")
    fake = fake_litellm([error, "never"])
    with pytest.raises(LLMCallError) as info:
        await make_client().complete(HOSTED, ASK)
    assert isinstance(info.value.__cause__, AuthenticationError)
    assert len(fake.requests) == 1


async def test_prompt_larger_than_context_is_rejected_before_calling() -> None:
    ollama = FakeOllama(["never"])
    huge = [{"role": "user", "content": "token " * 5000}]
    with pytest.raises(ContextTooLongError):
        await ollama.client().complete(LOCAL, huge)
    assert ollama.requests == []


def test_prompt_estimate_overcounts_prose() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 50
    assert estimate_prompt_tokens([{"role": "user", "content": text}]) >= len(text) // 4


def test_prompt_estimate_counts_each_digit_as_a_token() -> None:
    digits = "3141592653 " * 100
    assert estimate_prompt_tokens([{"role": "user", "content": digits}]) >= 1000


async def test_structured_output_sends_schema_as_ollama_format() -> None:
    ollama = FakeOllama(['{"answer": 42}'])
    parsed, result = await ollama.client().complete_structured(LOCAL, ASK, Answer)
    assert parsed == Answer(answer=42)
    assert ollama.payload()["format"]["required"] == ["answer"]
    assert result.calls == 1


async def test_structured_output_tolerates_code_fence() -> None:
    ollama = FakeOllama(['```json\n{"answer": 7}\n```'])
    parsed, _ = await ollama.client().complete_structured(LOCAL, ASK, Answer)
    assert parsed.answer == 7


async def test_structured_output_repairs_and_counts_all_tokens() -> None:
    ollama = FakeOllama(["forty-two", '{"answer": 42}'])
    parsed, result = await ollama.client().complete_structured(LOCAL, ASK, Answer)
    assert parsed.answer == 42
    assert (result.calls, result.tokens_in, result.tokens_out) == (2, 24, 14)
    repair_prompt = ollama.payload(1)["messages"][-1]["content"]
    assert "did not match the JSON schema" in repair_prompt


async def test_structured_output_failure_keeps_usage() -> None:
    ollama = FakeOllama(["nope", "still nope"])
    with pytest.raises(StructuredOutputError) as info:
        await ollama.client().complete_structured(LOCAL, ASK, Answer)
    assert (info.value.result.calls, info.value.result.tokens_out) == (2, 14)


async def test_results_carry_their_price_basis(fake_litellm: FakeFactory) -> None:
    fake_litellm(["42"])
    hosted = await make_client().complete(HOSTED, ASK)
    local = await FakeOllama(["42"]).client().complete(LOCAL, ASK)
    assert (hosted.price_basis, local.price_basis) == ("listed", "modeled")


async def test_unpriceable_model_fails_before_any_call(fake_litellm: FakeFactory) -> None:
    fake = fake_litellm(["never"])
    unpriced = ModelConfig(arm="strong", model="nobody/made-this-model-up")
    with pytest.raises(UnknownPriceError):
        await make_client().complete(unpriced, ASK)
    assert fake.requests == []


async def test_hosted_arm_waits_for_rate_limit_on_every_attempt(
    fake_litellm: FakeFactory,
) -> None:
    fake_litellm([rate_limited(), "1", "2"])
    limited = HOSTED.model_copy(update={"requests_per_minute": 600.0})  # 0.1 s apart
    client = make_client()
    loop = asyncio.get_running_loop()
    start = loop.time()
    first = await client.complete(limited, ASK)
    second = await client.complete(limited, ASK)
    assert (first.calls, second.calls) == (2, 1)
    assert loop.time() - start >= 0.15  # 2 waits of 0.1 s, minus Windows timer slack
    assert second.wall_s - second.latency_s >= 0.05  # slot wait is in wall time, not latency


async def test_local_arm_is_not_rate_limited() -> None:
    ollama = FakeOllama(["1", "2", "3"])
    client = ollama.client()
    for _ in range(3):
        await client.complete(LOCAL, ASK)
    assert client.limiters == {}


def test_merged_result_keeps_price_basis() -> None:
    base = dict(
        arm="mid",
        model="ollama_chat/qwen2.5:3b",
        content="x",
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
        shadow_cost_usd=0.0,
        latency_s=0.1,
        wall_s=0.1,
        calls=1,
    )
    first = LLMResult(price_basis="modeled", **base)  # type: ignore[arg-type]
    assert first.merged(first).price_basis == "modeled"


async def test_reasoning_effort_is_sent_when_set(fake_litellm: FakeFactory) -> None:
    fake = fake_litellm(["ok", "ok"])
    await make_client().complete(HOSTED.model_copy(update={"reasoning_effort": "low"}), ASK)
    await make_client().complete(HOSTED, ASK)
    assert fake.requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in fake.requests[1]
