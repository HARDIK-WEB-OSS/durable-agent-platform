import asyncio
import random
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import litellm
from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from dap.backoff import backoff_seconds
from dap.config import Settings
from dap.db import repo
from dap.db.models import Event
from dap.llm.pricing import (
    PriceBasis,
    call_cost,
    is_local_model,
    price_basis,
    shadow_cost,
    validate_price,
)
from dap.llm.ratelimit import RateLimiter

litellm.suppress_debug_info = True

Message = dict[str, str]
CHARS_PER_TOKEN_ESTIMATE = 3
DIGIT = re.compile(r"\d")
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class ContextTooLongError(LLMError):
    pass


class TransientLLMError(LLMError):
    pass


TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    Timeout,
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
    TimeoutError,
    TransientLLMError,
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    model: str
    api_key: SecretStr | None = None
    api_base: str | None = None
    num_ctx: int | None = None
    requests_per_minute: float | None = None
    reasoning_effort: str | None = None

    @property
    def local(self) -> bool:
        return is_local_model(self.model)


class LLMCallError(LLMError):
    def __init__(self, model: ModelConfig, calls: int, wall_s: float, cause: Exception) -> None:
        plural = "" if calls == 1 else "s"
        super().__init__(
            f"{model.model} failed after {calls} call{plural}: {type(cause).__name__}: {cause}"
        )
        self.arm = model.arm
        self.model = model.model
        self.calls = calls
        self.wall_s = wall_s
        self.error_type = type(cause).__name__


class LLMResult(BaseModel):
    arm: str
    model: str
    content: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    shadow_cost_usd: float
    price_basis: PriceBasis
    latency_s: float
    wall_s: float
    calls: int
    truncated: bool = False

    def merged(self, later: "LLMResult") -> "LLMResult":
        return later.model_copy(
            update={
                "tokens_in": self.tokens_in + later.tokens_in,
                "tokens_out": self.tokens_out + later.tokens_out,
                "cost_usd": self.cost_usd + later.cost_usd,
                "shadow_cost_usd": self.shadow_cost_usd + later.shadow_cost_usd,
                "latency_s": self.latency_s + later.latency_s,
                "wall_s": self.wall_s + later.wall_s,
                "calls": self.calls + later.calls,
                "truncated": self.truncated or later.truncated,
            }
        )


class StructuredOutputError(LLMError):
    def __init__(self, message: str, result: LLMResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class Completion:
    content: str
    tokens_in: int
    tokens_out: int
    hit_max_tokens: bool


ARM_NAMES = ("cheap", "mid", "strong")


def reasoning_effort(model: str, settings: Settings) -> str | None:
    if settings.llm_reasoning_effort is None:
        return None
    try:
        is_reasoning = litellm.supports_reasoning(model=model)
    except Exception:
        is_reasoning = False
    return settings.llm_reasoning_effort if is_reasoning else None


def configured_arms(settings: Settings) -> dict[str, ModelConfig]:
    arms: dict[str, ModelConfig] = {}
    for arm in ARM_NAMES:
        model: str | None = getattr(settings, f"{arm}_model")
        if not model:
            continue
        validate_price(model, settings.llm_shadow_anchor, settings.llm_price_overrides)
        local = is_local_model(model)
        arms[arm] = ModelConfig(
            arm=arm,
            model=model,
            api_key=getattr(settings, f"{arm}_api_key"),
            api_base=settings.ollama_api_base if local else None,
            num_ctx=settings.local_num_ctx if local else None,
            requests_per_minute=None if local else settings.llm_hosted_rpm,
            reasoning_effort=None if local else reasoning_effort(model, settings),
        )
    return arms


def litellm_kwargs(
    model: ModelConfig,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
        "num_retries": 0,
        "max_retries": 0,
    }
    if model.api_key is not None:
        kwargs["api_key"] = model.api_key.get_secret_value()
    if model.api_base is not None:
        kwargs["api_base"] = model.api_base
    if model.reasoning_effort is not None:
        kwargs["reasoning_effort"] = model.reasoning_effort
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


def ollama_payload(
    model: ModelConfig,
    messages: list[Message],
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
    if model.num_ctx is not None:
        options["num_ctx"] = model.num_ctx
    payload: dict[str, Any] = {
        "model": model.model.split("/", 1)[1],
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if response_format is not None:
        json_schema = response_format.get("json_schema")
        payload["format"] = json_schema["schema"] if json_schema else "json"
    return payload


def estimate_tokens(text: str) -> int:
    digits = len(DIGIT.findall(text))
    return digits + (len(text) - digits) // CHARS_PER_TOKEN_ESTIMATE


def estimate_prompt_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


def check_context(model: ModelConfig, messages: list[Message], max_tokens: int) -> None:
    if model.num_ctx is None:
        return
    prompt_tokens = estimate_prompt_tokens(messages)
    if prompt_tokens + max_tokens > model.num_ctx:
        raise ContextTooLongError(
            f"~{prompt_tokens} prompt + {max_tokens} output tokens exceed "
            f"{model.arm} context of {model.num_ctx}"
        )


def is_truncated(model: ModelConfig, completion: Completion, max_tokens: int) -> bool:
    context_full = model.num_ctx is not None and (
        completion.tokens_in + max_tokens >= model.num_ctx
    )
    return completion.hit_max_tokens or context_full


def json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    return stripped.strip()


@dataclass(frozen=True)
class LLMClient:
    timeout_s: float
    max_retries: int
    shadow_anchor: str
    price_overrides: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    retry_base_s: float = 1.0
    retry_cap_s: float = 20.0
    rng: random.Random = field(default_factory=random.Random)
    transport: httpx.AsyncBaseTransport | None = None
    limiters: dict[str, RateLimiter] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        anchor = settings.llm_shadow_anchor
        validate_price(anchor, anchor, settings.llm_price_overrides)
        return cls(
            timeout_s=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            shadow_anchor=settings.llm_shadow_anchor,
            price_overrides=settings.llm_price_overrides,
        )

    async def complete(
        self,
        model: ModelConfig,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        validate_price(model.model, self.shadow_anchor, self.price_overrides)
        check_context(model, messages, max_tokens)
        start = time.perf_counter()
        calls = 0
        while True:
            calls += 1
            await self.wait_for_slot(model)
            call_start = time.perf_counter()
            try:
                async with asyncio.timeout(self.timeout_s):
                    completion = await self.send(
                        model, messages, temperature, max_tokens, response_format
                    )
                break
            except TRANSIENT_ERRORS as exc:
                if calls > self.max_retries:
                    raise LLMCallError(model, calls, time.perf_counter() - start, exc) from exc
                delay = backoff_seconds(calls, self.retry_base_s, self.retry_cap_s, self.rng)
                await asyncio.sleep(delay)
            except Exception as exc:
                raise LLMCallError(model, calls, time.perf_counter() - start, exc) from exc
        end = time.perf_counter()
        tokens_in, tokens_out = completion.tokens_in, completion.tokens_out
        return LLMResult(
            arm=model.arm,
            model=model.model,
            content=completion.content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=call_cost(model.model, tokens_in, tokens_out, self.price_overrides),
            shadow_cost_usd=shadow_cost(
                model.model, tokens_in, tokens_out, self.shadow_anchor, self.price_overrides
            ),
            price_basis=price_basis(model.model, self.price_overrides),
            latency_s=end - call_start,
            wall_s=end - start,
            calls=calls,
            truncated=is_truncated(model, completion, max_tokens),
        )

    async def wait_for_slot(self, model: ModelConfig) -> None:
        """Client-side rate limit, applied to every attempt including retries."""
        if model.requests_per_minute is None:
            return
        limiter = self.limiters.get(model.model)
        if limiter is None:
            limiter = RateLimiter(model.requests_per_minute)
            self.limiters[model.model] = limiter
        await limiter.acquire()

    async def send(
        self,
        model: ModelConfig,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> Completion:
        if model.local:
            payload = ollama_payload(model, messages, temperature, max_tokens, response_format)
            return await self.ollama_chat(model, payload)
        kwargs = litellm_kwargs(model, messages, temperature, max_tokens, response_format)
        return await self.litellm_chat(kwargs)

    async def ollama_chat(self, model: ModelConfig, payload: dict[str, Any]) -> Completion:
        base_url = model.api_base or "http://localhost:11434"
        async with httpx.AsyncClient(
            base_url=base_url, timeout=self.timeout_s, transport=self.transport
        ) as http:
            try:
                response = await http.post("/api/chat", json=payload)
            except httpx.TransportError as exc:
                raise TransientLLMError(f"ollama unreachable: {exc!r}") from exc
        status = response.status_code
        if status == 429 or status >= 500:
            raise TransientLLMError(f"ollama HTTP {status}: {response.text[:200]}")
        if status >= 400:
            raise LLMError(f"ollama HTTP {status}: {response.text[:200]}")
        body = response.json()
        return Completion(
            content=body["message"]["content"],
            tokens_in=body.get("prompt_eval_count", 0),
            tokens_out=body.get("eval_count", 0),
            hit_max_tokens=body.get("done_reason") == "length",
        )

    async def litellm_chat(self, kwargs: dict[str, Any]) -> Completion:
        response = await acompletion(timeout=self.timeout_s, **kwargs)
        choice = response.choices[0]
        return Completion(
            content=choice.message.content or "",
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            hit_max_tokens=getattr(choice, "finish_reason", None) == "length",
        )

    async def complete_structured(
        self,
        model: ModelConfig,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        repair_attempts: int = 1,
    ) -> tuple[T, LLMResult]:
        history = list(messages)
        total: LLMResult | None = None
        for _ in range(repair_attempts + 1):
            result = await self.complete(
                model,
                history,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=json_schema_format(schema),
            )
            total = result if total is None else total.merged(result)
            try:
                return schema.model_validate_json(strip_code_fence(result.content)), total
            except ValidationError as exc:
                history += [
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": "That reply did not match the JSON schema: "
                        f"{exc.errors(include_url=False)}. Reply with only the corrected JSON.",
                    },
                ]
        assert total is not None
        raise StructuredOutputError(f"{model.arm} never produced valid {schema.__name__}", total)


async def record_llm_call(
    session: AsyncSession,
    run_id: uuid.UUID,
    result: LLMResult,
    step_id: uuid.UUID | None = None,
    ok: bool = True,
) -> Event:
    payload = {**result.model_dump(), "ok": ok}
    return await repo.append_event(session, run_id, "llm.call", payload, step_id=step_id)


async def record_llm_failure(
    session: AsyncSession,
    run_id: uuid.UUID,
    error: LLMCallError,
    step_id: uuid.UUID | None = None,
) -> Event:
    payload = {
        "arm": error.arm,
        "model": error.model,
        "calls": error.calls,
        "wall_s": error.wall_s,
        "error_type": error.error_type,
        "ok": False,
    }
    return await repo.append_event(session, run_id, "llm.call", payload, step_id=step_id)
