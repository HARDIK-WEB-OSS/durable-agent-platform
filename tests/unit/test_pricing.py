import litellm
import pytest

from dap.config import Settings
from dap.llm.client import configured_arms
from dap.llm.pricing import (
    UnknownPriceError,
    call_cost,
    params_billions,
    price_basis,
    price_per_million,
    shadow_cost,
    validate_price,
)

ANCHOR = "deepinfra/Qwen/Qwen2.5-7B-Instruct"


def anchor_cost(tokens_in: int, tokens_out: int) -> float:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=ANCHOR, prompt_tokens=tokens_in, completion_tokens=tokens_out
    )
    return float(prompt_cost + completion_cost)


def test_local_models_cost_nothing_to_run() -> None:
    assert call_cost("ollama_chat/qwen2.5:7b", 1_000_000, 1_000_000, {}) == 0.0


def test_override_prices_are_per_million_tokens() -> None:
    cost = call_cost("my/model", 2_000_000, 500_000, {"my/model": (1.0, 4.0)})
    assert cost == pytest.approx(2.0 + 2.0)


def test_known_hosted_model_uses_litellm_table() -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500
    )
    assert call_cost("gpt-4o-mini", 1000, 500, {}) == pytest.approx(prompt_cost + completion_cost)


def test_unknown_price_is_an_error_not_zero() -> None:
    with pytest.raises(UnknownPriceError, match="LLM_PRICE_OVERRIDES"):
        call_cost("nobody/made-this-model-up", 10, 10, {})


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ollama_chat/qwen2.5:1.5b", 1.5),
        ("ollama_chat/qwen2.5:3b", 3.0),
        ("ollama_chat/qwen2.5:7b", 7.0),
        (ANCHOR, 7.0),
    ],
)
def test_params_are_read_from_model_name(model: str, expected: float) -> None:
    assert params_billions(model) == expected


def test_anchor_has_a_real_listed_price() -> None:
    assert anchor_cost(1_000_000, 1_000_000) > 0


@pytest.mark.parametrize(("model", "scale"), [("1.5b", 1.5 / 7), ("3b", 3 / 7), ("7b", 1.0)])
def test_shadow_price_scales_anchor_by_parameter_count(model: str, scale: float) -> None:
    cost = shadow_cost(f"ollama_chat/qwen2.5:{model}", 1000, 200, ANCHOR, {})
    assert cost == pytest.approx(anchor_cost(1000, 200) * scale)


def test_shadow_price_of_hosted_model_is_its_real_price() -> None:
    assert shadow_cost("gpt-4o-mini", 1000, 200, ANCHOR, {}) == call_cost(
        "gpt-4o-mini", 1000, 200, {}
    )


def test_shadow_price_override_wins() -> None:
    overrides = {"ollama_chat/qwen2.5:3b": (1.0, 1.0)}
    assert shadow_cost("ollama_chat/qwen2.5:3b", 1_000_000, 0, ANCHOR, overrides) == 1.0


def test_default_arms_are_two_local_models_and_no_strong_arm() -> None:
    settings = Settings(_env_file=None, postgres_password="x")  # type: ignore[call-arg]
    arms = configured_arms(settings)
    assert {name: arm.model for name, arm in arms.items()} == {
        "cheap": "ollama_chat/qwen2.5:1.5b",
        "mid": "ollama_chat/qwen2.5:3b",
    }
    assert all(arm.local and arm.num_ctx == settings.local_num_ctx for arm in arms.values())
    assert all(arm.api_base == settings.ollama_api_base for arm in arms.values())
    assert all(arm.requests_per_minute is None for arm in arms.values())


def test_hosted_arm_gets_key_and_no_local_options() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        postgres_password="x",
        strong_model="gpt-4o-mini",
        strong_api_key="k",
    )
    strong = configured_arms(settings)["strong"]
    assert (strong.local, strong.num_ctx, strong.api_base) == (False, None, None)
    assert strong.api_key is not None
    assert strong.requests_per_minute == settings.llm_hosted_rpm


@pytest.mark.parametrize(
    ("arm_setting", "model"),
    [("strong_model", "nobody/made-this-model-up"), ("cheap_model", "ollama_chat/llama3")],
)
def test_unpriceable_arm_fails_at_startup_not_after_a_paid_call(
    arm_setting: str, model: str
) -> None:
    base = Settings(_env_file=None, postgres_password="x")
    settings = base.model_copy(update={arm_setting: model})
    with pytest.raises(UnknownPriceError):
        configured_arms(settings)


@pytest.mark.parametrize(
    ("model", "overrides", "basis"),
    [
        ("gpt-4o-mini", {}, "listed"),
        ("ollama_chat/qwen2.5:3b", {}, "modeled"),
        ("ollama_chat/qwen2.5:3b", {"ollama_chat/qwen2.5:3b": (1.0, 1.0)}, "override"),
        ("gpt-4o-mini", {"gpt-4o-mini": (1.0, 2.0)}, "override"),
    ],
)
def test_price_basis_labels_where_the_price_comes_from(
    model: str, overrides: dict[str, tuple[float, float]], basis: str
) -> None:
    assert price_basis(model, overrides) == basis


def test_price_per_million_reads_override() -> None:
    assert price_per_million("my/model", {"my/model": (0.5, 1.5)}) == (0.5, 1.5)


def test_price_per_million_matches_litellm_table() -> None:
    price_in, price_out = price_per_million("gpt-4o-mini", {})
    assert call_cost("gpt-4o-mini", 1000, 500, {}) == pytest.approx(
        (1000 * price_in + 500 * price_out) / 1_000_000
    )


def test_hosted_model_listed_at_zero_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    real = litellm.cost_per_token

    def free_hosted(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
        if model == "free/model":
            return 0.0, 0.0
        return real(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    monkeypatch.setattr(litellm, "cost_per_token", free_hosted)
    with pytest.raises(UnknownPriceError, match=r"\$0"):
        validate_price("free/model", ANCHOR, {})
    validate_price("free/model", ANCHOR, {"free/model": (0.1, 0.2)})


@pytest.mark.parametrize("bad", [(-1.0, 1.0), (1.0, float("nan")), (float("inf"), 1.0)])
def test_bad_override_values_are_rejected(bad: tuple[float, float]) -> None:
    with pytest.raises(UnknownPriceError, match="LLM_PRICE_OVERRIDES"):
        validate_price("ollama_chat/qwen2.5:3b", ANCHOR, {"x/y": bad})


def test_local_anchor_is_rejected() -> None:
    with pytest.raises(UnknownPriceError, match="anchor"):
        validate_price("ollama_chat/qwen2.5:3b", "ollama_chat/qwen2.5:7b", {})


def test_hosted_rpm_must_be_positive() -> None:
    with pytest.raises(ValueError, match="llm_hosted_rpm"):
        Settings(_env_file=None, postgres_password="x", llm_hosted_rpm=0)  # type: ignore[call-arg]


def test_reasoning_effort_only_goes_to_hosted_reasoning_models() -> None:
    base = Settings(_env_file=None, postgres_password="x")  # type: ignore[call-arg]
    reasoning = base.model_copy(update={"strong_model": "groq/openai/gpt-oss-120b"})
    plain = base.model_copy(update={"strong_model": "gpt-4o-mini"})
    off = reasoning.model_copy(update={"llm_reasoning_effort": None})
    assert configured_arms(reasoning)["strong"].reasoning_effort == "low"
    assert configured_arms(plain)["strong"].reasoning_effort is None
    assert configured_arms(off)["strong"].reasoning_effort is None
    assert configured_arms(reasoning)["mid"].reasoning_effort is None
