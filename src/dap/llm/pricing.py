import math
import re
from collections.abc import Mapping
from typing import Literal

import litellm

PER_MILLION = 1_000_000
LOCAL_PREFIXES = ("ollama/", "ollama_chat/")
PARAMS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)

Overrides = Mapping[str, tuple[float, float]]

# Where a call's shadow price comes from, stored on every llm.call event (review F10):
#   listed   - the provider's list price, from LiteLLM's price table
#   override - a price set by hand in LLM_PRICE_OVERRIDES
#   modeled  - a local model priced as the anchor's list price x (params / anchor params)
PriceBasis = Literal["listed", "override", "modeled"]


class UnknownPriceError(Exception):
    pass


def is_local_model(model: str) -> bool:
    return model.startswith(LOCAL_PREFIXES)


def params_billions(model: str) -> float:
    found = PARAMS_PATTERN.findall(model)
    if not found:
        raise UnknownPriceError(f"cannot read a parameter count like '7b' from {model!r}")
    return float(found[-1])


def listed_cost(model: str, tokens_in: int, tokens_out: int, overrides: Overrides) -> float:
    if model in overrides:
        price_in, price_out = overrides[model]
        return (tokens_in * price_in + tokens_out * price_out) / PER_MILLION
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=tokens_in, completion_tokens=tokens_out
        )
    except Exception as exc:
        raise UnknownPriceError(
            f"no price for {model!r}; add it to LLM_PRICE_OVERRIDES as [in, out] USD per 1M tokens"
        ) from exc
    return float(prompt_cost + completion_cost)


def price_per_million(model: str, overrides: Overrides) -> tuple[float, float]:
    """(input, output) USD per 1M tokens, from overrides or LiteLLM's table."""
    return (
        listed_cost(model, PER_MILLION, 0, overrides),
        listed_cost(model, 0, PER_MILLION, overrides),
    )


def call_cost(model: str, tokens_in: int, tokens_out: int, overrides: Overrides) -> float:
    if is_local_model(model):
        return 0.0
    return listed_cost(model, tokens_in, tokens_out, overrides)


def price_basis(model: str, overrides: Overrides) -> PriceBasis:
    if model in overrides:
        return "override"
    return "modeled" if is_local_model(model) else "listed"


def shadow_cost(
    model: str, tokens_in: int, tokens_out: int, anchor: str, overrides: Overrides
) -> float:
    if price_basis(model, overrides) != "modeled":
        return listed_cost(model, tokens_in, tokens_out, overrides)
    scale = params_billions(model) / params_billions(anchor)
    return scale * listed_cost(anchor, tokens_in, tokens_out, overrides)


def validate_overrides(overrides: Overrides) -> None:
    for model, prices in overrides.items():
        if len(prices) != 2 or not all(math.isfinite(p) and p >= 0 for p in prices):
            raise UnknownPriceError(
                f"LLM_PRICE_OVERRIDES[{model!r}] must be [in, out], finite and >= 0; got {prices!r}"
            )


def require_positive_price(model: str, overrides: Overrides, role: str) -> None:
    price_in, price_out = price_per_million(model, overrides)
    if price_in <= 0 and price_out <= 0:
        raise UnknownPriceError(
            f"{role} {model!r} is priced at $0 per 1M tokens; a paid model with no price would "
            "hide its cost. Set its real price in LLM_PRICE_OVERRIDES as [in, out]."
        )


def validate_price(model: str, anchor: str, overrides: Overrides) -> None:
    """Fail at startup, before any paid call, if a model's cost cannot be computed (review N03)."""
    validate_overrides(overrides)
    if is_local_model(anchor):
        raise UnknownPriceError(f"shadow anchor {anchor!r} must be a hosted model with a price")
    require_positive_price(anchor, overrides, "shadow anchor")
    if not is_local_model(model) and model not in overrides:
        require_positive_price(model, overrides, "hosted model")
    call_cost(model, 1, 1, overrides)
    shadow_cost(model, 1, 1, anchor, overrides)
