# Durable Agent Platform with RL Router

Runs AI agents on long multi-step jobs, survives worker crashes without losing progress or
repeating side effects, and uses a contextual-bandit router to pick the cheapest LLM that can
do each task well.

Work in progress. Benchmark numbers will be added once they are produced.

## Run locally

```powershell
Copy-Item .env.example .env
docker compose up -d
uv run pytest
```

## LLM arms

The router chooses between three arms. Model strings use LiteLLM's `<provider>/<model>` format
and are set in `.env`.

| Arm      | Default model              | Runs on                 | Paid cost (`cost_usd`) | Shadow cost basis |
|----------|----------------------------|-------------------------|------------------------|-------------------|
| `cheap`  | `ollama_chat/qwen2.5:1.5b` | local Ollama            | $0                     | modeled           |
| `mid`    | `ollama_chat/qwen2.5:3b`   | local Ollama            | $0                     | modeled           |
| `strong` | none: you choose           | hosted API via LiteLLM  | provider's price       | listed            |

Set up the local arms:

```powershell
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
```

Set up the strong arm in `.env`: pick a provider, then set `STRONG_MODEL` (for example
`groq/<model>` or `together_ai/<model>`) and `STRONG_API_KEY`.

### How cost is counted

Every LLM call is stored as an `llm.call` event with `cost_usd`, `shadow_cost_usd` and
`price_basis`. `price_basis` tells you where the shadow price comes from:

- **listed**: the provider's list price, from LiteLLM's bundled price table. This is what the
  hosted strong arm is charged.
- **override**: a price you set in `LLM_PRICE_OVERRIDES` as `[input, output]` USD per 1M tokens.
- **modeled**: a local model. Its shadow price is an assumption, not a market price. It is the
  anchor's list price (`LLM_SHADOW_ANCHOR`, Qwen2.5-7B on DeepInfra) times
  `params / 7`. Cost-saving results that involve local arms depend on this assumption and are
  labelled as modeled.

LiteLLM's table is bundled with the installed version and can go out of date. Compare the price
that `scripts/llm_smoke.py` prints with the provider's pricing page. If they differ, put the
current price in `LLM_PRICE_OVERRIDES`.

Prices are checked at startup, before any paid call. Startup fails if a hosted model has no
price, is priced at $0 without an explicit override, or if an override is negative or not a
number. A paid response is therefore never thrown away because its cost could not be computed.

### Rate limiting

Hosted arms are limited on the client to `LLM_HOSTED_RPM` requests per minute (default 30),
spaced evenly with no bursts. Every retry counts as a request. The wait shows up in `wall_s`, not
in `latency_s`. The limit applies per process, so N worker processes can send up to
N x `LLM_HOSTED_RPM`. Set it to the provider's limit divided by the number of workers. Local arms
are not limited because Ollama already runs one request at a time. If the provider still answers
with HTTP 429, the call is retried with backoff.

### Smoke test (Phase 3 acceptance)

```powershell
uv run python scripts/llm_smoke.py               # local arms only, spends nothing
uv run python scripts/llm_smoke.py --allow-paid  # also calls the hosted arm
```

Each arm is asked a small math question twice (`cold`, then `warm`) and must reply with
schema-valid JSON. Before any call, the script prints the hosted arm's price per 1M tokens and a
worst-case cost for the run. Without `--allow-paid` the hosted arm is not called and is reported
as `NOT RUN`. The script exits 0 only when all three arms pass.
