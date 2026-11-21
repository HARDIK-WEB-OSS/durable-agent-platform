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

## Agent

`src/dap/agent/` runs a ReAct-style agent on the durable engine. **Each LLM call and each tool
call is its own step**:

```
agent.think (turn 1) -> agent.act (turn 1) -> agent.think (turn 2) -> ... -> final_answer
```

The model replies with one JSON object, `{"thought", "tool", "tool_input"}`. The transcript is
never stored as a whole; it is rebuilt from the outputs of the run's committed steps, so a step
retried on another worker sees exactly the history committed before it.

### What survives a crash

| Effect | Guarantee | How |
|---|---|---|
| Writes to our own tables (`ctx.once`) | exactly once | effect and idempotency key commit in one transaction |
| LLM calls and tool runs (`ctx.external`) | at least once, **recorded once** | result is stored under an idempotency key; a retried step reuses it |

An LLM call or tool run repeats only if the worker dies in the gap between the call returning
and its result committing. While an external call runs, no DB connection or transaction is
held (review F02). Every real LLM call is logged as an `llm.call` event, including one that a
crash forces us to repeat, so cost totals stay honest.

### Limits

Checked before every LLM turn. Hitting one fails the run, with the reason as its error:

- `AGENT_MAX_STEPS` turns (a run can override this with `max_turns`)
- `AGENT_TIMEOUT_SECONDS` of wall-clock time since the run was created
- the run's `budget_usd`, compared with the paid cost of its `llm.call` events (a turn already
  in flight can overshoot by one call)
- the local model's context window: a transcript that no longer fits stops the run instead of
  being silently cut

### Tools

| Tool | Input | Safety |
|---|---|---|
| `calculator` | arithmetic expression | parsed with `ast`; only numbers, operators and a fixed list of math functions; exponent and result size capped |
| `python_exec` | Python script | hardened Docker container (see below) |
| `sql_query` | one SQLite `SELECT` | file opened `mode=ro` + `PRAGMA query_only` + an authorizer that allows only reads (no ATTACH, PRAGMA or DDL); one statement; 5 s timeout; 50-row cap; the file must be inside `SQL_DATA_DIR` |
| `search_docs` | search query | pgvector cosine search over `documents`; results are wrapped as `<document>` data |

**Prompt injection.** Tool output reaches the model inside an `<observation>` block marked as
data, and the system prompt tells the model never to follow instructions found there. A
document cannot close the block early. This lowers the risk; it does not remove it, because
small models do not always obey. The real protection is that no tool can change anything
outside its sandbox: SQL is read-only and Python has no network.

### python_exec sandbox

Each call runs in a fresh container that is thrown away afterwards:

```
docker run --rm -i --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=64m
  --memory 256m --memory-swap 256m --cpus 1 --pids-limit 64 --cap-drop ALL
  --security-opt no-new-privileges --user 65534:65534
  python:3.11-slim@sha256:07d48240...  python -I -
```

The code goes in on stdin, so nothing from the host is mounted. A wall-clock timeout
(`SANDBOX_TIMEOUT_SECONDS`) removes the container by name, not just the `docker` process.
Output is capped while it is being read. The image is pinned by digest, and the worker refuses
to start if the image is missing. `tests/integration/test_sandbox.py` runs the real container
and proves that:

- network access fails
- writing outside `/tmp` fails
- the process runs as uid 65534 with no capabilities
- the timeout kills the container
- the memory and process limits hold
- runaway output is cut off

**Hosts without Docker** (for example the deployed demo): set `PYTHON_EXEC_ENABLED=false`.
`python_exec` is then not offered to the model at all, and tasks that need it will do worse or
fail. Say so wherever results from that host are shown.

### Setup and demo

```powershell
docker pull python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
ollama pull nomic-embed-text
uv run alembic upgrade head
uv run python scripts/ingest_docs.py README.md
uv run python scripts/agent_demo.py "What is 17 * 23 + 4?" --arm mid --tools calculator
```

`nomic-embed-text` is used with its task prefixes: `search_document: ` when documents are
ingested and `search_query: ` for queries. `agent_demo.py` refuses hosted (paid) arms unless
you pass `--allow-paid`.
