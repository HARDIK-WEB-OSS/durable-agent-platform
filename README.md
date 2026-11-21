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

## Benchmark tasks

300 tasks, 100 from each of three public datasets, chosen once with seed 2026 and split 60/20/20
into train, val and test within each dataset:

| Dataset | Source | Licence | Tools the agent gets | Grader |
|---|---|---|---|---|
| GSM8K (test) | [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT | `calculator`, `python_exec` | exact numeric match of the last number in the answer |
| MBPP (sanitized) | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) | CC BY 4.0 | `python_exec` | the answer's code plus the dataset's asserts run in the Docker sandbox |
| Spider 1.0 (dev) | [yale-lily.github.io/spider](https://yale-lily.github.io/spider) | CC BY-SA 4.0 | `sql_query` | execution match: predicted and gold SQL run read-only, results compared |

**No LLM grades anything.** Details:

- The MBPP grader prints a random per-run marker after the asserts, so code that exits early or
  prints a fake "passed" line fails.
- Spider results are compared as multisets, or as lists when the gold query has `ORDER BY`. The
  column order may differ, and floats are compared to 6 decimals.
- A task was kept only if its own gold answer passes its grader. 3 Spider questions whose gold
  query returns nothing were skipped, so an empty result can't match by accident.

The split is stored in `src/dap/evals/tasks/split_v1.json`, which is committed. It holds task ids,
splits, the seed, and each source file's URL, licence and SHA-256 hash. The raw data is not
committed. Loading refuses to run if a raw file's hash differs, and the seed script refuses to
overwrite the manifest.

**The test split is reserved for the final benchmark.** The harness refuses it unless you pass
`--final-test`.

Known limitation: Spider databases are shared across splits (the split is by question), so the
router sees test-time databases during training.

### Setup

```powershell
# raw files go in data/raw (git-ignored)
New-Item -ItemType Directory -Force data\raw\gsm8k, data\raw\mbpp | Out-Null
Invoke-WebRequest "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" -OutFile data\raw\gsm8k\test.jsonl
Invoke-WebRequest "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json" -OutFile data\raw\mbpp\sanitized-mbpp.json
# Spider: download the dataset zip from https://yale-lily.github.io/spider and put
# dev.json, tables.json and database/<db>/<db>.sqlite under data\raw\spider
```

Then copy the Spider databases into `SQL_DATA_DIR`. The split is not regenerated; it already
exists:

```powershell
uv run python -c "from pathlib import Path; from dap.evals.split import install_spider_databases as i; i(Path('data/raw'), Path('data/sql'))"
```

### Running the evaluation

```powershell
uv run python scripts/run_eval.py --arms mid --limit 2               # smoke test, 2 tasks per dataset
uv run python scripts/run_eval.py --arms cheap mid                   # train + val on the local arms
uv run python scripts/run_eval.py --arms strong --allow-paid         # hosted arm
uv run python scripts/run_eval.py --arms strong --allow-paid --retry-errors
uv run python scripts/run_eval.py --summary
```

Results go to the `eval_results` table, one row per (task, arm), including tokens, paid and
shadow cost, LLM and tool time, and turns. You can stop and restart at any time:
- finished pairs are skipped
- runs still in flight are resumed (they are durable)
- infrastructure failures, such as a provider's daily limit, are recorded as `error`, not as
  wrong answers; they are left out of accuracy and retried with `--retry-errors`

An arm stops taking new tasks after 5 errors in a row.

Keep `--concurrency 1` for local arms: Ollama answers one request at a time, so queueing inside
it would be counted as model latency.

`python_exec` echoes the value of a script's final bare expression, as a notebook does. Small
models often end with `total` instead of `print(total)`. Only the agent's tool does this; graders
run submitted code exactly as written.
