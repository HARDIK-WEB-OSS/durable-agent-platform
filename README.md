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
