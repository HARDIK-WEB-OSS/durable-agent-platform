"""Run one agent task end to end on an in-process worker and print what happened.

    uv run python scripts/agent_demo.py "What is 17 * 23 + 4?" --arm mid --tools calculator
    uv run python scripts/agent_demo.py "Sum the squares of 1..20" --tools python_exec

Hosted (paid) arms need --allow-paid. Every step is durable: kill this script mid-run and
start a worker (python -m dap.engine.worker) to watch the run continue where it stopped.
"""

import argparse
import asyncio
import json
import sys
from decimal import Decimal

from dap.agent.loop import build_runtime, submit_agent_run
from dap.agent.memory import ACT_KIND, THINK_KIND
from dap.config import get_settings
from dap.db import repo
from dap.db.session import make_engine, make_sessionmaker
from dap.engine.runner import EngineConfig
from dap.engine.worker import Worker
from dap.states import RunStatus

TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


def show(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("task")
    parser.add_argument("--arm", default="mid", choices=["cheap", "mid", "strong"])
    parser.add_argument("--tools", nargs="*", help="default: every tool this host offers")
    parser.add_argument("--sql-db", help="file name inside SQL_DATA_DIR")
    parser.add_argument("--budget", type=Decimal, help="USD cap on paid LLM cost")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--allow-paid", action="store_true", help="allow a hosted (paid) arm")
    args = parser.parse_args()

    settings = get_settings()
    engine = make_engine(settings)
    sm = make_sessionmaker(engine)
    try:
        runtime = await build_runtime(settings, sm)
        model = runtime.arms.get(args.arm)
        if model is None:
            print(f"arm {args.arm!r} is not configured in .env")
            return 1
        if not model.local and not args.allow_paid:
            print(f"{args.arm} is hosted ({model.model}); re-run with --allow-paid")
            return 1
        async with sm.begin() as s:
            run = await submit_agent_run(
                s,
                args.task,
                args.arm,
                tools=args.tools,
                sql_db=args.sql_db,
                budget_usd=args.budget,
                max_turns=args.max_turns,
            )
        print(f"run {run.id} on {args.arm} ({model.model})\n")
        worker = Worker("demo", sm, runtime.handlers(), EngineConfig.from_settings(settings))
        task = asyncio.create_task(worker.run())
        while True:
            async with sm() as s:
                current = await repo.get_run(s, run.id)
            assert current is not None
            if current.status in TERMINAL:
                break
            await asyncio.sleep(0.2)
        worker.stop.set()
        await task
        async with sm() as s:
            steps = await repo.list_steps(s, run.id)
            events = await repo.list_events(s, run.id)
        for step in steps:
            out = step.output or {}
            if step.kind == THINK_KIND and "action" in out:
                a = out["action"]
                print(f"[{step.seq}] think: {show(a['thought'], 200)}")
                print(f"     -> {a['tool']}: {show(a['tool_input'])}")
            elif step.kind == THINK_KIND:
                print(f"[{step.seq}] think: {show(json.dumps(out), 300)}")
            elif step.kind == ACT_KIND:
                print(f"[{step.seq}] observation: {show(out.get('observation', ''))}")
        llm = [e.payload for e in events if e.type == "llm.call"]
        paid = sum(p.get("cost_usd", 0) for p in llm)
        shadow = sum(p.get("shadow_cost_usd", 0) for p in llm)
        tokens = sum(p.get("tokens_in", 0) + p.get("tokens_out", 0) for p in llm)
        print(f"\nstatus: {current.status.value}")
        print(f"result: {current.result}" if current.result else f"error: {current.error}")
        print(f"llm calls: {len(llm)}, tokens: {tokens}, paid: ${paid:.6f}, shadow: ${shadow:.8f}")
        return 0 if current.status == RunStatus.SUCCEEDED else 1
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
