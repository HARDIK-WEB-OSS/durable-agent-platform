"""Run the benchmark tasks through the agent on one or more arms and grade them.

    uv run python scripts/run_eval.py --arms mid --limit 2              # smoke: 2 per dataset
    uv run python scripts/run_eval.py --arms cheap mid                  # train + val, local
    uv run python scripts/run_eval.py --arms strong --allow-paid        # hosted arm
    uv run python scripts/run_eval.py --arms strong --allow-paid --retry-errors
    uv run python scripts/run_eval.py --summary                         # table only

Safe to stop and re-run with the same --name: finished pairs are skipped, in-flight runs
resume. The test split needs --final-test and is meant only for the final benchmark.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dap.agent.loop import build_runtime
from dap.agent.tools.python_exec import DockerSandbox
from dap.config import get_settings
from dap.db.session import make_engine, make_sessionmaker
from dap.engine.runner import EngineConfig
from dap.evals.graders import GRADING_SANDBOX_TIMEOUT_S, GradingContext
from dap.evals.harness import (
    EvalConfig,
    check_splits,
    evaluate_arm,
    format_summary,
    summarize,
)
from dap.evals.loaders import DATASETS
from dap.evals.split import SPLITS, load_split_tasks

RAW_DIR = Path("data/raw")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--name", default="v1", help="eval name; results are keyed by it")
    parser.add_argument("--arms", nargs="+", default=["mid"], choices=["cheap", "mid", "strong"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=SPLITS)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--limit", type=int, help="at most N tasks per dataset and split")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-paid", action="store_true", help="allow hosted (paid) arms")
    parser.add_argument("--final-test", action="store_true", help="allow the test split")
    parser.add_argument("--summary", action="store_true", help="print the table and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    check_splits(args.splits, args.final_test)
    settings = get_settings()
    engine = make_engine(settings)
    sm = make_sessionmaker(engine)
    try:
        if not args.summary:
            runtime = await build_runtime(settings, sm)
            for arm in args.arms:
                model = runtime.arms.get(arm)
                if model is None:
                    print(f"arm {arm!r} is not configured in .env")
                    return 1
                if not model.local and not args.allow_paid:
                    print(f"{arm} is hosted ({model.model}); add --allow-paid to run it")
                    return 1
            sandbox = DockerSandbox(settings.sandbox_image, timeout_s=GRADING_SANDBOX_TIMEOUT_S)
            ctx = GradingContext(sandbox, settings.sql_data_dir)
            tasks = [
                t
                for t in load_split_tasks(RAW_DIR, settings.sql_data_dir)
                if t.split in args.splits and t.task.dataset in args.datasets
            ]
            if args.limit:
                kept, seen = [], {}
                for t in tasks:
                    key = (t.task.dataset, t.split)
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] <= args.limit:
                        kept.append(t)
                tasks = kept
            cfg = EvalConfig(
                name=args.name,
                max_turns=args.max_turns,
                concurrency=args.concurrency,
                retry_errors=args.retry_errors,
            )
            for arm in args.arms:
                counts = await evaluate_arm(
                    sm, runtime, ctx, tasks, arm, cfg, EngineConfig.from_settings(settings)
                )
                print(f"{arm}: {dict(counts)}\n")
        print(f"eval {args.name!r}, splits {args.splits}")
        print(format_summary(await summarize(sm, args.name, args.splits)))
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
