"""Choose the benchmark tasks once and write src/dap/evals/tasks/split_v1.json.

    uv run python scripts/seed_tasks.py            # refuses if the manifest already exists

Needs the raw files in data/raw (see README, "Benchmark tasks") and Docker with the sandbox
image, because MBPP gold solutions are checked by running their asserts in the sandbox.
"""

import argparse
import asyncio
import random
import sys
from collections import Counter
from pathlib import Path

from dap.agent.tools.python_exec import DockerSandbox
from dap.agent.tools.sql_query import read_only_query, resolve_database
from dap.config import get_settings
from dap.evals.graders import GRADING_SANDBOX_TIMEOUT_S, GradingContext, gold_answer, grade
from dap.evals.loaders import DATASETS, Task, load_all
from dap.evals.split import (
    MANIFEST_PATH,
    build_manifest,
    install_spider_databases,
    save_manifest,
    select,
)

RAW_DIR = Path("data/raw")


def spider_has_rows(task: Task, ctx: GradingContext) -> bool:
    db = resolve_database(ctx.sql_data_dir, task.sql_db or "")
    _, rows = read_only_query(db, task.reference["gold_sql"], ctx.sql_timeout_s, 1000)
    return any(v is not None for row in rows for v in row)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = parser.parse_args()
    if MANIFEST_PATH.exists() and not args.force:
        print(f"{MANIFEST_PATH} exists; the split is made once. Use --force for a new version.")
        return 1

    settings = get_settings()
    copied = install_spider_databases(RAW_DIR, settings.sql_data_dir)
    print(f"spider databases installed into {settings.sql_data_dir / 'spider'}: {len(copied)} new")
    # Grading limit, not the agent's tool limit: it includes container start-up, and a slow
    # start must never reject a good task (that would make the selection depend on load).
    sandbox = DockerSandbox(settings.sandbox_image, timeout_s=GRADING_SANDBOX_TIMEOUT_S)
    await sandbox.check_available()
    ctx = GradingContext(sandbox, settings.sql_data_dir, sql_timeout_s=10.0)
    rejected: Counter[str] = Counter()

    async def accept(task: Task) -> str | None:
        result = await grade(task, gold_answer(task), ctx)
        if not result.correct:
            return f"gold answer fails its grader: {result.detail}"
        if task.dataset == "spider" and not await asyncio.to_thread(spider_has_rows, task, ctx):
            return "gold query returns no non-NULL value"
        return None

    def on_reject(task: Task, reason: str) -> None:
        rejected[task.dataset] += 1
        print(f"  skip {task.id}: {reason[:120]}")

    chosen = {}
    for dataset in DATASETS:
        candidates = load_all(RAW_DIR, dataset)
        rng = random.Random(f"{args.seed}:select:{dataset}")
        chosen[dataset] = await select(candidates, args.per_dataset, rng, accept, on_reject)
        print(
            f"{dataset}: kept {args.per_dataset} of {len(candidates)} (skipped {rejected[dataset]})"
        )

    manifest = build_manifest(args.seed, args.per_dataset, chosen, RAW_DIR, settings.sql_data_dir)
    save_manifest(manifest, force=args.force)
    counts = Counter((t["dataset"], t["split"]) for t in manifest["tasks"])
    for dataset in DATASETS:
        row = ", ".join(f"{s}={counts[(dataset, s)]}" for s in ("train", "val", "test"))
        print(f"  {dataset}: {row}")
    print(f"wrote {MANIFEST_PATH} ({len(manifest['tasks'])} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
