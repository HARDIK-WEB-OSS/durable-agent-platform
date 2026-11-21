"""The task split: chosen once with a fixed seed, saved, never regenerated.

The committed manifest (src/dap/evals/tasks/split_v1.json) holds only task ids, their split,
the seed, and the source files' URLs, licenses and SHA-256 hashes. Task text is rebuilt from
the raw files, and loading refuses if a raw file's hash differs from the manifest, so every
run of every experiment sees exactly the same 300 tasks.

The test split is for the final benchmark only (Phase 8); the harness refuses it otherwise.
"""

import hashlib
import json
import random
import shutil
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dap.evals.loaders import DATASETS, RAW_FILES, Dataset, Task, load_all

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
MANIFEST_PATH = Path(__file__).parent / "tasks" / "split_v1.json"

SOURCES: dict[Dataset, dict[str, str]] = {
    "gsm8k": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl",
        "license": "MIT",
        "citation": "Cobbe et al. 2021, Training Verifiers to Solve Math Word Problems",
    },
    "mbpp": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/"
        "mbpp/sanitized-mbpp.json",
        "license": "CC BY 4.0",
        "citation": "Austin et al. 2021, Program Synthesis with Large Language Models",
    },
    "spider": {
        "url": "https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J "
        "(linked from https://yale-lily.github.io/spider)",
        "license": "CC BY-SA 4.0",
        "citation": "Yu et al. 2018, Spider: A Large-Scale Human-Labeled Dataset for Complex "
        "and Cross-Domain Semantic Parsing and Text-to-SQL Task",
    },
}

FILTERS = [
    "every kept task's own gold answer must be graded correct by its grader "
    "(MBPP gold code passes its asserts in the sandbox; Spider gold SQL runs read-only)",
    "spider: the gold query must return at least one row with a non-NULL value "
    "within the grading timeout, so an empty result cannot match by accident",
]


class ManifestError(Exception):
    pass


@dataclass(frozen=True)
class SplitTask:
    task: Task
    split: Split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_splits(ids: Sequence[str], rng: random.Random) -> dict[str, Split]:
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * FRACTIONS["train"])
    n_val = round(len(shuffled) * FRACTIONS["val"])
    out: dict[str, Split] = {}
    for i, task_id in enumerate(shuffled):
        out[task_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


async def select(
    candidates: Sequence[Task],
    count: int,
    rng: random.Random,
    accept: Callable[[Task], Awaitable[str | None]],
    on_reject: Callable[[Task, str], None] | None = None,
) -> list[Task]:
    """Walk the candidates in a seeded random order; keep the first `count` that pass."""
    order = list(candidates)
    rng.shuffle(order)
    kept: list[Task] = []
    for task in order:
        if len(kept) == count:
            break
        reason = await accept(task)
        if reason is None:
            kept.append(task)
        elif on_reject is not None:
            on_reject(task, reason)
    if len(kept) < count:
        raise ManifestError(
            f"only {len(kept)} of {len(candidates)} candidates passed; need {count}"
        )
    return kept


def build_manifest(
    seed: int, per_dataset: int, chosen: dict[Dataset, list[Task]], raw_dir: Path, sql_dir: Path
) -> dict[str, Any]:
    tasks: list[dict[str, str]] = []
    for dataset in DATASETS:
        ids = [t.id for t in chosen[dataset]]
        splits = assign_splits(ids, random.Random(f"{seed}:{dataset}"))
        tasks += [
            {"id": i, "dataset": dataset, "split": splits[i]} for i in sorted(ids, key=sort_key)
        ]
    spider_dbs = sorted({t.sql_db for t in chosen["spider"] if t.sql_db})
    return {
        "version": 1,
        "seed": seed,
        "per_dataset": per_dataset,
        "fractions": FRACTIONS,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            d: {**SOURCES[d], "file": RAW_FILES[d], "sha256": sha256_file(raw_dir / RAW_FILES[d])}
            for d in DATASETS
        },
        "sql_databases": {db: sha256_file(sql_dir / db) for db in spider_dbs},
        "filters": FILTERS,
        "tasks": tasks,
    }


def sort_key(task_id: str) -> tuple[str, int]:
    dataset, _, number = task_id.partition(":")
    return dataset, int(number)


def save_manifest(
    manifest: dict[str, Any], path: Path = MANIFEST_PATH, force: bool = False
) -> None:
    if path.exists() and not force:
        raise ManifestError(
            f"{path} already exists. The split is made once; regenerating it would leak "
            "test tasks into training. Pass force only if you mean to start a new split version."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_split_tasks(
    raw_dir: Path, sql_dir: Path, manifest: dict[str, Any] | None = None
) -> list[SplitTask]:
    """Rebuild the manifest's tasks from the raw files, refusing any file whose hash changed."""
    manifest = manifest or load_manifest()
    for dataset, source in manifest["sources"].items():
        actual = sha256_file(raw_dir / source["file"])
        if actual != source["sha256"]:
            raise ManifestError(f"{dataset}: {source['file']} sha256 {actual} != manifest")
    for db, digest in manifest["sql_databases"].items():
        if sha256_file(sql_dir / db) != digest:
            raise ManifestError(f"database {db} differs from the manifest")
    wanted = {t["id"]: t["split"] for t in manifest["tasks"]}
    out: list[SplitTask] = []
    for dataset in (d for d in DATASETS if d in manifest["sources"]):
        for task in load_all(raw_dir, dataset):
            if task.id in wanted:
                out.append(SplitTask(task, wanted.pop(task.id)))
    if wanted:
        raise ManifestError(f"manifest tasks missing from raw files: {sorted(wanted)[:5]}")
    return out


def install_spider_databases(raw_dir: Path, sql_dir: Path) -> list[str]:
    """Copy Spider dev databases to SQL_DATA_DIR/spider/<db>.sqlite (where tasks look)."""
    target = sql_dir / "spider"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for db_dir in sorted((raw_dir / "spider" / "database").iterdir()):
        src = db_dir / f"{db_dir.name}.sqlite"
        dst = target / src.name
        if src.is_file() and (not dst.exists() or sha256_file(dst) != sha256_file(src)):
            shutil.copyfile(src, dst)
            copied.append(src.name)
    return copied
