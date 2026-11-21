"""Turn the raw benchmark files into agent tasks.

Raw files live under data/raw (git-ignored, downloaded once; see split.SOURCES):
    data/raw/gsm8k/test.jsonl            openai/grade-school-math, MIT
    data/raw/mbpp/sanitized-mbpp.json    google-research/mbpp, CC BY 4.0
    data/raw/spider/dev.json + database/ Spider 1.0 dev, CC BY-SA 4.0

Task ids are stable: "gsm8k:<line index>", "mbpp:<task_id>", "spider:<dev index>".
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dap.db.models import JsonDict

Dataset = Literal["gsm8k", "mbpp", "spider"]
DATASETS: tuple[Dataset, ...] = ("gsm8k", "mbpp", "spider")

GSM8K_SUFFIX = "\n\nGive the final answer as a number only, with no units or words."
MBPP_SUFFIX = (
    "\n\nWrite a complete Python solution. It must pass tests like:\n{test}\n\n"
    "Your final answer must be only the Python code (any imports plus the function "
    "definition), with no explanation and no test code."
)
SPIDER_SUFFIX = (
    "\n\nThe question is about the SQLite database you can query with sql_query. Inspect the "
    "schema first (SELECT name, sql FROM sqlite_master). Your final answer must be one SQLite "
    "SELECT query that answers the question, not the result of the query."
)
GSM8K_ANSWER = re.compile(r"####\s*(.+?)\s*$")
TOP_LEVEL_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


@dataclass(frozen=True)
class Task:
    id: str
    dataset: Dataset
    prompt: str
    tools: tuple[str, ...]
    reference: JsonDict = field(repr=False)
    sql_db: str | None = None  # file name inside SQL_DATA_DIR


def gsm8k_tasks(path: Path) -> Iterator[Task]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            match = GSM8K_ANSWER.search(record["answer"])
            if match is None:
                raise ValueError(f"gsm8k line {i}: no '#### answer'")
            yield Task(
                id=f"gsm8k:{i}",
                dataset="gsm8k",
                prompt=record["question"].strip() + GSM8K_SUFFIX,
                tools=("calculator", "python_exec"),
                reference={"answer": match.group(1).replace(",", "")},
            )


def mbpp_tasks(path: Path) -> Iterator[Task]:
    for record in json.loads(path.read_text(encoding="utf-8")):
        tests: list[str] = record["test_list"]
        yield Task(
            id=f"mbpp:{record['task_id']}",
            dataset="mbpp",
            prompt=record["prompt"].strip() + MBPP_SUFFIX.format(test=tests[0]),
            tools=("python_exec",),
            reference={
                "test_imports": record["test_imports"],
                "tests": tests,
                "gold_code": record["code"],
            },
        )


def spider_db_file(db_id: str) -> str:
    return f"spider/{db_id}.sqlite"


def spider_tasks(path: Path) -> Iterator[Task]:
    for i, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        gold: str = record["query"].strip()
        yield Task(
            id=f"spider:{i}",
            dataset="spider",
            prompt=record["question"].strip() + SPIDER_SUFFIX,
            tools=("sql_query",),
            reference={
                "gold_sql": gold,
                "db_id": record["db_id"],
                "ordered": bool(TOP_LEVEL_ORDER_BY.search(gold)),
            },
            sql_db=spider_db_file(record["db_id"]),
        )


RAW_FILES: dict[Dataset, str] = {
    "gsm8k": "gsm8k/test.jsonl",
    "mbpp": "mbpp/sanitized-mbpp.json",
    "spider": "spider/dev.json",
}


def load_all(raw_dir: Path, dataset: Dataset) -> list[Task]:
    path = raw_dir / RAW_FILES[dataset]
    loader = {"gsm8k": gsm8k_tasks, "mbpp": mbpp_tasks, "spider": spider_tasks}[dataset]
    return list(loader(path))
