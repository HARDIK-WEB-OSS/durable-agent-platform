"""Deterministic graders. No LLM judges anything.

gsm8k   exact numeric match of the last number in the answer (commas and $ ignored)
mbpp    the answer's code plus the dataset's asserts run in the python_exec Docker sandbox;
        correct only if every assert passes and a random per-run marker is printed afterwards,
        so code that exits early or prints a fake "passed" line still fails
spider  execution match: predicted and gold SQL both run read-only on the task's database;
        results compared as multisets (as lists when the gold query has ORDER BY), allowing a
        different column order, floats compared to 6 decimals
"""

import asyncio
import itertools
import re
import secrets
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from dap.agent.tools.base import ToolError
from dap.agent.tools.python_exec import DockerSandbox
from dap.agent.tools.sql_query import Row, read_only_query, resolve_database
from dap.db.models import JsonDict
from dap.evals.loaders import Task, spider_db_file

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)```", re.DOTALL)
MAX_GRADE_ROWS = 10_000
MAX_PERMUTED_COLUMNS = 6
# MBPP asserts run in milliseconds; this limit mostly covers Docker container start-up, which
# can take several seconds on a loaded machine. Grading must not fail a correct answer for it.
GRADING_SANDBOX_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Grade:
    correct: bool
    detail: str


@dataclass(frozen=True)
class GradingContext:
    sandbox: DockerSandbox | None
    sql_data_dir: Path
    sql_timeout_s: float = 10.0


def parse_number(text: str) -> Fraction | None:
    found = NUMBER.findall(text.replace("$", ""))
    if not found:
        return None
    try:
        return Fraction(found[-1].replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def strip_code(text: str) -> str:
    match = CODE_FENCE.search(text)
    return (match.group(1) if match else text).strip()


def grade_gsm8k(answer: str, reference: JsonDict) -> Grade:
    expected = parse_number(reference["answer"])
    got = parse_number(answer)
    if expected is None:
        raise ValueError(f"reference {reference['answer']!r} is not a number")
    if got is None:
        return Grade(False, f"no number in answer {answer[:80]!r}")
    if got == expected:
        return Grade(True, f"{got} == {expected}")
    return Grade(False, f"{got} != {expected}")


async def grade_mbpp(answer: str, reference: JsonDict, sandbox: DockerSandbox | None) -> Grade:
    if sandbox is None:
        raise RuntimeError("grading MBPP needs the python_exec Docker sandbox")
    marker = f"ALL_TESTS_PASSED_{secrets.token_hex(8)}"
    script = "\n".join(
        [
            *reference["test_imports"],
            strip_code(answer),
            "",
            *reference["tests"],
            f"print({marker!r})",
        ]
    )
    result = await sandbox.run(script + "\n")
    if result.timed_out:
        return Grade(False, f"tests timed out after {sandbox.timeout_s:g} s")
    if result.exit_code == 0 and marker in result.output:
        return Grade(True, f"all {len(reference['tests'])} tests passed")
    tail = " | ".join(result.output.strip().splitlines()[-3:])
    return Grade(False, f"exit {result.exit_code}: {tail[:300]}")


def normalize(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.hex()
    return value


def same_results(gold: Sequence[Row], predicted: Sequence[Row], ordered: bool) -> bool:
    gold_n = [tuple(normalize(v) for v in row) for row in gold]
    pred_n = [tuple(normalize(v) for v in row) for row in predicted]
    if len(gold_n) != len(pred_n):
        return False
    width = len(gold_n[0]) if gold_n else 0
    if any(len(r) != width for r in pred_n):
        return False
    orders = [tuple(range(width))]
    if width <= MAX_PERMUTED_COLUMNS:
        orders = list(itertools.permutations(range(width)))
    for order in orders:
        permuted = [tuple(row[i] for i in order) for row in pred_n]
        if (permuted == gold_n) if ordered else (Counter(permuted) == Counter(gold_n)):
            return True
    return False


def grade_spider_sync(answer: str, reference: JsonDict, ctx: GradingContext) -> Grade:
    sql = strip_code(answer).strip().rstrip(";").strip()
    if not sql:
        return Grade(False, "empty query")
    db = resolve_database(ctx.sql_data_dir, spider_db_file(reference["db_id"]))
    _, gold = read_only_query(db, reference["gold_sql"], ctx.sql_timeout_s, MAX_GRADE_ROWS)
    try:
        _, predicted = read_only_query(db, sql, ctx.sql_timeout_s, MAX_GRADE_ROWS)
    except ToolError as exc:
        return Grade(False, f"query failed: {exc}")
    if same_results(gold, predicted, reference["ordered"]):
        return Grade(True, f"execution match ({len(gold)} rows)")
    return Grade(False, f"result mismatch: {len(predicted)} rows vs {len(gold)} expected")


async def grade(task: Task, answer: str | None, ctx: GradingContext) -> Grade:
    if answer is None or not answer.strip():
        return Grade(False, "no answer")
    if task.dataset == "gsm8k":
        return grade_gsm8k(answer, task.reference)
    if task.dataset == "mbpp":
        return await grade_mbpp(answer, task.reference, ctx.sandbox)
    return await asyncio.to_thread(grade_spider_sync, answer, task.reference, ctx)


def gold_answer(task: Task) -> str:
    """The dataset's own answer, used to prove each grader accepts it before a task is kept."""
    if task.dataset == "gsm8k":
        return str(task.reference["answer"])
    if task.dataset == "mbpp":
        return str(task.reference["gold_code"])
    return str(task.reference["gold_sql"])
