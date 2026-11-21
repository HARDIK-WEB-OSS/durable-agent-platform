import json
import random
import sqlite3
from fractions import Fraction
from pathlib import Path

import pytest

from dap.evals.graders import (
    GradingContext,
    grade,
    grade_gsm8k,
    grade_spider_sync,
    parse_number,
    same_results,
    strip_code,
)
from dap.evals.loaders import gsm8k_tasks, mbpp_tasks, spider_tasks
from dap.evals.split import (
    ManifestError,
    assign_splits,
    load_split_tasks,
    save_manifest,
    select,
    sha256_file,
)

# ---- loaders -------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_gsm8k_loader_reads_answer_after_hashes(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "g.jsonl",
        [{"question": "How many?", "answer": "work...\n#### 1,234"}],
    )
    [task] = gsm8k_tasks(path)
    assert (task.id, task.reference) == ("gsm8k:0", {"answer": "1234"})
    assert task.prompt.startswith("How many?") and "number only" in task.prompt
    assert task.tools == ("calculator", "python_exec")


def test_mbpp_loader_shows_first_test_and_keeps_all(tmp_path: Path) -> None:
    record = {
        "task_id": 7,
        "prompt": "Write add.",
        "code": "def add(a, b): return a + b",
        "test_imports": [],
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    [task] = mbpp_tasks(path)
    assert task.id == "mbpp:7" and "assert add(1, 2) == 3" in task.prompt
    assert "assert add(0, 0) == 0" not in task.prompt
    assert task.reference["tests"] == record["test_list"]


def test_spider_loader_marks_ordered_queries(tmp_path: Path) -> None:
    rows = [
        {"db_id": "shop", "question": "Q1", "query": "SELECT name FROM t ORDER BY name"},
        {"db_id": "shop", "question": "Q2", "query": "SELECT count(*) FROM t"},
    ]
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    first, second = spider_tasks(path)
    assert (first.reference["ordered"], second.reference["ordered"]) == (True, False)
    assert first.sql_db == "spider/shop.sqlite" and first.tools == ("sql_query",)


# ---- gsm8k grader --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("18", Fraction(18)),
        ("$1,234.50", Fraction("1234.5")),
        ("The answer is 7 apples", Fraction(7)),
        ("from 3 to 12", Fraction(12)),
        ("-4", Fraction(-4)),
        ("0.5", Fraction(1, 2)),
        ("no digits", None),
    ],
)
def test_parse_number_takes_the_last_number(text: str, value: Fraction | None) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize(
    ("answer", "correct"),
    [("18", True), ("18.0", True), ("$18", True), ("17", False), ("eighteen", False)],
)
def test_gsm8k_grade_is_exact(answer: str, correct: bool) -> None:
    assert grade_gsm8k(answer, {"answer": "18"}).correct is correct


def test_strip_code_takes_the_fenced_block() -> None:
    assert strip_code("Here:\n```python\ndef f():\n    return 1\n```\nDone") == (
        "def f():\n    return 1"
    )
    assert strip_code("  SELECT 1  ") == "SELECT 1"


# ---- spider grader -------------------------------------------------------------------------


def test_same_results_rules() -> None:
    gold = [(1, "a"), (2, "b")]
    assert same_results(gold, [(2, "b"), (1, "a")], ordered=False)
    assert not same_results(gold, [(2, "b"), (1, "a")], ordered=True)
    assert same_results(gold, [("a", 1), ("b", 2)], ordered=True)  # column order may differ
    assert not same_results(gold, [(1, "a")], ordered=False)
    assert not same_results(gold, [(1, "a"), (1, "a")], ordered=False)  # multiset, not set
    assert same_results([(0.1 + 0.2,)], [(0.3,)], ordered=False)
    assert same_results([(2,)], [(2.0,)], ordered=False)


@pytest.fixture
def spider_dir(tmp_path: Path) -> Path:
    (tmp_path / "spider").mkdir()
    conn = sqlite3.connect(tmp_path / "spider" / "shop.sqlite")
    conn.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT, price REAL);"
        "INSERT INTO t VALUES (1, 'pen', 2.0), (2, 'book', 12.5), (3, 'cup', 4.0);"
    )
    conn.commit()
    conn.close()
    return tmp_path


REF = {"gold_sql": "SELECT name FROM t WHERE price > 3", "db_id": "shop", "ordered": False}


@pytest.mark.parametrize(
    ("answer", "correct"),
    [
        ("SELECT name FROM t WHERE price > 3", True),
        ("select name from t where price >= 4 order by id desc;", True),
        ("```sql\nSELECT name FROM t WHERE NOT price < 3.5\n```", True),
        ("SELECT name FROM t", False),
        ("SELECT id FROM t WHERE price > 3", False),
        ("SELECT nope FROM t", False),
        ("DELETE FROM t", False),
        ("", False),
    ],
)
def test_spider_grade_by_execution(spider_dir: Path, answer: str, correct: bool) -> None:
    ctx = GradingContext(None, spider_dir)
    assert grade_spider_sync(answer, REF, ctx).correct is correct
    assert grade_spider_sync(REF["gold_sql"], REF, ctx).correct  # DELETE changed nothing


async def test_missing_answer_is_incorrect(spider_dir: Path) -> None:
    [task] = gsm8k_tasks(
        write_jsonl(spider_dir / "g.jsonl", [{"question": "q", "answer": "#### 1"}])
    )
    ctx = GradingContext(None, spider_dir)
    assert not (await grade(task, None, ctx)).correct
    assert not (await grade(task, "   ", ctx)).correct


# ---- split ---------------------------------------------------------------------------------


def test_assign_splits_is_seeded_and_60_20_20() -> None:
    ids = [f"x:{i}" for i in range(100)]
    first = assign_splits(ids, random.Random("s"))
    assert first == assign_splits(ids, random.Random("s"))
    assert first != assign_splits(ids, random.Random("other"))
    counts = {s: list(first.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 60, "val": 20, "test": 20}


async def test_select_skips_rejects_and_is_deterministic(tmp_path: Path) -> None:
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(20)]
    tasks = list(gsm8k_tasks(write_jsonl(tmp_path / "g.jsonl", rows)))

    async def odd_only(task: object) -> str | None:
        return None if int(task.reference["answer"]) % 2 else "even"  # type: ignore[attr-defined]

    first = await select(tasks, 5, random.Random(1), odd_only)
    again = await select(tasks, 5, random.Random(1), odd_only)
    assert [t.id for t in first] == [t.id for t in again]
    assert all(int(t.reference["answer"]) % 2 for t in first)
    with pytest.raises(ManifestError, match="only 10"):
        await select(tasks, 11, random.Random(1), odd_only)


def test_manifest_is_never_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    save_manifest({"tasks": []}, path)
    with pytest.raises(ManifestError, match="made once"):
        save_manifest({"tasks": [1]}, path)
    save_manifest({"tasks": [1]}, path, force=True)
    assert json.loads(path.read_text())["tasks"] == [1]


def test_loading_refuses_changed_raw_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "gsm8k").mkdir(parents=True)
    path = write_jsonl(raw / "gsm8k" / "test.jsonl", [{"question": "q", "answer": "#### 1"}])
    manifest = {
        "sources": {"gsm8k": {"file": "gsm8k/test.jsonl", "sha256": sha256_file(path)}},
        "sql_databases": {},
        "tasks": [{"id": "gsm8k:0", "dataset": "gsm8k", "split": "train"}],
    }
    [loaded] = load_split_tasks(raw, tmp_path, manifest)
    assert (loaded.task.id, loaded.split) == ("gsm8k:0", "train")
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="sha256"):
        load_split_tasks(raw, tmp_path, manifest)
