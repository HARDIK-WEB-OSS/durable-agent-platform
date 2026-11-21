import sqlite3
from pathlib import Path

import pytest

from dap.agent.tools import ToolBox, UnknownToolError
from dap.agent.tools.base import ToolError, truncate
from dap.agent.tools.calculator import CalculatorTool, evaluate
from dap.agent.tools.python_exec import DockerSandbox, PythonExecTool
from dap.agent.tools.sql_query import SqlQueryTool, resolve_database

IMAGE = "python:3.11-slim@sha256:" + "0" * 64


# ---- calculator ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(12 / 3) * 2", "8"),
        ("2 ** 10", "1024"),
        ("2^10", "1024"),
        ("7 // 2 + 7 % 2", "4"),
        ("-3 + abs(-5)", "2"),
        ("sqrt(16) + round(2.6)", "7"),
        ("max(1, 9, 4) - min(3, 2)", "7"),
        ("0.1 + 0.2", "0.30000000000000004"),
    ],
)
async def test_calculator_evaluates_arithmetic(expr: str, expected: str) -> None:
    assert await CalculatorTool().run(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "(1).__class__",
        "[x for x in range(3)]",
        "lambda: 1",
        "a + 1",
        "sqrt(x=4)",
        "True + 1",
        "'a' * 3",
    ],
)
def test_calculator_rejects_anything_but_arithmetic(expr: str) -> None:
    with pytest.raises(ToolError):
        evaluate(expr)


@pytest.mark.parametrize("expr", ["9 ** 9 ** 9", "10 ** 5000", "2 ** 1001", "x" * 600])
def test_calculator_refuses_huge_work(expr: str) -> None:
    with pytest.raises(ToolError):
        evaluate(expr)


@pytest.mark.parametrize("expr", ["1 / 0", "sqrt(-1)", "log(0)", "2 +"])
def test_calculator_errors_are_tool_errors(expr: str) -> None:
    with pytest.raises(ToolError):
        evaluate(expr)


# ---- sql_query -----------------------------------------------------------------------------


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        INSERT INTO items (name, price) VALUES ('pen', 2.0), ('book', 12.5), ('cup', NULL);
        """
    )
    conn.commit()
    conn.close()
    return path


async def test_sql_select_returns_header_rows_and_count(shop_db: Path) -> None:
    out = await SqlQueryTool(shop_db).run("SELECT name, price FROM items ORDER BY id")
    assert out.splitlines() == [
        "name | price",
        "pen | 2.0",
        "book | 12.5",
        "cup | NULL",
        "(3 rows)",
    ]


async def test_sql_row_limit_is_reported(shop_db: Path) -> None:
    out = await SqlQueryTool(shop_db, max_rows=2).run("SELECT name FROM items")
    assert out.splitlines()[-1] == "(2 rows; more rows not shown, limit 2)"


async def test_sql_schema_is_readable(shop_db: Path) -> None:
    out = await SqlQueryTool(shop_db).run("SELECT name FROM sqlite_master WHERE type='table'")
    assert "items" in out


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO items (name) VALUES ('hack')",
        "UPDATE items SET price = 0",
        "DELETE FROM items",
        "DROP TABLE items",
        "CREATE TABLE t (x)",
        "PRAGMA query_only = OFF",
        "ATTACH DATABASE 'evil.sqlite' AS evil",
        "SELECT 1; DELETE FROM items",
    ],
)
async def test_sql_cannot_change_anything(shop_db: Path, sql: str) -> None:
    with pytest.raises(ToolError):
        await SqlQueryTool(shop_db).run(sql)
    out = await SqlQueryTool(shop_db).run("SELECT count(*) AS n FROM items")
    assert out.splitlines()[1] == "3"
    assert not (shop_db.parent / "evil.sqlite").exists()


async def test_sql_slow_query_is_aborted(shop_db: Path) -> None:
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"
    with pytest.raises(ToolError, match="longer than"):
        await SqlQueryTool(shop_db, timeout_s=0.3).run(endless)


def test_database_must_be_inside_data_dir(shop_db: Path) -> None:
    assert resolve_database(shop_db.parent, "shop.sqlite") == shop_db.resolve()
    with pytest.raises(ValueError, match="outside"):
        resolve_database(shop_db.parent / "sub", "../shop.sqlite")
    with pytest.raises(FileNotFoundError):
        resolve_database(shop_db.parent, "missing.sqlite")


# ---- toolbox -------------------------------------------------------------------------------


def test_toolbox_offers_only_what_the_host_supports(shop_db: Path) -> None:
    box = ToolBox(sql_data_dir=shop_db.parent)
    assert list(box.for_run({})) == ["calculator"]
    with_db = box.for_run({"sql_db": "shop.sqlite"})
    assert list(with_db) == ["calculator", "sql_query"]
    sandboxed = ToolBox(sandbox=DockerSandbox(IMAGE)).for_run({"tools": ["python_exec"]})
    assert list(sandboxed) == ["python_exec"]


def test_toolbox_rejects_unknown_tool_names() -> None:
    with pytest.raises(UnknownToolError):
        ToolBox().for_run({"tools": ["calculator", "rm_rf"]})


def test_truncate_marks_cut_text() -> None:
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdefgh", 3) == "abc\n... [truncated 5 characters]"


# ---- python_exec command line (the live sandbox is tested in tests/integration) -----------


def test_sandbox_requires_image_pinned_by_digest() -> None:
    with pytest.raises(ValueError, match="digest"):
        DockerSandbox("python:3.11-slim")


def test_sandbox_command_has_every_hardening_flag() -> None:
    cmd = DockerSandbox(IMAGE).run_command("box")
    joined = " ".join(cmd)
    for flag in [
        "--rm",
        "--network none",
        "--read-only",
        "--tmpfs /tmp:rw",
        "--memory 256m",
        "--memory-swap 256m",
        "--cpus 1",
        "--pids-limit 64",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--name box",
    ]:
        assert flag in joined, flag
    assert cmd[-4:] == [IMAGE, "python", "-I", "-"]
    assert "--volume" not in cmd and "-v" not in cmd


def test_python_exec_description_states_limits() -> None:
    text = PythonExecTool(DockerSandbox(IMAGE, timeout_s=7)).description
    assert "no network" in text and "7 s" in text
