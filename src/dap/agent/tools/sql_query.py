import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from dap.agent.tools.base import ToolError

PROGRESS_EVERY_OPS = 10_000
# Everything else (ATTACH, PRAGMA, writes, schema changes) is denied before it runs.
READ_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)


def allow_reads_only(action: int, *args: object) -> int:
    return sqlite3.SQLITE_OK if action in READ_ACTIONS else sqlite3.SQLITE_DENY


def resolve_database(data_dir: Path, name: str) -> Path:
    """Map a run's database name to a file inside data_dir; nothing outside it is reachable."""
    root = data_dir.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"database {name!r} is outside {root}")
    if not path.is_file():
        raise FileNotFoundError(f"database {path} does not exist")
    return path


@dataclass(frozen=True)
class SqlQueryTool:
    """One read-only SELECT against a SQLite file.

    Read-only four ways: the file is opened with mode=ro, PRAGMA query_only blocks writes, an
    authorizer denies everything except reads (so no ATTACH, PRAGMA or DDL), and only one
    statement is accepted per call. A progress handler aborts queries past timeout_s.
    """

    database: Path
    max_rows: int = 50
    timeout_s: float = 5.0
    name: str = "sql_query"

    @property
    def description(self) -> str:
        return (
            "Run one read-only SQLite SELECT on the task's database. tool_input: the SQL. "
            f"Returns at most {self.max_rows} rows. Start with "
            "SELECT name, sql FROM sqlite_master to see the schema."
        )

    async def run(self, tool_input: str) -> str:
        return await asyncio.to_thread(self.query, tool_input)

    def connect(self) -> sqlite3.Connection:
        uri = f"{self.database.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_s)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(allow_reads_only)
        return conn

    def query(self, sql: str) -> str:
        deadline = time.monotonic() + self.timeout_s
        conn = self.connect()
        try:
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), PROGRESS_EVERY_OPS)
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchmany(self.max_rows + 1)
            except sqlite3.OperationalError as exc:
                if time.monotonic() > deadline:
                    raise ToolError(f"query took longer than {self.timeout_s:g} s") from exc
                raise ToolError(f"SQLite error: {exc}") from exc
            except (sqlite3.Error, sqlite3.Warning) as exc:
                raise ToolError(f"SQLite error: {exc}") from exc
            columns = [d[0] for d in cursor.description or []]
        finally:
            conn.close()
        return format_rows(columns, rows, self.max_rows)


def format_rows(columns: list[str], rows: list[tuple[object, ...]], max_rows: int) -> str:
    if not columns:
        return "(statement returned no result set)"
    lines = [" | ".join(columns)]
    lines += [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows[:max_rows]]
    shown = min(len(rows), max_rows)
    footer = f"({shown} row{'s' if shown != 1 else ''}"
    footer += f"; more rows not shown, limit {max_rows})" if len(rows) > max_rows else ")"
    return "\n".join([*lines, footer])
