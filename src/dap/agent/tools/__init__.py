import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.agent.tools.base import Tool, ToolError
from dap.agent.tools.calculator import CalculatorTool
from dap.agent.tools.python_exec import DockerSandbox, PythonExecTool
from dap.agent.tools.search_docs import SearchDocsTool
from dap.agent.tools.sql_query import SqlQueryTool, resolve_database
from dap.db.models import JsonDict
from dap.llm.embeddings import Embedder

__all__ = ["TOOL_NAMES", "Tool", "ToolBox", "ToolError", "UnknownToolError"]

log = logging.getLogger(__name__)

TOOL_NAMES = ("calculator", "python_exec", "sql_query", "search_docs")


class UnknownToolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolBox:
    """Builds the tools one run may use, from its input and what this host supports.

    Run input keys: "tools" (list of names; default: every tool this host can offer) and
    "sql_db" (a file name inside sql_data_dir; sql_query is offered only when it is set).
    """

    sessionmaker: async_sessionmaker[AsyncSession] | None = None
    embedder: Embedder | None = None
    sandbox: DockerSandbox | None = None
    sql_data_dir: Path = Path("data/sql")
    sql_timeout_s: float = 5.0
    search_top_k: int = 4

    def for_run(self, run_input: JsonDict) -> dict[str, Tool]:
        wanted: list[str] = run_input.get("tools") or list(TOOL_NAMES)
        unknown = sorted(set(wanted) - set(TOOL_NAMES))
        if unknown:
            raise UnknownToolError(f"unknown tools {unknown}; known: {list(TOOL_NAMES)}")
        tools: dict[str, Tool] = {}
        for name in wanted:
            tool = self.build(name, run_input)
            if tool is not None:
                tools[name] = tool
        return tools

    def build(self, name: str, run_input: JsonDict) -> Tool | None:
        if name == "calculator":
            return CalculatorTool()
        if name == "python_exec":
            if self.sandbox is None:
                log.info("python_exec requested but disabled on this host")
                return None
            return PythonExecTool(self.sandbox)
        if name == "sql_query":
            db = run_input.get("sql_db")
            if not db:
                return None
            return SqlQueryTool(
                resolve_database(self.sql_data_dir, db), timeout_s=self.sql_timeout_s
            )
        if self.sessionmaker is None or self.embedder is None:
            return None
        return SearchDocsTool(self.sessionmaker, self.embedder, self.search_top_k)
