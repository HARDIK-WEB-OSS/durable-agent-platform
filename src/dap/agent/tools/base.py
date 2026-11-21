from typing import Protocol


class ToolError(Exception):
    """The tool ran but the input was bad or the work failed; the model sees the message.

    Anything else a tool raises (Docker down, DB unreachable) is an infrastructure fault:
    the step fails and the engine retries it with backoff.
    """


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str:
        """One line for the system prompt: what it does and what tool_input must be."""
        ...

    async def run(self, tool_input: str) -> str: ...


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} characters]"
