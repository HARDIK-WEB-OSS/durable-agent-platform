"""The agent transcript is never stored as a whole; it is rebuilt from step outputs.

Each succeeded step of a run contributes one message: a think step its action (or a format
error), an act step its observation. Steps are durable, so the transcript survives crashes
and a retried step sees exactly the history that was committed before it.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dap.agent.prompts import (
    action_message,
    format_error_message,
    observation_message,
    system_prompt,
    task_message,
)
from dap.agent.tools.base import Tool
from dap.db.models import JsonDict, Step
from dap.llm.client import Message
from dap.states import StepStatus

THINK_KIND = "agent.think"
ACT_KIND = "agent.act"


@dataclass(frozen=True)
class Turn:
    kind: str
    output: JsonDict


async def load_history(session: AsyncSession, run_id: uuid.UUID, before_seq: int) -> list[Turn]:
    stmt = (
        select(Step.kind, Step.output)
        .where(
            Step.run_id == run_id,
            Step.seq < before_seq,
            Step.status == StepStatus.SUCCEEDED,
            Step.kind.in_((THINK_KIND, ACT_KIND)),
        )
        .order_by(Step.seq)
    )
    return [Turn(kind, output or {}) for kind, output in (await session.execute(stmt)).all()]


def build_messages(task: str, tools: dict[str, Tool], history: Sequence[Turn]) -> list[Message]:
    messages: list[Message] = [
        {"role": "system", "content": system_prompt(tools)},
        {"role": "user", "content": task_message(task)},
    ]
    for turn in history:
        out = turn.output
        if turn.kind == THINK_KIND and "action" in out:
            messages.append({"role": "assistant", "content": action_message(out["action"])})
        elif turn.kind == THINK_KIND and "format_error" in out:
            messages.append({"role": "assistant", "content": out.get("raw") or "(empty reply)"})
            messages.append({"role": "user", "content": format_error_message(out["format_error"])})
        elif turn.kind == ACT_KIND:
            content = observation_message(out["tool"], out["observation"])
            messages.append({"role": "user", "content": content})
    return messages
