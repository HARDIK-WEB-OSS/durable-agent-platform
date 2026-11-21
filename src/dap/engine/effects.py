"""Checkpointing for external effects: LLM calls, tool runs, anything slow or outside Postgres.

`run_once` (idempotency.py) holds a DB transaction open while its effect runs, which suits fast
writes to our own tables: the effect and its key commit together, so the effect happens exactly
once. An LLM call or a sandboxed tool run takes 1-60 s and cannot join our transaction, so here
(review F02):

    1. read the key; if a result is stored, return it without running the effect
    2. run the effect with NO transaction or connection held
    3. INSERT the key ... ON CONFLICT DO NOTHING
    4. return the stored row, so concurrent duplicates all agree on the first result

Guarantee: at-least-once execution, recorded once. A crash between 2 and 3 runs the effect
again on retry; a crash after 3 does not.
"""

import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.db.models import IdempotencyKey, JsonDict
from dap.engine.idempotency import load_result

ExternalEffect = Callable[[], Awaitable[JsonDict]]


async def record_once(
    sessionmaker: async_sessionmaker[AsyncSession],
    key: str,
    step_id: uuid.UUID,
    effect: ExternalEffect,
) -> JsonDict:
    stored = await load_result(sessionmaker, key)
    if stored is not None:
        return stored
    result = await effect()
    async with sessionmaker.begin() as session:
        await session.execute(
            insert(IdempotencyKey)
            .values(key=key, step_id=step_id, result=result)
            .on_conflict_do_nothing(index_elements=[IdempotencyKey.key])
        )
    stored = await load_result(sessionmaker, key)
    if stored is None:
        raise RuntimeError(f"idempotency key {key!r} vanished right after insert")
    return stored
