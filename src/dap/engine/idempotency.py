import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.db.models import IdempotencyKey, JsonDict

Effect = Callable[[AsyncSession], Awaitable[JsonDict]]


async def run_once(
    sessionmaker: async_sessionmaker[AsyncSession],
    key: str,
    step_id: uuid.UUID,
    effect: Effect,
) -> JsonDict:
    try:
        return await execute_and_record(sessionmaker, key, step_id, effect)
    except IntegrityError:
        stored = await load_result(sessionmaker, key)
        if stored is None:
            raise
        return stored


async def execute_and_record(
    sessionmaker: async_sessionmaker[AsyncSession],
    key: str,
    step_id: uuid.UUID,
    effect: Effect,
) -> JsonDict:
    async with sessionmaker.begin() as session:
        stored = await session.get(IdempotencyKey, key)
        if stored is not None:
            return stored.result
        result = await effect(session)
        session.add(IdempotencyKey(key=key, step_id=step_id, result=result))
    return result


async def load_result(sessionmaker: async_sessionmaker[AsyncSession], key: str) -> JsonDict | None:
    async with sessionmaker() as session:
        stored = await session.get(IdempotencyKey, key)
        return None if stored is None else stored.result
