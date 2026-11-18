import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.integration


async def test_postgres_is_reachable(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        version = (await conn.execute(text("SHOW server_version"))).scalar_one()
    assert str(version).startswith("16")


async def test_pgvector_is_available(engine: AsyncEngine) -> None:
    query = text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    async with engine.connect() as conn:
        found = (await conn.execute(query)).scalar_one_or_none()
    assert found == 1
