import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dap.config import Settings, get_settings
from dap.db.models import Base
from dap.db.session import make_engine, make_sessionmaker

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")  # test modules import litellm directly


@pytest.fixture(scope="session")
def settings() -> Settings:
    base = get_settings()
    return base.model_copy(update={"postgres_db": f"{base.postgres_db}_test"})


async def ensure_database(settings: Settings) -> None:
    admin = create_async_engine(
        settings.model_copy(update={"postgres_db": "postgres"}).database_url,
        isolation_level="AUTOCOMMIT",
    )
    async with admin.connect() as conn:
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": settings.postgres_db},
        )
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{settings.postgres_db}"'))
    await admin.dispose()


@pytest.fixture(scope="session")
def migrated_db(settings: Settings) -> None:
    asyncio.run(ensure_database(settings))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = settings.database_url
    command.upgrade(cfg, "head")


@pytest.fixture
async def engine(settings: Settings, migrated_db: None) -> AsyncIterator[AsyncEngine]:
    eng = make_engine(settings)
    yield eng
    await eng.dispose()


@pytest.fixture
async def clean_db(engine: AsyncEngine) -> None:
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def sessionmaker(engine: AsyncEngine, clean_db: None) -> async_sessionmaker[AsyncSession]:
    return make_sessionmaker(engine)


@pytest.fixture
async def session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as s:
        yield s
