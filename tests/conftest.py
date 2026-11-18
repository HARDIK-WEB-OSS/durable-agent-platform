from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from dap.config import Settings, get_settings
from dap.db.session import make_engine


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    eng = make_engine(settings)
    yield eng
    await eng.dispose()
