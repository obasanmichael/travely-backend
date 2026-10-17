from __future__ import annotations

import os

import pytest

# Ensure tests never pick up a developer .env file from the repo root.
os.environ["AUTH_DISABLED"] = "true"
os.environ["ENV"] = "development"


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _load_catalog():
    from services.data_cache import load_catalog

    load_catalog()


TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
requires_db = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="session")
def migrated_db():
    """A test database migrated to head once per session."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    return config


@pytest.fixture
def empty_db(migrated_db):
    """Truncate every application table so a test only sees the rows it creates."""
    import asyncio

    from sqlalchemy import text

    from pipelines.db import session_scope

    async def _truncate():
        async with session_scope(TEST_DATABASE_URL) as session:
            tables = (await session.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename NOT IN ('alembic_version', 'spatial_ref_sys')"
            ))).scalars().all()
            await session.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))

    asyncio.run(_truncate())
    return migrated_db
