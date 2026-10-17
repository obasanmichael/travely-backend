"""Run against a disposable Postgres (TEST_DATABASE_URL); skipped when none is configured."""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from db.base import Base
from db.migration_filters import include_object
from db.url import to_asyncpg_url

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


def _alembic_config() -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    return config


@pytest.fixture(scope="module")
def migrated_db():
    config = _alembic_config()
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield config
    command.downgrade(config, "base")


def _run(coro):
    return asyncio.run(coro)


async def _execute(sql: str, params: dict | None = None):
    engine = create_async_engine(to_asyncpg_url(TEST_DATABASE_URL))
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(sql), params or {})
            return result.fetchall() if result.returns_rows else None
    finally:
        await engine.dispose()


def test_downgrade_then_upgrade_is_repeatable(migrated_db):
    command.downgrade(migrated_db, "base")
    command.upgrade(migrated_db, "head")
    rows = _run(_execute("SELECT to_regclass('public.destinations') IS NOT NULL"))
    assert rows[0][0] is True


def test_models_match_migrations(migrated_db):
    def _diff(sync_conn):
        context = MigrationContext.configure(
            sync_conn, opts={"include_object": include_object, "compare_type": True}
        )
        return compare_metadata(context, Base.metadata)

    async def _compare():
        engine = create_async_engine(to_asyncpg_url(TEST_DATABASE_URL))
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(_diff)
        finally:
            await engine.dispose()

    assert _run(_compare()) == []


async def _seed_place() -> int:
    await _execute(
        "INSERT INTO countries (code, name, currency) VALUES ('NG', 'Nigeria', 'NGN') "
        "ON CONFLICT DO NOTHING"
    )
    rows = await _execute(
        "INSERT INTO states (country_code, name, slug) VALUES ('NG', 'Ogun', :slug) RETURNING id",
        {"slug": f"ogun-{uuid.uuid4().hex[:6]}"},
    )
    return rows[0][0]


def test_unverified_destination_cannot_be_published(migrated_db):
    state_id = _run(_seed_place())
    with pytest.raises(IntegrityError, match="published_requires_verification"):
        _run(
            _execute(
                "INSERT INTO destinations (slug, name, country_code, state_id, location, status) "
                "VALUES (:slug, 'Olumo Rock', 'NG', :state_id, "
                "ST_GeogFromText('POINT(3.3489 7.1608)'), 'published')",
                {"slug": f"olumo-{uuid.uuid4().hex[:6]}", "state_id": state_id},
            )
        )


def test_duplicate_quiz_submission_is_rejected(migrated_db):
    uid = f"user-{uuid.uuid4().hex[:8]}"
    _run(_execute("INSERT INTO users (firebase_uid) VALUES (:uid)", {"uid": uid}))
    insert = (
        "INSERT INTO recommendation_runs (uid, idempotency_key, inputs, outputs, engine_version) "
        "VALUES (:uid, 'same-key', '{}', '{}', '3.0.0')"
    )
    _run(_execute(insert, {"uid": uid}))
    with pytest.raises(IntegrityError, match="uq_recommendation_runs_uid_idem"):
        _run(_execute(insert, {"uid": uid}))


def test_deleting_user_removes_their_runs(migrated_db):
    uid = f"user-{uuid.uuid4().hex[:8]}"
    _run(_execute("INSERT INTO users (firebase_uid) VALUES (:uid)", {"uid": uid}))
    _run(
        _execute(
            "INSERT INTO recommendation_runs (uid, idempotency_key, inputs, outputs, engine_version) "
            "VALUES (:uid, 'k', '{}', '{}', '3.0.0')",
            {"uid": uid},
        )
    )
    _run(_execute("DELETE FROM users WHERE firebase_uid = :uid", {"uid": uid}))
    rows = _run(_execute("SELECT count(*) FROM recommendation_runs WHERE uid = :uid", {"uid": uid}))
    assert rows[0][0] == 0
