from __future__ import annotations

import asyncio

from sqlalchemy import func, select

from db.models import TAG_CATEGORIES, Tag
from pipelines.db import session_scope
from pipelines.taxonomy import TAXONOMY, seed_taxonomy
from tests.conftest import TEST_DATABASE_URL, requires_db


def test_taxonomy_definitions_are_unique_and_valid():
    slugs = [t.slug for t in TAXONOMY]
    assert len(slugs) == len(set(slugs))
    assert all(t.category in TAG_CATEGORIES for t in TAXONOMY)


@requires_db
def test_seed_taxonomy_is_idempotent(migrated_db):
    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_taxonomy(session)
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_taxonomy(session)
            return await session.scalar(select(func.count()).select_from(Tag))

    assert asyncio.run(_run()) == len(TAXONOMY)
