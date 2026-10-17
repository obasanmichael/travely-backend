"""Seed the shared interest taxonomy (defined in engine.taxonomy) into the tags table."""
from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Tag
from engine.taxonomy import TAXONOMY

__all__ = ["TAXONOMY", "seed_taxonomy"]


async def seed_taxonomy(session: AsyncSession) -> int:
    for tag in TAXONOMY:
        values = {
            "slug": tag.slug,
            "label": tag.label,
            "category": tag.category,
            "description": tag.description,
            "is_interest": tag.is_interest,
        }
        stmt = insert(Tag).values(**values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["slug"],
                set_={k: stmt.excluded[k] for k in values if k != "slug"},
            )
        )
    return len(TAXONOMY)
