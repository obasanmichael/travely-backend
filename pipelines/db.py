from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.models import Source
from db.url import to_asyncpg_url


@asynccontextmanager
async def session_scope(database_url: Optional[str] = None) -> AsyncIterator[AsyncSession]:
    url = database_url or os.environ.get("DATABASE_URL", "")
    engine = create_async_engine(to_asyncpg_url(url))
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            async with session.begin():
                yield session
    finally:
        await engine.dispose()


async def get_or_create_source(
    session: AsyncSession,
    *,
    kind: str,
    url: str,
    title: str,
    licence: Optional[str],
) -> Source:
    """One row per (kind, url): re-running a job reuses the source instead of duplicating it."""
    existing = await session.scalar(select(Source).where(Source.kind == kind, Source.url == url))
    now = datetime.now(timezone.utc)
    if existing is not None:
        existing.retrieved_at = now
        existing.title = title
        existing.licence = licence
        return existing
    source = Source(kind=kind, url=url, title=title, licence=licence, retrieved_at=now)
    session.add(source)
    await session.flush()
    return source
