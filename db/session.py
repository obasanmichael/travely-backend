from __future__ import annotations

from typing import AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db.url import to_asyncpg_url

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def init_engine(database_url: str) -> AsyncEngine:
    global _engine, _sessionmaker
    _engine = create_async_engine(
        to_asyncpg_url(database_url),
        pool_size=5,
        max_overflow=5,
        # Neon suspends idle computes; pre-ping drops connections that died with it.
        pool_pre_ping=True,
        pool_recycle=300,
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


def engine_configured() -> bool:
    return _engine is not None


async def ping() -> bool:
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def get_session() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("Database engine is not initialised")
    async with _sessionmaker() as session:
        yield session
