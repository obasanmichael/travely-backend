"""Holds the engine snapshot the API ranks against and refreshes it on publish."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sqlalchemy import text

from db import session as db_session
from engine.snapshot import Snapshot, load_snapshot

logger = logging.getLogger(__name__)

_snapshot: Optional[Snapshot] = None


def current() -> Optional[Snapshot]:
    return _snapshot


def set_snapshot(snapshot: Optional[Snapshot]) -> None:
    """Swap atomically; requests already running keep the snapshot they started with."""
    global _snapshot
    _snapshot = snapshot


async def refresh(statuses: tuple[str, ...]) -> Snapshot:
    async for session in db_session.get_session():
        snapshot = await load_snapshot(session, statuses)
        set_snapshot(snapshot)
        logger.info(
            "engine snapshot loaded: %d destinations, catalog_version=%s, statuses=%s",
            snapshot.size, snapshot.catalog_version, ",".join(statuses),
        )
        return snapshot
    raise RuntimeError("No database session available")


async def _latest_version() -> Optional[int]:
    async for session in db_session.get_session():
        return await session.scalar(text("SELECT max(id) FROM catalog_versions"))
    return None


async def watch(statuses: tuple[str, ...], interval_seconds: int) -> None:
    """Reload when a new catalog version appears. Failures keep the last good snapshot."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            snapshot = current()
            latest = await _latest_version()
            if snapshot is None or latest != snapshot.catalog_version:
                await refresh(statuses)
        except Exception:
            logger.exception("engine snapshot refresh failed; keeping the previous snapshot")
