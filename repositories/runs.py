"""Users and recommendation runs. Every query is scoped by the verified Firebase uid."""
from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import RecommendationRun, User


class IdempotencyConflict(Exception):
    """The same idempotency key was sent with different quiz answers."""


async def ensure_user(session: AsyncSession, uid: str, email: Optional[str], is_anonymous: bool) -> None:
    stmt = insert(User).values(firebase_uid=uid, email=email, is_anonymous=is_anonymous)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=["firebase_uid"],
            set_={"email": stmt.excluded.email, "is_anonymous": stmt.excluded.is_anonymous},
        )
    )


async def save_run(
    session: AsyncSession,
    *,
    uid: str,
    idempotency_key: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    engine_version: str,
    catalog_version: Optional[int],
) -> tuple[uuid.UUID, dict[str, Any], bool]:
    """Store a run once per (uid, key). Returns (run id, outputs, created).

    A repeated key returns the run stored the first time, so a double-submitted
    quiz shows the same results instead of a second, possibly different run.
    """
    stmt = insert(RecommendationRun).values(
        uid=uid,
        idempotency_key=idempotency_key,
        inputs=inputs,
        outputs=outputs,
        engine_version=engine_version,
        catalog_version_id=catalog_version,
    )
    created_id = await session.scalar(
        stmt.on_conflict_do_nothing(constraint="uq_recommendation_runs_uid_idem").returning(
            RecommendationRun.id
        )
    )
    if created_id is not None:
        return created_id, outputs, True
    existing = (
        await session.execute(
            select(
                RecommendationRun.id, RecommendationRun.inputs, RecommendationRun.outputs
            ).where(RecommendationRun.uid == uid, RecommendationRun.idempotency_key == idempotency_key)
        )
    ).one()
    if existing.inputs != inputs:
        raise IdempotencyConflict(idempotency_key)
    return existing.id, existing.outputs, False
