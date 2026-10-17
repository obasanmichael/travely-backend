"""The signed-in (or anonymous) user's own data. Every query is scoped to the verified uid."""
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.schemas import DestinationOut, RecommendationResponse, RunDetail
from api.v1.serializers import destination_out
from core.security import verify_firebase_token
from db import session as db_session
from db.models import RecommendationRun, SavedDestination
from repositories.runs import ensure_user
from services import snapshot_store

router = APIRouter(prefix="/v1/me", tags=["Me"])


async def _session():
    if not db_session.engine_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    async for session in db_session.get_session():
        yield session


def _is_anonymous(token: dict[str, Any]) -> bool:
    return (token.get("firebase") or {}).get("sign_in_provider") == "anonymous"


class RunSummary(BaseModel):
    id: UUID
    created_at: datetime
    inputs: dict[str, Any]
    top_places: list[str]


class MergeRequest(BaseModel):
    anonymous_token: str = Field(min_length=20, max_length=4096)


@router.get("/saved", response_model=list[DestinationOut])
async def saved(
    token: dict = Depends(verify_firebase_token), session: AsyncSession = Depends(_session)
) -> list[DestinationOut]:
    snapshot = snapshot_store.current()
    ids = (await session.scalars(
        select(SavedDestination.destination_id)
        .where(SavedDestination.uid == token["uid"])
        .order_by(SavedDestination.created_at.desc())
    )).all()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    by_id = {d.id: d for d in snapshot.catalog}
    # Places unpublished since saving are left out rather than shown stale.
    return [destination_out(by_id[str(i)], snapshot) for i in ids if str(i) in by_id]


@router.put("/saved/{destination_id}", status_code=204)
async def save(
    destination_id: UUID,
    token: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(_session),
) -> Response:
    snapshot = snapshot_store.current()
    if snapshot is None or all(d.id != str(destination_id) for d in snapshot.catalog):
        raise HTTPException(status_code=404, detail="We couldn't find that place")
    async with session.begin():
        await ensure_user(session, token["uid"], token.get("email"), _is_anonymous(token))
        await session.execute(
            insert(SavedDestination)
            .values(uid=token["uid"], destination_id=destination_id)
            .on_conflict_do_nothing()
        )
    return Response(status_code=204)


@router.delete("/saved/{destination_id}", status_code=204)
async def unsave(
    destination_id: UUID,
    token: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(_session),
) -> Response:
    async with session.begin():
        await session.execute(
            delete(SavedDestination).where(
                SavedDestination.uid == token["uid"], SavedDestination.destination_id == destination_id
            )
        )
    return Response(status_code=204)


@router.get("/runs", response_model=list[RunSummary])
async def runs(
    limit: int = Query(default=20, ge=1, le=50),
    token: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(_session),
) -> list[RunSummary]:
    rows = (await session.execute(
        select(RecommendationRun.id, RecommendationRun.created_at, RecommendationRun.inputs, RecommendationRun.outputs)
        .where(RecommendationRun.uid == token["uid"])
        .order_by(RecommendationRun.created_at.desc())
        .limit(limit)
    )).all()
    return [
        RunSummary(
            id=r.id,
            created_at=r.created_at,
            inputs=r.inputs,
            top_places=[rec["destination"]["name"] for rec in (r.outputs.get("recommendations") or [])[:3]],
        )
        for r in rows
    ]


@router.get("/runs/{run_id}", response_model=RunDetail)
async def run_detail(
    run_id: UUID,
    token: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(_session),
) -> RunDetail:
    """One stored trip. Scoped to the caller: someone else's run id is just a 404."""
    row = (await session.execute(
        select(RecommendationRun.id, RecommendationRun.created_at, RecommendationRun.inputs, RecommendationRun.outputs)
        .where(RecommendationRun.id == run_id, RecommendationRun.uid == token["uid"])
    )).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="We couldn't find that trip")
    outputs = RecommendationResponse.model_validate(row.outputs)
    outputs.run_id = str(row.id)
    return RunDetail(id=row.id, created_at=row.created_at, inputs=row.inputs, outputs=outputs)


@router.post("/merge", status_code=204)
async def merge_anonymous(
    body: MergeRequest,
    token: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(_session),
) -> Response:
    """Move a guest session's saves and history into the account the user just signed into."""
    from firebase_admin import auth as firebase_auth

    if _is_anonymous(token):
        raise HTTPException(status_code=409, detail="Sign in to an account before merging a guest session")
    try:
        guest = firebase_auth.verify_id_token(body.anonymous_token)
    except Exception:
        raise HTTPException(status_code=401, detail="The guest session has expired") from None
    if not _is_anonymous(guest):
        raise HTTPException(status_code=409, detail="Only guest sessions can be merged")
    if guest["uid"] == token["uid"]:
        return Response(status_code=204)

    async with session.begin():
        await ensure_user(session, token["uid"], token.get("email"), False)
        await session.execute(text(
            "INSERT INTO saved_destinations (uid, destination_id, created_at) "
            "SELECT :to, destination_id, created_at FROM saved_destinations WHERE uid = :from "
            "ON CONFLICT DO NOTHING"
        ), {"to": token["uid"], "from": guest["uid"]})
        await session.execute(
            update(RecommendationRun).where(RecommendationRun.uid == guest["uid"]).values(uid=token["uid"])
        )
        await session.execute(text("DELETE FROM users WHERE firebase_uid = :from"), {"from": guest["uid"]})
    return Response(status_code=204)


