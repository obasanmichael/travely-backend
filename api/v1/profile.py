"""Profile, preferences, data export and account deletion for the signed-in user."""
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.me import _is_anonymous, _session
from core.security import verify_firebase_token
from db.models import RecommendationRun, SavedDestination, State, User
from engine.content import INTEREST_SLUGS
from repositories.runs import ensure_user

router = APIRouter(prefix="/v1/me", tags=["Me"])

Tier = Literal["budget", "mid", "premium"]
Pace = Literal["relaxed", "balanced", "packed"]
Level = Literal["love", "like"]


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Optional[Tier] = None
    pace: Optional[Pace] = None
    interests: dict[str, Level] = Field(default_factory=dict, max_length=8)

    @field_validator("interests")
    @classmethod
    def _known(cls, value: dict[str, Level]) -> dict[str, Level]:
        unknown = sorted(set(value) - INTEREST_SLUGS)
        if unknown:
            raise ValueError(f"Unknown interests: {', '.join(unknown)}")
        return value


class ProfileOut(BaseModel):
    email: Optional[str]
    display_name: Optional[str]
    home_state: Optional[str]
    preferences: Preferences
    analytics_consent: bool
    marketing_consent: bool


class ProfilePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    home_state: Optional[str] = Field(default=None, pattern=r"^NG-[A-Z]{2}$")
    clear_home_state: bool = False
    preferences: Optional[Preferences] = None
    analytics_consent: Optional[bool] = None
    marketing_consent: Optional[bool] = None


async def _profile(session: AsyncSession, uid: str) -> ProfileOut:
    row = (await session.execute(
        select(User.email, User.display_name, User.preferences, User.analytics_consent, User.marketing_consent,
               text("(SELECT iso_code FROM states WHERE id = users.home_state_id)"))
        .where(User.firebase_uid == uid)
    )).one_or_none()
    if row is None:
        return ProfileOut(email=None, display_name=None, home_state=None, preferences=Preferences(),
                          analytics_consent=False, marketing_consent=False)
    return ProfileOut(
        email=row[0], display_name=row[1], home_state=row[5],
        preferences=Preferences.model_validate(row[2] or {}),
        analytics_consent=row[3], marketing_consent=row[4],
    )


@router.get("/profile", response_model=ProfileOut)
async def get_profile(token: dict = Depends(verify_firebase_token), session: AsyncSession = Depends(_session)) -> ProfileOut:
    return await _profile(session, token["uid"])


@router.patch("/profile", response_model=ProfileOut)
async def patch_profile(
    patch: ProfilePatch,
    token: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(_session),
) -> ProfileOut:
    uid = token["uid"]
    async with session.begin():
        await ensure_user(session, uid, token.get("email"), _is_anonymous(token))
        values: dict[str, Any] = {}
        if patch.display_name is not None:
            values["display_name"] = patch.display_name.strip()
        if patch.home_state is not None:
            state_id = await session.scalar(select(State.id).where(State.iso_code == patch.home_state))
            if state_id is None:
                raise HTTPException(status_code=422, detail="Unknown state")
            values["home_state_id"] = state_id
        if patch.clear_home_state:
            values["home_state_id"] = None
        if patch.preferences is not None:
            values["preferences"] = patch.preferences.model_dump(exclude_none=True)
        if patch.analytics_consent is not None:
            values["analytics_consent"] = patch.analytics_consent
        if patch.marketing_consent is not None:
            values["marketing_consent"] = patch.marketing_consent
        if values:
            await session.execute(User.__table__.update().where(User.firebase_uid == uid).values(**values))
    return await _profile(session, uid)


@router.get("/export")
async def export_data(token: dict = Depends(verify_firebase_token), session: AsyncSession = Depends(_session)) -> Response:
    """Everything Travely stores about the caller, as a downloadable JSON file."""
    import json

    uid = token["uid"]
    profile = await _profile(session, uid)
    saved = (await session.execute(
        select(SavedDestination.destination_id, SavedDestination.created_at).where(SavedDestination.uid == uid)
    )).all()
    runs = (await session.execute(
        select(RecommendationRun.id, RecommendationRun.created_at, RecommendationRun.inputs, RecommendationRun.outputs,
               RecommendationRun.engine_version).where(RecommendationRun.uid == uid).order_by(RecommendationRun.created_at)
    )).all()

    def iso(value: datetime) -> str:
        return value.isoformat()

    payload = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "profile": profile.model_dump(),
        "saved_places": [{"destination_id": str(r.destination_id), "saved_at": iso(r.created_at)} for r in saved],
        "trips": [
            {"id": str(r.id), "created_at": iso(r.created_at), "engine_version": r.engine_version,
             "answers": r.inputs, "results": r.outputs}
            for r in runs
        ],
    }
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="travely-data.json"'},
    )


@router.delete("", status_code=204)
async def delete_account(token: dict = Depends(verify_firebase_token), session: AsyncSession = Depends(_session)) -> Response:
    """Delete the caller's data, then their Firebase account. Both go, or the call fails and can be retried."""
    uid = token["uid"]
    async with session.begin():
        # Runs, saves and run events cascade from the user row.
        await session.execute(delete(User).where(User.firebase_uid == uid))
    from firebase_admin import auth as firebase_auth

    try:
        firebase_auth.delete_user(uid)
    except firebase_auth.UserNotFoundError:
        pass
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Your data was deleted but we couldn't remove your sign-in yet. Try again in a moment.",
        ) from None
    return Response(status_code=204)
