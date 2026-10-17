import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from api.routes.recommendations import limiter
from api.v1.schemas import (
    CostComponentOut,
    CostOut,
    DestinationOut,
    ExclusionsOut,
    MoneyRange,
    PhotoOut,
    ReasonOut,
    RecommendationOut,
    RecommendationRequest,
    RecommendationResponse,
)
from core.security import verify_firebase_token
from db import session as db_session
from engine.content import Preferences
from engine.fuzzy import budget_band
from engine.ranker import ENGINE_VERSION, Result, recommend
from engine.ranker import Request as EngineRequest
from engine.snapshot import Snapshot
from repositories.runs import IdempotencyConflict, ensure_user, save_run
from services import snapshot_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Recommendations"])

TOP_TAGS = 4


def _serialise(result: Result, snapshot: Snapshot, request: RecommendationRequest) -> RecommendationResponse:
    items = []
    for rec in result.recommendations:
        d = rec.destination
        details = snapshot.details.get(d.id)
        photo = details.photo if details else None
        items.append(
            RecommendationOut(
                destination=DestinationOut(
                    id=d.id,
                    slug=d.slug,
                    name=d.name,
                    state=d.state_name,
                    state_iso=d.place.state_iso,
                    latitude=d.place.lat,
                    longitude=d.place.lon,
                    description=details.description if details else None,
                    wikipedia_url=details.wikipedia_url if details else None,
                    photo=PhotoOut(**photo.__dict__) if photo else None,
                    tags=[t for t, _ in sorted(d.tags.items(), key=lambda kv: -kv[1])[:TOP_TAGS]],
                    best_months=list(details.best_months) if details else [],
                    month_scores=list(details.month_scores) if details else [],
                ),
                score=rec.score,
                match=rec.content,
                practicality=rec.practicality,
                season_fit=rec.season_fit,
                matched_interests=list(rec.matched),
                cost=CostOut(
                    total=MoneyRange(low=round(rec.estimate.total.low), high=round(rec.estimate.total.high)),
                    confidence=rec.estimate.confidence.value,
                    components=[
                        CostComponentOut(
                            name=c.name,
                            low=round(c.cost.low),
                            high=round(c.cost.high),
                            confidence=c.confidence.value,
                            basis=c.basis,
                            observed_on=c.observed_on,
                        )
                        for c in rec.estimate.components
                    ],
                    road_km=round(rec.estimate.road_km),
                    travel_hours=round(rec.estimate.travel_hours, 1),
                ),
                reasons=[ReasonOut(kind=r.kind, text=r.text) for r in rec.reasons],
            )
        )
    per_person_per_day = request.budget / (request.travellers * request.days)
    return RecommendationResponse(
        engine_version=result.engine_version,
        catalog_version=snapshot.catalog_version,
        budget_band=budget_band(per_person_per_day),
        relaxed=list(result.relaxed),
        exclusions=ExclusionsOut(**result.exclusions.__dict__),
        considered=result.considered,
        cheapest_feasible_budget=(
            round(result.cheapest_feasible_budget) if result.cheapest_feasible_budget is not None else None
        ),
        notes=list(result.notes),
        recommendations=items,
    )


@router.post("/recommendations", response_model=RecommendationResponse)
@limiter.limit("30/minute")
async def create_recommendations(
    request: Request,
    payload: RecommendationRequest,
    token: dict[str, Any] = Depends(verify_firebase_token),
    idempotency_key: Optional[str] = Header(
        default=None, alias="Idempotency-Key", min_length=8, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$"
    ),
) -> RecommendationResponse:
    snapshot = snapshot_store.current()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    origin = snapshot.states.get(payload.origin_state)
    if origin is None:
        raise HTTPException(status_code=422, detail=f"Unknown origin state {payload.origin_state}")

    engine_request = EngineRequest(
        origin=origin.place,
        origin_name=origin.name,
        days=payload.days,
        travellers=payload.travellers,
        budget=payload.budget,
        tier=payload.tier,
        preferences=Preferences(dict(payload.interests), payload.pace, payload.with_children),
        month=payload.month,
        max_travel_hours=payload.max_travel_hours,
        limit=payload.limit,
    )
    response = _serialise(recommend(snapshot.catalog, snapshot.inputs, engine_request), snapshot, payload)

    if idempotency_key is None or not db_session.engine_configured():
        return response

    uid = token["uid"]
    provider = (token.get("firebase") or {}).get("sign_in_provider")
    async for session in db_session.get_session():
        async with session.begin():
            await ensure_user(session, uid, token.get("email"), is_anonymous=provider == "anonymous")
            try:
                run_id, outputs, created = await save_run(
                    session,
                    uid=uid,
                    idempotency_key=idempotency_key,
                    inputs=payload.model_dump(mode="json"),
                    outputs=response.model_dump(mode="json", exclude={"run_id"}),
                    engine_version=ENGINE_VERSION,
                    catalog_version=snapshot.catalog_version,
                )
            except IdempotencyConflict:
                raise HTTPException(
                    status_code=409,
                    detail="This request id was already used with different answers. Start a new search.",
                ) from None
    stored = RecommendationResponse.model_validate(outputs)
    stored.run_id = str(run_id)
    if not created:
        logger.info("replayed run %s for repeated idempotency key", run_id)
    return stored
