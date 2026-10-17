"""Unauthenticated, cacheable data for the marketing website."""
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from api.routes.recommendations import limiter
from api.v1.recommendations import _items
from api.v1.schemas import PublicSample
from engine.content import Preferences
from engine.ranker import Request as EngineRequest
from engine.ranker import recommend
from services import snapshot_store

router = APIRouter(prefix="/v1/public", tags=["Public"])

_cache: dict[tuple[date, Optional[int]], PublicSample] = {}


@router.get("/sample", response_model=PublicSample)
@limiter.limit("60/minute")
def sample(request: Request) -> PublicSample:
    """A real engine run for a fixed example trip, so the website's numbers are never invented."""
    snapshot = snapshot_store.current()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    today = date.today()
    key = (today, snapshot.catalog_version)
    if key in _cache:
        return _cache[key]

    origin = snapshot.states.get("NG-LA")
    if origin is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    engine_request = EngineRequest(
        origin=origin.place,
        origin_name=origin.name,
        days=1,
        travellers=2,
        budget=150_000,
        tier="mid",
        preferences=Preferences({"nature": "love", "photography": "like"}),
        month=today.month,
        limit=1,
    )
    result = recommend(snapshot.catalog, snapshot.inputs, engine_request)
    picks = _items(result.recommendations, snapshot)
    if not picks:
        raise HTTPException(status_code=503, detail="No example available yet")
    sample_out = PublicSample(
        origin=origin.name,
        days=1,
        travellers=2,
        budget=150_000,
        month=today.month,
        pick=picks[0],
        places=snapshot.size,
        states=len({d.place.state_iso for d in snapshot.catalog}),
    )
    _cache.clear()
    _cache[key] = sample_out
    return sample_out
