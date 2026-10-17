import hashlib
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.v1.schemas import DestinationOut, DestinationPage
from api.v1.serializers import destination_out
from engine.season import GREAT_SCORE
from services import snapshot_store

router = APIRouter(prefix="/v1/destinations", tags=["Destinations"])


def _daily_order(destination_id: str, day: date) -> str:
    # Stable within a day, different across days: the home page rotates without randomness in tests.
    return hashlib.sha256(f"{day.isoformat()}:{destination_id}".encode()).hexdigest()


@router.get("", response_model=DestinationPage)
def browse(
    q: Optional[str] = Query(default=None, min_length=2, max_length=60),
    state: Optional[str] = Query(default=None, pattern=r"^NG-[A-Z]{2}$"),
    interest: Optional[str] = Query(default=None, max_length=40),
    month: Optional[int] = Query(default=None, ge=1, le=12, description="Only places that are great in this month"),
    page: int = Query(default=1, ge=1, le=500),
    page_size: int = Query(default=24, ge=1, le=60),
) -> DestinationPage:
    """Browse the published catalog. Documented places with photos come first."""
    snapshot = snapshot_store.current()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    needle = q.lower() if q else None
    matches = []
    for d in snapshot.catalog:
        if state and d.place.state_iso != state:
            continue
        if interest and d.tags.get(interest, 0) < 0.5:
            continue
        details = snapshot.details.get(d.id)
        if month and not (details and len(details.month_scores) == 12 and details.month_scores[month - 1] >= GREAT_SCORE):
            continue
        if needle and needle not in d.name.lower() and needle not in d.state_name.lower():
            continue
        matches.append(d)
    matches.sort(key=lambda d: (-d.quality, d.name))
    start = (page - 1) * page_size
    return DestinationPage(
        items=[destination_out(d, snapshot) for d in matches[start : start + page_size]],
        total=len(matches),
        page=page,
        page_size=page_size,
    )


@router.get("/featured", response_model=list[DestinationOut])
def featured(limit: int = Query(default=6, ge=1, le=12)) -> list[DestinationOut]:
    """Well-documented places with photos, for the home page. Rotates daily."""
    snapshot = snapshot_store.current()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    with_photos = [
        d for d in snapshot.catalog
        if (details := snapshot.details.get(d.id)) and details.photo and len(details.month_scores) == 12
    ]
    best = sorted(with_photos, key=lambda d: (-d.quality, _daily_order(d.id, date.today())))
    return [destination_out(d, snapshot) for d in best[:limit]]


@router.get("/{slug}", response_model=DestinationOut)
def by_slug(slug: str) -> DestinationOut:
    snapshot = snapshot_store.current()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    match = next((d for d in snapshot.catalog if d.slug == slug), None)
    if match is None:
        raise HTTPException(status_code=404, detail="We couldn't find that place")
    return destination_out(match, snapshot)
