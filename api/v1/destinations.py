import hashlib
from datetime import date

from fastapi import APIRouter, HTTPException, Query

from api.v1.schemas import DestinationOut
from api.v1.serializers import destination_out
from services import snapshot_store

router = APIRouter(prefix="/v1/destinations", tags=["Destinations"])


def _daily_order(destination_id: str, day: date) -> str:
    # Stable within a day, different across days: the home page rotates without randomness in tests.
    return hashlib.sha256(f"{day.isoformat()}:{destination_id}".encode()).hexdigest()


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
