from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.v1.schemas import MetaResponse, StateOut, TagOut
from engine.costs import EATING_OUT_MARKUP
from engine.taxonomy import TAXONOMY
from services import snapshot_store

router = APIRouter(prefix="/v1", tags=["Meta"])


@router.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    """Everything the quiz needs to render its choices: one source of truth for the frontend."""
    snapshot = snapshot_store.current()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Catalog is loading, try again shortly")
    return MetaResponse(
        states=sorted(
            (StateOut(iso=s.iso, name=s.name) for s in snapshot.states.values()), key=lambda s: s.name
        ),
        interests=[
            TagOut(slug=t.slug, label=t.label, category=t.category, description=t.description)
            for t in TAXONOMY
            if t.is_interest
        ],
        tiers=list(EATING_OUT_MARKUP),
        paces=["relaxed", "balanced", "packed"],
        catalog_size=snapshot.size,
        catalog_version=snapshot.catalog_version,
    )
