from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.v1.schemas import BudgetBandResponse, MetaResponse, StateOut, TagOut
from engine.costs import EATING_OUT_MARKUP
from engine.fuzzy import budget_band
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


@router.get("/meta/budget-band", response_model=BudgetBandResponse)
def band(per_person_per_day: float = Query(ge=0, le=100_000_000)) -> BudgetBandResponse:
    """The Low/Medium/High label for a daily spend, from the same fuzzy definition the engine uses."""
    return BudgetBandResponse(band=budget_band(per_person_per_day), per_person_per_day=per_person_per_day)
