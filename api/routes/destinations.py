from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from core.security import verify_firebase_token
from services.data_cache import get_catalog_df

router = APIRouter(tags=["Destinations"])


class DestinationSummary(BaseModel):
    destination: str
    state: str
    city: str
    destination_type: str
    activities: str
    climate: str
    avg_cost_per_day: float
    best_season: str
    budget_category: str


def _budget_band(cost: float) -> str:
    if cost <= 12000:
        return "low"
    if cost <= 25000:
        return "medium"
    return "high"


@router.get("/destinations", response_model=List[DestinationSummary])
def list_destinations(
    state: Optional[str] = Query(None),
    destination_type: Optional[str] = Query(None),
    budget_band: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    q: Optional[str] = Query(None, min_length=1),
    _token: Dict[str, Any] = Depends(verify_firebase_token),
) -> List[Dict[str, Any]]:
    df = get_catalog_df()
    if df.empty:
        return []

    if state:
        df = df[df["state"].str.lower() == state.lower()]

    if destination_type:
        needle = destination_type.lower()
        df = df[df["destination_type_keywords"].str.contains(needle, na=False)]

    if budget_band:
        df = df[df["avg_cost_per_day"].apply(lambda cost: _budget_band(float(cost)) == budget_band)]

    if q:
        needle = q.lower()
        df = df[
            df["destination"].str.lower().str.contains(needle, na=False)
            | df["city"].str.lower().str.contains(needle, na=False)
            | df["state"].str.lower().str.contains(needle, na=False)
        ]

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        cost = float(row["avg_cost_per_day"])
        results.append(
            {
                "destination": str(row["destination"]),
                "state": str(row["state"]),
                "city": str(row["city"]),
                "destination_type": str(row["destination_type"]),
                "activities": str(row["activities"]),
                "climate": str(row["climate"]),
                "avg_cost_per_day": cost,
                "best_season": str(row["best_season"]),
                "budget_category": _budget_band(cost),
            }
        )

    return sorted(results, key=lambda item: item["destination"])
