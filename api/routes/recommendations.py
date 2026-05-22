from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.security import verify_firebase_token
from services.recommendation_engine import get_recommendations

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Recommendations"])
limiter = Limiter(key_func=get_remote_address)


class RecommendationRequest(BaseModel):
    budget: float = Field(gt=0, le=50000)
    destination_type: Optional[str] = Field(default=None, min_length=1)
    activity_type: Optional[str] = Field(default=None, min_length=1)


class Recommendation(BaseModel):
    destination: str
    state: str
    city: str
    destination_type: str
    activities: str
    climate: str
    avg_cost_per_day: float
    best_season: str
    accommodation_type: str
    nearby_hotel: str
    hotel_price_range: str
    feeding_cost_range: str
    necessities_range: str
    budget_category: str
    score: float


class RecommendationResponse(BaseModel):
    user_budget_category: str
    recommendations: List[Recommendation]


@router.post("/recommendations", response_model=RecommendationResponse)
@limiter.limit("30/minute")
def recommend(
    request: Request,
    payload: RecommendationRequest,
    token: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    uid = token.get("uid", "unknown")
    logger.info("Recommendation request from uid=%s", uid)

    recommendations = get_recommendations(
        budget=payload.budget,
        destination_type=payload.destination_type,
        activity_type=payload.activity_type,
    )

    if not recommendations or "recommendations" not in recommendations:
        return {"user_budget_category": "Medium", "recommendations": []}

    return recommendations
