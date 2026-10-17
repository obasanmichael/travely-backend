from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engine.content import INTEREST_SLUGS

Tier = Literal["budget", "mid", "premium"]
Pace = Literal["relaxed", "balanced", "packed"]
Level = Literal["love", "like"]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin_state: str = Field(pattern=r"^NG-[A-Z]{2}$", examples=["NG-LA"])
    month: Optional[int] = Field(default=None, ge=1, le=12, description="Omit for flexible dates")
    days: int = Field(ge=1, le=30)
    travellers: int = Field(ge=1, le=20)
    budget: float = Field(gt=0, le=1_000_000_000, description="Total for the whole group, in NGN")
    tier: Tier = "mid"
    interests: dict[str, Level] = Field(default_factory=dict, max_length=8)
    pace: Pace = "balanced"
    with_children: bool = False
    max_travel_hours: Optional[float] = Field(default=None, gt=0, le=24)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("interests")
    @classmethod
    def _known_interests(cls, value: dict[str, Level]) -> dict[str, Level]:
        unknown = sorted(set(value) - INTEREST_SLUGS)
        if unknown:
            raise ValueError(f"Unknown interests: {', '.join(unknown)}")
        return value


class MoneyRange(BaseModel):
    low: int
    high: int


class CostComponentOut(BaseModel):
    name: str
    low: int
    high: int
    confidence: str
    basis: str
    observed_on: Optional[date] = None


class CostOut(BaseModel):
    total: MoneyRange
    confidence: str
    components: list[CostComponentOut]
    road_km: int
    travel_hours: float


class PhotoOut(BaseModel):
    url: str
    file_page_url: str
    author: str
    licence: str


class ReasonOut(BaseModel):
    kind: str
    text: str


class DestinationOut(BaseModel):
    id: str
    slug: str
    name: str
    state: str
    state_iso: str
    latitude: float
    longitude: float
    description: Optional[str] = None
    wikipedia_url: Optional[str] = None
    photo: Optional[PhotoOut] = None
    tags: list[str]
    best_months: list[int]
    month_scores: list[float] = Field(description="0-1 suitability for each month, January first")


class RecommendationOut(BaseModel):
    destination: DestinationOut
    score: float
    match: float = Field(description="0-1 fit with the user's interests")
    practicality: float = Field(description="0-1 fuzzy practicality")
    season_fit: float
    matched_interests: list[str]
    cost: CostOut
    reasons: list[ReasonOut]


class ExclusionsOut(BaseModel):
    over_budget: int
    too_far: int
    cost_unknown: int


class RecommendationResponse(BaseModel):
    run_id: Optional[str] = None
    engine_version: str
    catalog_version: Optional[int]
    budget_band: str
    relaxed: list[str]
    exclusions: ExclusionsOut
    considered: int
    cheapest_feasible_budget: Optional[int]
    notes: list[str]
    recommendations: list[RecommendationOut]


class StateOut(BaseModel):
    iso: str
    name: str


class TagOut(BaseModel):
    slug: str
    label: str
    category: str
    description: str


class MetaResponse(BaseModel):
    states: list[StateOut]
    interests: list[TagOut]
    tiers: list[str]
    paces: list[str]
    catalog_size: int
    catalog_version: Optional[int]
