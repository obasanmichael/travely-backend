"""The four-stage recommendation pipeline.

1. Hard constraints remove what can't work (over budget, too far, unknown
   cost). If too few remain, constraints are relaxed in a fixed order and the
   response says exactly what was relaxed.
2. Fuzzy practicality scores affordability x season x travel burden.
3. Content similarity scores fit with the user's interests.
4. A weighted blend is diversified with Maximal Marginal Relevance and each
   pick gets reasons built from its own score components.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from engine.content import (
    Preferences,
    cosine_many,
    has_interests,
    labels,
    matched_interests,
    pairwise_similarity,
    profile_vector,
    tag_vector,
)
from engine.costs import CostEstimate, CostInputs, Place, TripRequest, estimate_trip
from engine.fuzzy import practicality_many, travel_burden
from engine.season import MonthClimate, season_fit

ENGINE_VERSION = "3.0.0"

WEIGHTS = {"content": 0.55, "practicality": 0.35, "quality": 0.10}
# Without interests, content can't discriminate; lean on practicality and quality.
WEIGHTS_NO_INTERESTS = {"content": 0.0, "practicality": 0.75, "quality": 0.25}
BUDGET_TOLERANCE = 1.15
RELAXED_BUDGET_TOLERANCE = 1.35
# 1.0 = pure relevance, 0.0 = pure variety. At 0.75 a waterfall lover got three
# near-identical waterfalls in one state; 0.6 keeps the best match first and then mixes.
MMR_LAMBDA = 0.6
SAME_STATE_SIMILARITY = 0.3
# Variety is only chosen among places that match the user at all; places below
# this content score only fill slots nothing relevant could.
CONTENT_FLOOR = 0.15


@dataclass(frozen=True)
class Destination:
    id: str
    slug: str
    name: str
    place: Place
    state_name: str
    tags: dict[str, float]
    climate: tuple[MonthClimate, ...]
    quality: float
    city_key: Optional[str] = None


@dataclass(frozen=True)
class Request:
    origin: Place
    origin_name: str
    days: int
    travellers: int
    budget: float  # total for the whole group, in NGN
    tier: str
    preferences: Preferences
    month: Optional[int] = None  # None means flexible
    max_travel_hours: Optional[float] = None
    limit: int = 5

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("Budget must be positive")
        if self.month is not None and not 1 <= self.month <= 12:
            raise ValueError("Month must be 1-12")
        if not 1 <= self.limit <= 20:
            raise ValueError("Limit must be 1-20")


@dataclass(frozen=True)
class Reason:
    kind: str  # interests | cost | season | travel | confidence
    text: str


@dataclass(frozen=True)
class Recommendation:
    destination: Destination
    estimate: CostEstimate
    score: float
    content: float
    practicality: float
    season_fit: float
    best_month: Optional[int]
    matched: tuple[str, ...]
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Exclusions:
    over_budget: int = 0
    too_far: int = 0
    cost_unknown: int = 0


@dataclass(frozen=True)
class Result:
    recommendations: tuple[Recommendation, ...]
    relaxed: tuple[str, ...]
    exclusions: Exclusions
    considered: int
    engine_version: str = ENGINE_VERSION
    cheapest_feasible_budget: Optional[float] = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Places priced except for accommodation, shown separately and labelled, so an
    # overnight search isn't empty while rates are still being verified.
    unverified_stays: tuple[Recommendation, ...] = ()
    # Cheapest trip counting everything except an unverified stay, so "nothing fits"
    # can say whether the budget or the missing rates is the real problem.
    cheapest_without_stay: Optional[float] = None


def _naira(amount: float) -> str:
    rounded = int(round(amount, -3)) if amount >= 10_000 else int(round(amount, -2))
    return f"₦{rounded:,}"


def _month_name(month: int) -> str:
    return calendar.month_name[month]


@dataclass
class _Candidate:
    destination: Destination
    estimate: CostEstimate


def _eligible(
    candidates: Sequence[_Candidate],
    request: Request,
    tolerance: float,
    enforce_distance: bool,
) -> tuple[list[_Candidate], Exclusions]:
    kept: list[_Candidate] = []
    over_budget = too_far = 0
    for candidate in candidates:
        if candidate.estimate.total.low > request.budget * tolerance:
            over_budget += 1
            continue
        if (
            enforce_distance
            and request.max_travel_hours is not None
            and candidate.estimate.travel_hours > request.max_travel_hours
        ):
            too_far += 1
            continue
        kept.append(candidate)
    return kept, Exclusions(over_budget=over_budget, too_far=too_far)


def _reasons(
    request: Request,
    candidate: _Candidate,
    matched: Sequence[str],
    season_score: float,
    best_month: Optional[int],
) -> tuple[Reason, ...]:
    reasons: list[Reason] = []
    if matched:
        reasons.append(Reason("interests", f"Matches your interest in {', '.join(labels(matched[:3])).lower()}"))

    total = candidate.estimate.total
    if "accommodation" in candidate.estimate.missing:
        reasons.append(Reason(
            "cost",
            f"Travel and food come to {_naira(total.low)}–{_naira(total.high)}; "
            "places-to-stay prices aren't verified yet",
        ))
    elif total.high <= request.budget:
        reasons.append(Reason("cost", f"About {_naira(request.budget - total.mid)} under your budget"))
    elif total.low <= request.budget:
        reasons.append(Reason("cost", f"Fits your budget at the lower end ({_naira(total.low)}–{_naira(total.high)})"))
    else:
        reasons.append(Reason("cost", f"Slightly over budget: from {_naira(total.low)}"))

    if request.month is not None:
        if season_score >= 0.8:
            reasons.append(Reason("season", f"{_month_name(request.month)} is a great time to visit"))
        elif season_score < 0.5 and best_month:
            reasons.append(Reason("season", f"Better in {_month_name(best_month)}; {_month_name(request.month)} is rainy here"))
    elif best_month:
        reasons.append(Reason("season", f"Best in {_month_name(best_month)}"))

    hours = candidate.estimate.travel_hours
    if candidate.destination.place.state_iso == request.origin.state_iso and hours < 1.5:
        reasons.append(Reason("travel", f"Close to home in {request.origin_name}"))
    else:
        reasons.append(Reason("travel", f"About {max(1, round(hours))} hrs by road from {request.origin_name}"))

    reasons.append(Reason("confidence", f"Cost estimate confidence: {candidate.estimate.confidence.value}"))
    return tuple(reasons)


def _mmr(scores: np.ndarray, similarity: np.ndarray, limit: int) -> list[int]:
    """Greedy Maximal Marginal Relevance: trade a little score for a lot less repetition."""
    selected: list[int] = []
    remaining = list(np.argsort(-scores))
    while remaining and len(selected) < limit:
        if not selected:
            best = remaining[0]
        else:
            penalty = similarity[np.ix_(remaining, selected)].max(axis=1)
            mmr = MMR_LAMBDA * scores[remaining] - (1 - MMR_LAMBDA) * penalty
            best = remaining[int(np.argmax(mmr))]
        selected.append(best)
        remaining.remove(best)
    return selected


def _rank(candidates: Sequence[_Candidate], request: Request, limit: int) -> list[Recommendation]:
    """Stages 2-4 for a set of candidates that already passed the hard constraints."""
    if not candidates:
        return []
    eligible = list(candidates)
    month = request.month
    # Stage 2: fuzzy practicality.
    seasons = [season_fit(c.destination.climate, c.destination.tags, month) for c in eligible]
    affordability = np.array([request.budget / max(c.estimate.total.mid, 1.0) for c in eligible])
    burden = np.array([travel_burden(c.estimate.travel_hours, request.days) for c in eligible])
    practical = practicality_many(affordability, np.array([s for s, _ in seasons]), burden)

    # Stage 3: content similarity.
    matrix = np.vstack([tag_vector(c.destination.tags) for c in eligible])
    content = cosine_many(profile_vector(request.preferences), matrix)

    # Stage 4: blend, diversify, explain.
    weights = WEIGHTS if has_interests(request.preferences) else WEIGHTS_NO_INTERESTS
    quality = np.array([c.destination.quality for c in eligible])
    scores = weights["content"] * content + weights["practicality"] * practical + weights["quality"] * quality

    similarity = pairwise_similarity(matrix)
    states = np.array([c.destination.place.state_iso for c in eligible])
    similarity = np.maximum(similarity, np.where(states[:, None] == states[None, :], SAME_STATE_SIMILARITY, 0))
    if has_interests(request.preferences):
        relevant = np.flatnonzero(content >= CONTENT_FLOOR)
        picks = [int(relevant[i]) for i in _mmr(scores[relevant], similarity[np.ix_(relevant, relevant)], limit)]
        fillers = [int(i) for i in np.argsort(-scores) if int(i) not in set(relevant)]
        picks += fillers[: limit - len(picks)]
    else:
        picks = _mmr(scores, similarity, limit)

    recommendations = []
    for index in picks:
        candidate = eligible[index]
        season_score, best_month = seasons[index]
        matched = tuple(matched_interests(request.preferences, candidate.destination.tags))
        recommendations.append(
            Recommendation(
                destination=candidate.destination,
                estimate=candidate.estimate,
                score=round(float(scores[index]), 4),
                content=round(float(content[index]), 4),
                practicality=float(practical[index]),
                season_fit=season_score,
                best_month=best_month,
                matched=matched,
                reasons=_reasons(request, candidate, matched, season_score, best_month),
            )
        )
    return recommendations


UNVERIFIED_STAYS_LIMIT = 4


def _unverified_stays(estimated: Sequence[_Candidate], request: Request, room: int) -> list[Recommendation]:
    """Overnight trips to places whose only gap is accommodation, if what we can price fits."""
    if room <= 0 or request.days < 2:
        return []
    partial = [
        c for c in estimated
        if c.estimate.missing == ("accommodation",)
        and c.estimate.total.low <= request.budget * BUDGET_TOLERANCE
        and (request.max_travel_hours is None or c.estimate.travel_hours <= request.max_travel_hours)
    ]
    return _rank(partial, request, min(room, UNVERIFIED_STAYS_LIMIT))


def recommend(
    catalog: Sequence[Destination], inputs: CostInputs, request: Request
) -> Result:
    trip = TripRequest(request.origin, request.days, request.travellers, request.tier)

    estimated = [
        _Candidate(d, estimate_trip(inputs, trip, d.id, d.place, d.city_key)) for d in catalog
    ]
    known = [c for c in estimated if c.estimate.complete]
    cost_unknown = len(estimated) - len(known)

    # Stage 1, relaxing in a fixed order until there are enough results.
    relaxed: list[str] = []
    eligible, exclusions = _eligible(known, request, BUDGET_TOLERANCE, enforce_distance=True)
    if len(eligible) < request.limit:
        eligible, exclusions = _eligible(known, request, RELAXED_BUDGET_TOLERANCE, enforce_distance=True)
        relaxed.append("budget")
    if len(eligible) < request.limit and request.max_travel_hours is not None:
        eligible, exclusions = _eligible(known, request, RELAXED_BUDGET_TOLERANCE, enforce_distance=False)
        relaxed.append("travel_time")
    exclusions = Exclusions(exclusions.over_budget, exclusions.too_far, cost_unknown)
    # Relaxation that didn't add anything isn't worth telling the user about.
    if relaxed and len(eligible) == len(_eligible(known, request, BUDGET_TOLERANCE, True)[0]):
        relaxed = []

    cheapest = min((c.estimate.total.low for c in known), default=None)
    cheapest_without_stay = min(
        (c.estimate.total.low for c in estimated if set(c.estimate.missing) <= {"accommodation"}),
        default=None,
    )
    if not eligible:
        notes = ("No destination fits this budget yet.",) if known else ("No priced destinations yet.",)
        return Result(
            (), tuple(relaxed), exclusions, len(catalog), cheapest_feasible_budget=cheapest, notes=notes,
            unverified_stays=tuple(_unverified_stays(estimated, request, request.limit)),
            cheapest_without_stay=cheapest_without_stay,
        )

    recommendations = _rank(eligible, request, request.limit)
    unverified = _unverified_stays(estimated, request, request.limit - len(recommendations))
    notes = () if has_interests(request.preferences) else ("Popular picks: add interests for a closer match.",)
    return Result(
        tuple(recommendations),
        tuple(relaxed),
        exclusions,
        len(catalog),
        cheapest_feasible_budget=cheapest,
        notes=notes,
        unverified_stays=tuple(unverified),
        cheapest_without_stay=cheapest_without_stay,
    )
