from __future__ import annotations

from datetime import date

import pytest

from engine.content import Preferences
from engine.costs import MEAL_BASKET, CostInputs, Observation, Place
from engine.ranker import Destination, Request, recommend
from engine.season import MonthClimate

MAY = date(2026, 5, 1)
TODAY = date(2026, 10, 1)
LAGOS = Place("NG-LA", "south_west", 3.38, 6.52)

DRY_WINTER = tuple(
    MonthClimate(m, 26.0, r)
    for m, r in zip(range(1, 13), [5, 10, 40, 90, 150, 200, 260, 280, 200, 120, 20, 5])
)

FOOD = {item: Observation(1000.0, MAY) for item in MEAL_BASKET}
STATES = {
    "NG-LA": ("south_west", 3.4, 6.5),
    "NG-OG": ("south_west", 3.35, 7.15),
    "NG-OY": ("south_west", 3.9, 7.4),
    "NG-CR": ("south_south", 9.3, 6.4),
    "NG-BA": ("north_east", 10.2, 9.8),
}


def _inputs(rates: dict[str, float]) -> CostInputs:
    return CostInputs(
        as_of=TODAY,
        fares={iso: {"bus": Observation(9000, MAY), "local_bus": Observation(1200, MAY)} for iso in STATES},
        food_prices={zone: FOOD for zone in {"south_west", "south_south", "north_east"}},
        accommodation={("destination", d, "mid"): [Observation(r, date(2026, 9, 1))] * 3 for d, r in rates.items()},
    )


def _dest(id_, iso, tags, quality=0.5, offset=0.0):
    zone, lon, lat = STATES[iso]
    return Destination(
        id=id_, slug=id_, name=id_.title(), place=Place(iso, zone, lon + offset, lat + offset),
        state_name=iso, tags=tags, climate=DRY_WINTER, quality=quality,
    )


CATALOG = [
    _dest("falls-a", "NG-OG", {"waterfalls": 1.0, "nature": 0.8, "hiking": 0.5}),
    _dest("falls-b", "NG-OG", {"waterfalls": 1.0, "nature": 0.8, "hiking": 0.5}, offset=0.05),
    _dest("falls-c", "NG-OG", {"waterfalls": 1.0, "nature": 0.8, "hiking": 0.5}, offset=0.1),
    _dest("hills", "NG-OY", {"hills-mountains": 1.0, "hiking": 0.9, "nature": 0.6}),
    _dest("museum", "NG-LA", {"museums": 1.0, "history": 0.8}),
    _dest("beach", "NG-LA", {"beaches": 1.0, "relaxation": 0.7}),
    _dest("plateau", "NG-CR", {"nature": 1.0, "hiking": 0.8, "waterfalls": 0.4}),
    _dest("park", "NG-BA", {"wildlife": 1.0, "nature": 1.0}),
    _dest("unpriced", "NG-LA", {"waterfalls": 1.0, "hiking": 1.0}),
]
RATES = {d.id: 30000.0 for d in CATALOG if d.id != "unpriced"}
RATES["park"] = 60000.0


def _request(**overrides) -> Request:
    values = dict(
        origin=LAGOS, origin_name="Lagos", days=3, travellers=2, budget=400_000, tier="mid",
        preferences=Preferences({"waterfalls": "love", "hiking": "like"}), month=1, limit=3,
    )
    values.update(overrides)
    return Request(**values)


def test_top_pick_matches_loved_interest_and_explains_why():
    result = recommend(CATALOG, _inputs(RATES), _request())
    top = result.recommendations[0]
    assert "waterfalls" in top.matched
    kinds = {r.kind for r in top.reasons}
    assert {"interests", "cost", "season", "travel", "confidence"} <= kinds


def test_unpriced_destinations_are_never_recommended():
    result = recommend(CATALOG, _inputs(RATES), _request(limit=8))
    assert "unpriced" not in {r.destination.id for r in result.recommendations}
    assert result.exclusions.cost_unknown == 1


def test_diversity_avoids_three_near_identical_waterfalls():
    result = recommend(CATALOG, _inputs(RATES), _request(limit=3))
    ids = [r.destination.id for r in result.recommendations]
    assert sum(i.startswith("falls") for i in ids) < 3


def test_results_stay_within_budget_tolerance():
    request = _request(budget=150_000, limit=8)
    result = recommend(CATALOG, _inputs(RATES), request)
    for rec in result.recommendations:
        assert rec.estimate.total.low <= request.budget * 1.35


def test_relaxation_is_disclosed():
    tight = _request(budget=150_000, limit=8)
    result = recommend(CATALOG, _inputs(RATES), tight)
    over = [r for r in result.recommendations if r.estimate.total.low > tight.budget * 1.15]
    if over:
        assert "budget" in result.relaxed


def test_nothing_affordable_returns_guidance_not_an_empty_shrug():
    result = recommend(CATALOG, _inputs(RATES), _request(budget=1_000))
    assert result.recommendations == ()
    assert result.cheapest_feasible_budget and result.cheapest_feasible_budget > 1_000
    assert result.notes


def test_travel_time_limit_excludes_far_places_unless_relaxed():
    result = recommend(CATALOG, _inputs(RATES), _request(max_travel_hours=3, limit=2))
    assert "park" not in {r.destination.id for r in result.recommendations}


def test_rainy_month_lowers_outdoor_practicality():
    dry = recommend(CATALOG, _inputs(RATES), _request(month=1, limit=8))
    wet = recommend(CATALOG, _inputs(RATES), _request(month=8, limit=8))
    dry_score = {r.destination.id: r.practicality for r in dry.recommendations}
    wet_score = {r.destination.id: r.practicality for r in wet.recommendations}
    shared = set(dry_score) & set(wet_score) - {"museum"}
    assert shared and all(wet_score[i] <= dry_score[i] for i in shared)


def test_raising_budget_never_removes_an_affordable_result():
    low = recommend(CATALOG, _inputs(RATES), _request(budget=250_000, limit=8))
    high = recommend(CATALOG, _inputs(RATES), _request(budget=600_000, limit=8))
    assert {r.destination.id for r in low.recommendations} <= {r.destination.id for r in high.recommendations}


def test_no_interests_falls_back_to_popular_picks():
    result = recommend(CATALOG, _inputs(RATES), _request(preferences=Preferences({})))
    assert result.recommendations
    assert any("Popular picks" in n for n in result.notes)


def test_request_validation():
    with pytest.raises(ValueError):
        _request(budget=0)
    with pytest.raises(ValueError):
        _request(month=13)
    with pytest.raises(ValueError):
        Preferences({"teleportation": "love"})


def test_variety_never_beats_relevance():
    result = recommend(CATALOG, _inputs(RATES), _request(limit=3))
    ids = [r.destination.id for r in result.recommendations]
    assert "museum" not in ids and "beach" not in ids
    assert "hills" in ids


def test_unrelated_places_only_fill_empty_slots():
    result = recommend(CATALOG, _inputs(RATES), _request(limit=8))
    contents = [r.content for r in result.recommendations]
    relevant = [c for c in contents if c >= 0.15]
    assert contents[: len(relevant)] == relevant
