from __future__ import annotations

from datetime import date

import pytest

from engine.costs import (
    MEAL_BASKET,
    Confidence,
    CostInputs,
    Observation,
    Place,
    Range,
    TripRequest,
    estimate_trip,
    haversine_km,
    meal_cost,
)

TODAY = date(2026, 10, 1)
MAY = date(2026, 5, 1)

LAGOS = Place("NG-LA", "south_west", 3.38, 6.52)
OBUDU = Place("NG-CR", "south_south", 9.36, 6.39)
LEKKI = Place("NG-LA", "south_west", 3.53, 6.44)

FOOD = {item: Observation(1000.0, MAY) for item in MEAL_BASKET}


def _inputs(**overrides) -> CostInputs:
    values = dict(
        as_of=TODAY,
        fares={
            "NG-LA": {"bus": Observation(10000, MAY), "local_bus": Observation(1500, MAY)},
            "NG-CR": {"bus": Observation(10000, MAY), "local_bus": Observation(1200, MAY)},
        },
        food_prices={"south_west": FOOD, "south_south": FOOD},
        accommodation={
            ("destination", "obudu", "mid"): [
                Observation(40000, date(2026, 9, 1)),
                Observation(50000, date(2026, 9, 1)),
                Observation(60000, date(2026, 9, 1)),
            ],
            ("state", "NG-LA", "budget"): [Observation(20000, date(2026, 9, 1))],
        },
        entry_fees={"obudu": [Observation(5000, date(2026, 9, 1))]},
    )
    values.update(overrides)
    return CostInputs(**values)


def test_range_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Range(10, 5)


def test_trip_request_validates_inputs():
    with pytest.raises(ValueError):
        TripRequest(LAGOS, days=0, travellers=1, tier="mid")
    with pytest.raises(ValueError):
        TripRequest(LAGOS, days=2, travellers=1, tier="luxury")


def test_haversine_lagos_to_obudu_is_about_660km():
    assert 640 < haversine_km(LAGOS.lon, LAGOS.lat, OBUDU.lon, OBUDU.lat) < 680


def test_meal_cost_needs_every_basket_item():
    partial = {"south_west": {"Beef Boneless": Observation(7000, MAY)}}
    assert meal_cost(_inputs(food_prices=partial), "south_west", "budget") is None


def test_full_trip_estimate_has_every_component():
    trip = TripRequest(LAGOS, days=3, travellers=2, tier="mid")
    estimate = estimate_trip(_inputs(), trip, "obudu", OBUDU)

    assert estimate.complete
    names = {c.name for c in estimate.components}
    assert names == {"intercity_transport", "local_transport", "food", "accommodation", "entry_fees"}
    accommodation = next(c for c in estimate.components if c.name == "accommodation")
    # 2 nights, 1 room, p25..p75 of 40k/50k/60k
    assert accommodation.cost == Range(90000, 110000)
    entry = next(c for c in estimate.components if c.name == "entry_fees")
    assert entry.cost == Range(10000, 10000)
    assert estimate.total.low < estimate.total.high
    # Food is modelled from ingredients, so the whole estimate can't be "high".
    assert estimate.confidence == Confidence.MEDIUM


def test_intercity_fare_is_scaled_by_distance_and_clamped():
    trip = TripRequest(LAGOS, days=1, travellers=1, tier="budget")
    far = estimate_trip(_inputs(), trip, "obudu", OBUDU)
    transport = next(c for c in far.components if c.name == "intercity_transport")
    # ~860 km by road -> ~2.87x the 300 km reference, return trip
    assert 2 * 10000 * 2.5 * 0.8 < transport.cost.low < 2 * 10000 * 3.0 * 0.8


def test_same_state_trip_has_no_intercity_cost():
    trip = TripRequest(LAGOS, days=2, travellers=1, tier="budget")
    estimate = estimate_trip(_inputs(), trip, "lekki", LEKKI)
    transport = next(c for c in estimate.components if c.name == "intercity_transport")
    assert transport.cost == Range(0, 0)


def test_day_trip_needs_no_accommodation():
    trip = TripRequest(LAGOS, days=1, travellers=3, tier="premium")
    estimate = estimate_trip(_inputs(), trip, "somewhere", OBUDU)
    assert "accommodation" not in estimate.missing


def test_missing_accommodation_is_reported_not_invented():
    trip = TripRequest(LAGOS, days=2, travellers=1, tier="premium")
    estimate = estimate_trip(_inputs(), trip, "obudu", OBUDU)
    assert estimate.missing == ("accommodation",)
    assert estimate.confidence == Confidence.LOW
    assert not estimate.complete


def test_state_level_or_thin_accommodation_caps_confidence():
    trip = TripRequest(LAGOS, days=2, travellers=1, tier="budget")
    estimate = estimate_trip(_inputs(), trip, "lekki", LEKKI)
    accommodation = next(c for c in estimate.components if c.name == "accommodation")
    assert accommodation.confidence == Confidence.MEDIUM


def test_stale_fares_drop_confidence():
    old = date(2025, 1, 1)
    fares = {"NG-LA": {"bus": Observation(10000, old), "local_bus": Observation(1500, old)},
             "NG-CR": {"bus": Observation(10000, old), "local_bus": Observation(1200, old)}}
    trip = TripRequest(LAGOS, days=3, travellers=2, tier="mid")
    estimate = estimate_trip(_inputs(fares=fares), trip, "obudu", OBUDU)
    assert estimate.confidence == Confidence.LOW


def test_more_travellers_and_rooms_scale_costs():
    one = estimate_trip(_inputs(), TripRequest(LAGOS, 3, 1, "mid"), "obudu", OBUDU)
    three = estimate_trip(_inputs(), TripRequest(LAGOS, 3, 3, "mid"), "obudu", OBUDU)
    acc_one = next(c for c in one.components if c.name == "accommodation").cost
    acc_three = next(c for c in three.components if c.name == "accommodation").cost
    assert acc_three == acc_one.scale(2)  # 3 travellers -> 2 rooms
    assert three.total.low > one.total.low
