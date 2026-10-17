"""Trip cost estimation from sourced inputs.

Every figure is a range with a confidence level and a record of what it was
built from. A component with no data is reported as missing, never filled in.
See docs/ENGINE.md for the assumptions and why they were chosen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Mapping, Optional, Sequence

# --- Documented assumptions (each is a candidate for calibration) -----------

# Straight-line distance understates road distance; 1.3 is a common detour factor.
ROAD_DETOUR_FACTOR = 1.3
# NBS publishes one average intercity bus fare per state. We treat it as the fare
# for a typical ~300 km route and scale by distance, clamped so short hops and
# cross-country trips stay plausible.
INTERCITY_REFERENCE_KM = 300.0
INTERCITY_SCALE_BOUNDS = (0.5, 3.0)
AVERAGE_ROAD_SPEED_KMH = 55.0
LOCAL_TRIPS_PER_DAY = (2, 6)
MEALS_PER_DAY = 3
PEOPLE_PER_ROOM = 2

# One plate of rice and stew with beef, priced from NBS zonal food prices
# (all per kg except vegetable oil, per 75cl bottle).
MEAL_BASKET: Mapping[str, float] = {
    "Rice Local, short-Grained": 0.20,
    "Beef Boneless": 0.10,
    "Tomatoes, fresh": 0.10,
    "Onions, fresh": 0.03,
    "Vegetable Oil, 75cl": 0.04,
}
# Ingredient cost -> price paid eating out, per comfort tier (low, high multiplier).
EATING_OUT_MARKUP: Mapping[str, tuple[float, float]] = {
    "budget": (1.6, 2.4),
    "mid": (3.0, 5.0),
    "premium": (6.0, 10.0),
}
# Observations older than this lower the confidence of the estimate. NBS series
# publish with a lag of several months, so they get a longer window than our own
# curated observations.
FRESH_DAYS = 120
OFFICIAL_SERIES_FRESH_DAYS = 180
MIN_OBSERVATIONS_FOR_HIGH = 3


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @staticmethod
    def worst(values: Sequence["Confidence"]) -> "Confidence":
        order = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
        return min(values, key=order.index) if values else Confidence.LOW


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < self.low:
            raise ValueError(f"Invalid range {self.low}..{self.high}")

    def __add__(self, other: "Range") -> "Range":
        return Range(self.low + other.low, self.high + other.high)

    def scale(self, factor: float) -> "Range":
        return Range(self.low * factor, self.high * factor)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2


ZERO = Range(0.0, 0.0)


@dataclass(frozen=True)
class Observation:
    amount: float
    observed_on: date


@dataclass(frozen=True)
class CostInputs:
    """Everything the estimator needs, loaded once per catalog snapshot."""

    as_of: date
    # state iso -> mode -> (fare, observed month)
    fares: Mapping[str, Mapping[str, Observation]]
    # zone -> item -> (price, month)
    food_prices: Mapping[str, Mapping[str, Observation]]
    # (scope, key, tier) -> observations; scope is "destination", "city" or "state"
    accommodation: Mapping[tuple[str, str, str], Sequence[Observation]] = field(default_factory=dict)
    # destination id -> per-person entry fee observations
    entry_fees: Mapping[str, Sequence[Observation]] = field(default_factory=dict)


@dataclass(frozen=True)
class Place:
    state_iso: str
    zone: str
    lon: float
    lat: float


@dataclass(frozen=True)
class TripRequest:
    origin: Place
    days: int
    travellers: int
    tier: str

    def __post_init__(self) -> None:
        if self.days < 1 or self.travellers < 1:
            raise ValueError("A trip needs at least one day and one traveller")
        if self.tier not in EATING_OUT_MARKUP:
            raise ValueError(f"Unknown tier {self.tier}")

    @property
    def nights(self) -> int:
        return self.days - 1

    @property
    def rooms(self) -> int:
        return math.ceil(self.travellers / PEOPLE_PER_ROOM)


@dataclass(frozen=True)
class Component:
    name: str
    cost: Range
    confidence: Confidence
    basis: str
    observed_on: Optional[date] = None


@dataclass(frozen=True)
class CostEstimate:
    total: Range
    components: tuple[Component, ...]
    missing: tuple[str, ...]
    confidence: Confidence
    road_km: float
    travel_hours: float

    @property
    def complete(self) -> bool:
        return not self.missing


def haversine_km(a_lon: float, a_lat: float, b_lon: float, b_lat: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(a_lat), math.radians(b_lat)
    d_phi = math.radians(b_lat - a_lat)
    d_lambda = math.radians(b_lon - a_lon)
    h = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def _freshness(observed_on: date, as_of: date, window_days: int = FRESH_DAYS) -> Confidence:
    return Confidence.HIGH if (as_of - observed_on).days <= window_days else Confidence.LOW


def _official_freshness(observed_on: date, as_of: date) -> Confidence:
    return _freshness(observed_on, as_of, OFFICIAL_SERIES_FRESH_DAYS)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def intercity_transport(
    inputs: CostInputs, trip: TripRequest, destination: Place, road_km: float
) -> Optional[Component]:
    if trip.origin.state_iso == destination.state_iso:
        return Component("intercity_transport", ZERO, Confidence.HIGH, "Same state: no intercity travel")
    fare = inputs.fares.get(trip.origin.state_iso, {}).get("bus")
    if fare is None:
        return None
    low, high = INTERCITY_SCALE_BOUNDS
    scale = min(max(road_km / INTERCITY_REFERENCE_KM, low), high)
    one_way = fare.amount * scale
    # Return trip for everyone, with +/-20% around the distance-scaled fare.
    cost = Range(one_way * 0.8, one_way * 1.2).scale(2 * trip.travellers)
    confidence = Confidence.worst([Confidence.MEDIUM, _official_freshness(fare.observed_on, inputs.as_of)])
    return Component(
        "intercity_transport",
        cost,
        confidence,
        f"NBS average intercity bus fare from origin state, scaled to ~{road_km:.0f} km by road",
        fare.observed_on,
    )


def local_transport(inputs: CostInputs, trip: TripRequest, destination: Place) -> Optional[Component]:
    # The loader keys in-city bus fares as "local_bus" to keep them apart from intercity "bus".
    fare = inputs.fares.get(destination.state_iso, {}).get("local_bus")
    if fare is None:
        return None
    low_trips, high_trips = LOCAL_TRIPS_PER_DAY
    cost = Range(fare.amount * low_trips, fare.amount * high_trips).scale(trip.days * trip.travellers)
    return Component(
        "local_transport",
        cost,
        Confidence.worst([Confidence.MEDIUM, _official_freshness(fare.observed_on, inputs.as_of)]),
        f"NBS in-city bus fare, {low_trips}-{high_trips} trips a day",
        fare.observed_on,
    )


def meal_cost(inputs: CostInputs, zone: str, tier: str) -> Optional[tuple[Range, date]]:
    prices = inputs.food_prices.get(zone, {})
    if not all(item in prices for item in MEAL_BASKET):
        return None
    ingredients = sum(prices[item].amount * qty for item, qty in MEAL_BASKET.items())
    oldest = min(prices[item].observed_on for item in MEAL_BASKET)
    low, high = EATING_OUT_MARKUP[tier]
    return Range(ingredients * low, ingredients * high), oldest


def food(inputs: CostInputs, trip: TripRequest, destination: Place) -> Optional[Component]:
    meal = meal_cost(inputs, destination.zone, trip.tier)
    if meal is None:
        return None
    per_meal, observed_on = meal
    cost = per_meal.scale(MEALS_PER_DAY * trip.days * trip.travellers)
    return Component(
        "food",
        cost,
        # Modelled from ingredient prices, so never better than medium.
        Confidence.worst([Confidence.MEDIUM, _official_freshness(observed_on, inputs.as_of)]),
        f"NBS zonal food prices for a rice-and-stew plate, {trip.tier} eating-out markup",
        observed_on,
    )


def accommodation(
    inputs: CostInputs,
    trip: TripRequest,
    destination_id: str,
    destination: Place,
    city_key: Optional[str],
) -> Optional[Component]:
    if trip.nights == 0:
        return Component("accommodation", ZERO, Confidence.HIGH, "Day trip: no overnight stay")
    scopes = [("destination", destination_id), ("city", city_key), ("state", destination.state_iso)]
    for scope, key in scopes:
        if key is None:
            continue
        observations = inputs.accommodation.get((scope, key, trip.tier)) or ()
        if not observations:
            continue
        amounts = [o.amount for o in observations]
        newest = max(o.observed_on for o in observations)
        per_night = Range(_percentile(amounts, 0.25), _percentile(amounts, 0.75))
        confidence = _freshness(newest, inputs.as_of)
        if len(observations) < MIN_OBSERVATIONS_FOR_HIGH or scope == "state":
            confidence = Confidence.worst([confidence, Confidence.MEDIUM])
        return Component(
            "accommodation",
            per_night.scale(trip.nights * trip.rooms),
            confidence,
            f"{len(observations)} verified {trip.tier} rate(s) at {scope} level",
            newest,
        )
    return None


def entry_fees(inputs: CostInputs, trip: TripRequest, destination_id: str) -> Component:
    observations = inputs.entry_fees.get(destination_id) or ()
    if not observations:
        # Unknown fees are flagged in the basis text but don't block the estimate.
        return Component("entry_fees", ZERO, Confidence.LOW, "Entry fee not yet verified")
    amounts = [o.amount for o in observations]
    newest = max(o.observed_on for o in observations)
    cost = Range(min(amounts), max(amounts)).scale(trip.travellers)
    return Component("entry_fees", cost, _freshness(newest, inputs.as_of), "Verified entry fee", newest)


def estimate_trip(
    inputs: CostInputs,
    trip: TripRequest,
    destination_id: str,
    destination: Place,
    city_key: Optional[str] = None,
) -> CostEstimate:
    straight_km = haversine_km(trip.origin.lon, trip.origin.lat, destination.lon, destination.lat)
    road_km = straight_km * ROAD_DETOUR_FACTOR
    builders = {
        "intercity_transport": lambda: intercity_transport(inputs, trip, destination, road_km),
        "local_transport": lambda: local_transport(inputs, trip, destination),
        "food": lambda: food(inputs, trip, destination),
        "accommodation": lambda: accommodation(inputs, trip, destination_id, destination, city_key),
        "entry_fees": lambda: entry_fees(inputs, trip, destination_id),
    }
    components: list[Component] = []
    missing: list[str] = []
    for name, build in builders.items():
        component = build()
        if component is None:
            missing.append(name)
        else:
            components.append(component)

    total = ZERO
    for component in components:
        total = total + component.cost
    scored = [c.confidence for c in components if c.name != "entry_fees"]
    confidence = Confidence.LOW if missing else Confidence.worst(scored)
    return CostEstimate(
        total=total,
        components=tuple(components),
        missing=tuple(missing),
        confidence=confidence,
        road_km=road_km,
        travel_hours=road_km / AVERAGE_ROAD_SPEED_KMH,
    )
