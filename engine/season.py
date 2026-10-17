"""How good a month is for visiting a destination, from its climate normals.

Rain matters for outdoor places (trails, beaches, parks) and barely at all for
indoor ones (museums, malls), so the score blends the two by the destination's
tag weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

OUTDOOR_TAGS = frozenset({
    "nature", "waterfalls", "hills-mountains", "lakes-rivers", "beaches", "caves", "wildlife",
    "hiking", "swimming", "boating", "birdwatching", "camping", "photography", "adventure",
    "amusement",
})
INDOOR_TAGS = frozenset({"museums", "art", "shopping", "food", "nightlife"})

# Monthly rainfall at or below DRY_MM is ideal for outdoor visits; at or above
# WET_MM roads, trails and beaches are at their worst.
DRY_MM = 50.0
WET_MM = 250.0
WETTEST_SCORE = 0.2
COMFORT_TEMP_C = (18.0, 30.0)
HOT_LIMIT_C = 36.0
HOTTEST_SCORE = 0.6


@dataclass(frozen=True)
class MonthClimate:
    month: int
    temp_mean_c: float
    precip_mm: float


def _rain_score(precip_mm: float) -> float:
    if precip_mm <= DRY_MM:
        return 1.0
    if precip_mm >= WET_MM:
        return WETTEST_SCORE
    fraction = (precip_mm - DRY_MM) / (WET_MM - DRY_MM)
    return 1.0 - fraction * (1.0 - WETTEST_SCORE)


def _heat_score(temp_c: float) -> float:
    low, high = COMFORT_TEMP_C
    if low <= temp_c <= high:
        return 1.0
    if temp_c > high:
        fraction = min((temp_c - high) / (HOT_LIMIT_C - high), 1.0)
        return 1.0 - fraction * (1.0 - HOTTEST_SCORE)
    # Nigeria rarely sees cold monthly means; treat cool highlands as mildly less comfortable.
    return max(0.8, 1.0 - (low - temp_c) * 0.04)


def outdoor_share(tags: Mapping[str, float]) -> float:
    outdoor = sum(w for t, w in tags.items() if t in OUTDOOR_TAGS)
    indoor = sum(w for t, w in tags.items() if t in INDOOR_TAGS)
    if outdoor + indoor == 0:
        # Mixed or unknown places (history, culture, palaces): mostly outdoors in practice.
        return 0.6
    return outdoor / (outdoor + indoor)


def month_score(climate: MonthClimate, tags: Mapping[str, float]) -> float:
    outdoor = outdoor_share(tags)
    outdoor_fit = _rain_score(climate.precip_mm) * _heat_score(climate.temp_mean_c)
    indoor_fit = _heat_score(climate.temp_mean_c) ** 0.5
    return round(outdoor * outdoor_fit + (1 - outdoor) * indoor_fit, 3)


def season_fit(
    climate: Sequence[MonthClimate],
    tags: Mapping[str, float],
    month: Optional[int],
) -> tuple[float, Optional[int]]:
    """(score for the travel month, best month). With no month (flexible), use the best month."""
    by_month = {c.month: c for c in climate}
    if len(by_month) != 12:
        # Without a full year of data we can't judge; neutral rather than invented.
        return 0.5, None
    scores = {m: month_score(c, tags) for m, c in by_month.items()}
    best = max(scores, key=lambda m: (scores[m], -m))
    if month is None:
        return scores[best], best
    return scores[month], best


def best_months(climate: Sequence[MonthClimate], tags: Mapping[str, float], tolerance: float = 0.05) -> list[int]:
    """Months scoring within `tolerance` of the best: what the UI shows as 'best time to visit'."""
    if len({c.month for c in climate}) != 12:
        return []
    scores = {c.month: month_score(c, tags) for c in climate}
    top = max(scores.values())
    return sorted(m for m, s in scores.items() if s >= top - tolerance)
