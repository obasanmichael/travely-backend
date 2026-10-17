from __future__ import annotations

from engine.season import MonthClimate, best_months, month_score, outdoor_share, season_fit

# Yankari-like: dry Nov-Mar, very wet Jul-Sep.
SAVANNA = [
    MonthClimate(m, t, r)
    for m, t, r in [
        (1, 25, 0), (2, 27, 2), (3, 30, 6), (4, 31, 59), (5, 29, 91), (6, 27, 109),
        (7, 25, 260), (8, 25, 300), (9, 26, 160), (10, 27, 78), (11, 26, 4), (12, 25, 1),
    ]
]
PARK = {"nature": 1.0, "wildlife": 0.9, "hiking": 0.6}
MUSEUM = {"museums": 1.0, "history": 0.8}


def test_outdoor_places_score_dry_months_higher():
    assert month_score(SAVANNA[0], PARK) > month_score(SAVANNA[7], PARK)


def test_indoor_places_barely_care_about_rain():
    dry, wet = month_score(SAVANNA[0], MUSEUM), month_score(SAVANNA[7], MUSEUM)
    assert abs(dry - wet) < 0.05


def test_untagged_places_lean_outdoor():
    assert outdoor_share({"history": 1.0}) == 0.6


def test_season_fit_for_month_and_flexible():
    wet_score, best = season_fit(SAVANNA, PARK, month=8)
    flexible_score, flexible_best = season_fit(SAVANNA, PARK, month=None)
    assert wet_score < 0.5
    assert best in {1, 2, 3, 11, 12}
    assert flexible_score >= wet_score and flexible_best == best


def test_incomplete_climate_is_neutral_not_guessed():
    assert season_fit(SAVANNA[:6], PARK, month=3) == (0.5, None)
    assert best_months(SAVANNA[:6], PARK) == []


def test_best_months_are_the_dry_season():
    months = best_months(SAVANNA, PARK)
    assert set(months) <= {11, 12, 1, 2, 3}
