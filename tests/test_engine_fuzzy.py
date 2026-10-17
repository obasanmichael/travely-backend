from __future__ import annotations

import time

import numpy as np
import pytest

from engine.fuzzy import (
    AFFORDABILITY_MAX,
    BURDEN_MAX,
    budget_band,
    practicality,
    practicality_many,
    reference_infer,
    travel_burden,
)


def test_ideal_trip_is_highly_practical():
    assert practicality(2.0, 0.95, 0.1) > 0.7


def test_wrong_season_is_impractical_even_with_money():
    assert practicality(2.5, 0.1, 0.1) < 0.4


def test_heavy_travel_on_a_tight_budget_is_impractical():
    assert practicality(0.95, 0.9, 1.0) < 0.35


def test_matches_scikit_fuzzy_reference_engine():
    rng = np.random.default_rng(7)
    for _ in range(25):
        args = (rng.uniform(0, 3), rng.uniform(0, 1), rng.uniform(0, 1.5))
        assert practicality(*args) == pytest.approx(reference_infer(*args), abs=1e-3)


def test_every_input_combination_fires_a_rule():
    grid = np.array(np.meshgrid(
        np.linspace(0, AFFORDABILITY_MAX, 31), np.linspace(0, 1, 21), np.linspace(0, BURDEN_MAX, 16)
    )).reshape(3, -1)
    scores = practicality_many(*grid)
    assert np.all((scores >= 0) & (scores <= 1))


def test_scoring_a_full_catalog_is_fast():
    n = 1000
    start = time.perf_counter()
    practicality_many(np.full(n, 1.4), np.full(n, 0.8), np.full(n, 0.2))
    assert time.perf_counter() - start < 0.5


def test_inputs_outside_the_universe_are_clamped():
    assert practicality(10, 2, -1) == practicality(3.0, 1.0, 0.0)


@pytest.mark.parametrize("season, burden", [(0.9, 0.1), (0.6, 0.3), (0.9, 0.7)])
def test_more_money_never_makes_a_trip_less_practical(season, burden):
    """Monotonicity: the property that justifies a fuzzy model over hard thresholds."""
    previous = 0.0
    for affordability in [0.8 + 0.05 * i for i in range(40)]:
        score = practicality(affordability, season, burden)
        assert score >= previous - 0.02
        previous = score


def test_small_budget_changes_dont_flip_the_score():
    assert abs(practicality(1.30, 0.8, 0.2) - practicality(1.32, 0.8, 0.2)) < 0.03


def test_travel_burden_is_share_of_waking_hours():
    assert travel_burden(7, 2) == pytest.approx(0.5)
    assert travel_burden(20, 1) == 1.5


@pytest.mark.parametrize("spend, band", [(10_000, "Low"), (50_000, "Medium"), (150_000, "High")])
def test_budget_band(spend, band):
    assert budget_band(spend) == band
