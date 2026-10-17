"""Mamdani fuzzy inference for trip practicality.

Inputs are normalised ratios, so there is no hard Naira cap anywhere:
- affordability: the user's budget divided by the trip's estimated cost
- season_fit: 0-1 suitability of the travel month (engine.season)
- travel_burden: share of the trip's waking hours spent on the road

Output: practicality 0-1. RULES below is the whole policy; it can be read and
argued with by someone who doesn't write code.

The rules are evaluated with numpy over every candidate at once (min for AND,
max for OR, 1-x for NOT, max aggregation, centroid defuzzification: the same
operators scikit-fuzzy's control API uses). scikit-fuzzy's own engine is kept
as `reference_infer` and the tests check both agree.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping, Sequence, Union

import numpy as np
import skfuzzy as fuzz

AFFORDABILITY_MAX = 3.0
BURDEN_MAX = 1.5
WAKING_HOURS_PER_DAY = 14.0

UNIVERSES = {
    "affordability": np.linspace(0, AFFORDABILITY_MAX, 301),
    "season_fit": np.linspace(0, 1, 101),
    "travel_burden": np.linspace(0, BURDEN_MAX, 151),
    "practicality": np.linspace(0, 1, 101),
}

# (variable, term) -> (shape, parameters)
TERMS: Mapping[tuple[str, str], tuple[str, Sequence[float]]] = {
    # Below 1.0 the trip costs more than the budget. The hard constraint already
    # removed anything far over, so "tight" covers the near misses.
    ("affordability", "tight"): ("trap", (0, 0, 0.85, 1.05)),
    ("affordability", "comfortable"): ("tri", (0.95, 1.3, 1.7)),
    ("affordability", "generous"): ("trap", (1.5, 1.9, 3.0, 3.0)),
    ("season_fit", "poor"): ("trap", (0, 0, 0.35, 0.5)),
    ("season_fit", "fair"): ("tri", (0.4, 0.6, 0.8)),
    ("season_fit", "ideal"): ("trap", (0.7, 0.85, 1, 1)),
    ("travel_burden", "light"): ("trap", (0, 0, 0.15, 0.3)),
    ("travel_burden", "moderate"): ("tri", (0.2, 0.4, 0.6)),
    ("travel_burden", "heavy"): ("trap", (0.5, 0.7, 1.5, 1.5)),
    ("practicality", "low"): ("trap", (0, 0, 0.25, 0.45)),
    ("practicality", "medium"): ("tri", (0.3, 0.5, 0.7)),
    ("practicality", "high"): ("trap", (0.55, 0.75, 1, 1)),
}

# An expression is a (variable, term) leaf or ("and"|"or", *exprs) / ("not", expr).
Expr = Union[tuple[str, str], tuple]


def AND(*exprs: Expr) -> Expr:
    return ("and", *exprs)


def OR(*exprs: Expr) -> Expr:
    return ("or", *exprs)


def NOT(expr: Expr) -> Expr:
    return ("not", expr)


def A(term: str) -> Expr:
    return ("affordability", term)


def S(term: str) -> Expr:
    return ("season_fit", term)


def B(term: str) -> Expr:
    return ("travel_burden", term)

RULES: tuple[tuple[Expr, str], ...] = (
    # Affordable, in season, easy to reach: the ideal case.
    (AND(A("generous"), S("ideal"), B("light")), "high"),
    (AND(A("comfortable"), S("ideal"), B("light")), "high"),
    (AND(A("generous"), S("fair"), B("light")), "high"),
    # One thing is merely OK.
    (AND(A("comfortable"), S("fair"), NOT(B("heavy"))), "medium"),
    (AND(OR(A("comfortable"), A("generous")), S("ideal"), B("moderate")), "medium"),
    (AND(A("generous"), S("fair"), B("moderate")), "medium"),
    # Money is tight but everything else lines up.
    (AND(A("tight"), S("ideal"), B("light")), "medium"),
    (AND(A("tight"), OR(NOT(S("ideal")), NOT(B("light")))), "low"),
    # Wrong season for this place.
    (S("poor"), "low"),
    # Most of the trip spent on the road.
    (AND(B("heavy"), NOT(A("generous"))), "low"),
    (AND(B("heavy"), A("generous"), S("ideal")), "medium"),
    (AND(B("heavy"), A("generous"), NOT(S("ideal"))), "low"),
)


def _mf(universe: np.ndarray, shape: str, params: Sequence[float]) -> np.ndarray:
    return fuzz.trapmf(universe, list(params)) if shape == "trap" else fuzz.trimf(universe, list(params))


@lru_cache(maxsize=None)
def _term_curve(variable: str, term: str) -> np.ndarray:
    shape, params = TERMS[(variable, term)]
    return _mf(UNIVERSES[variable], shape, params)


def _evaluate(expr: Expr, leaf: Callable[[str, str], np.ndarray]) -> np.ndarray:
    head = expr[0]
    if head == "and":
        return np.minimum.reduce([_evaluate(e, leaf) for e in expr[1:]])
    if head == "or":
        return np.maximum.reduce([_evaluate(e, leaf) for e in expr[1:]])
    if head == "not":
        return 1.0 - _evaluate(expr[1], leaf)
    return leaf(*expr)


def _clip(values: np.ndarray, upper: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, upper)


def practicality_many(
    affordability: Sequence[float], season_fit: Sequence[float], burden: Sequence[float]
) -> np.ndarray:
    """Practicality for many candidates in one pass."""
    inputs = {
        "affordability": _clip(affordability, AFFORDABILITY_MAX),
        "season_fit": _clip(season_fit, 1.0),
        "travel_burden": _clip(burden, BURDEN_MAX),
    }

    def leaf(variable: str, term: str) -> np.ndarray:
        return np.interp(inputs[variable], UNIVERSES[variable], _term_curve(variable, term))

    output = UNIVERSES["practicality"]
    aggregated = np.zeros((len(inputs["affordability"]), len(output)))
    for expr, consequent in RULES:
        strength = _evaluate(expr, leaf)
        aggregated = np.maximum(
            aggregated, np.minimum(strength[:, None], _term_curve("practicality", consequent)[None, :])
        )
    return np.round(_centroid(output, aggregated), 4)


def _centroid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centroid of piecewise-linear curves (one per row), as skfuzzy.defuzz computes it."""
    x1, x2 = x[:-1], x[1:]
    y1, y2 = y[:, :-1], y[:, 1:]
    areas = (x2 - x1) * (y1 + y2) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        centres = np.where(
            y1 + y2 > 0, x1 + (x2 - x1) * (y1 + 2 * y2) / (3 * (y1 + y2)), (x1 + x2) / 2
        )
    total = areas.sum(axis=1)
    if np.any(total == 0):
        # A gap in the rule base would make the score undefined; fail loudly.
        raise ValueError("No fuzzy rule fired for at least one input")
    return (areas * centres).sum(axis=1) / total


def practicality(affordability: float, season_fit: float, burden: float) -> float:
    return float(practicality_many([affordability], [season_fit], [burden])[0])


def reference_infer(affordability: float, season_fit: float, burden: float) -> float:
    """The same rules run through scikit-fuzzy's control engine (slow; tests only)."""
    from skfuzzy import control as ctrl

    variables = {
        name: (ctrl.Consequent if name == "practicality" else ctrl.Antecedent)(universe, name)
        for name, universe in UNIVERSES.items()
    }
    for (name, term), (shape, params) in TERMS.items():
        variables[name][term] = _mf(UNIVERSES[name], shape, params)

    def build(expr: Expr):
        head = expr[0]
        if head in ("and", "or"):
            parts = [build(e) for e in expr[1:]]
            combined = parts[0]
            for part in parts[1:]:
                combined = (combined & part) if head == "and" else (combined | part)
            return combined
        if head == "not":
            return ~build(expr[1])
        return variables[head][expr[1]]

    rules = [ctrl.Rule(build(expr), variables["practicality"][out]) for expr, out in RULES]
    simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    simulation.input["affordability"] = min(max(affordability, 0.0), AFFORDABILITY_MAX)
    simulation.input["season_fit"] = min(max(season_fit, 0.0), 1.0)
    simulation.input["travel_burden"] = min(max(burden, 0.0), BURDEN_MAX)
    simulation.compute()
    return float(simulation.output["practicality"])


def travel_burden(one_way_hours: float, days: int) -> float:
    """Fraction of the trip's waking hours spent travelling there and back."""
    return min((2 * one_way_hours) / (days * WAKING_HOURS_PER_DAY), BURDEN_MAX)


def budget_band(daily_spend_per_person: float) -> str:
    """Low/Medium/High label for the UI, served by the API so the frontend never re-implements it.

    Bands are in Naira per person per day. They are provisional: the food and
    transport parts line up with the NBS-based tier estimates, but they must be
    recalibrated once verified accommodation rates exist.
    """
    universe = np.linspace(0, 200_000, 2001)
    value = min(max(daily_spend_per_person, 0.0), 200_000.0)
    memberships = {
        "Low": fuzz.interp_membership(universe, fuzz.trapmf(universe, [0, 0, 20_000, 35_000]), value),
        "Medium": fuzz.interp_membership(universe, fuzz.trimf(universe, [25_000, 50_000, 90_000]), value),
        "High": fuzz.interp_membership(
            universe, fuzz.trapmf(universe, [70_000, 110_000, 200_000, 200_000]), value
        ),
    }
    return max(memberships, key=lambda band: (memberships[band], band == "Medium"))
