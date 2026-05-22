from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import skfuzzy as fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.data_cache import get_catalog_df
from services.vocabulary import (
    activity_filter_tokens,
    destination_filter_tokens,
    normalize_activity_type,
    normalize_destination_type,
)

logger = logging.getLogger(__name__)


def classify_budget(budget: float) -> str:
    budget_range = np.arange(0, 50001, 1)
    low_budget = fuzz.trapmf(budget_range, [0, 0, 12000, 18000])
    medium_budget = fuzz.trimf(budget_range, [15000, 25000, 35000])
    high_budget = fuzz.trapmf(budget_range, [32000, 40000, 50000, 50000])

    low_score = fuzz.interp_membership(budget_range, low_budget, budget)
    medium_score = fuzz.interp_membership(budget_range, medium_budget, budget)
    high_score = fuzz.interp_membership(budget_range, high_budget, budget)

    if low_score >= medium_score and low_score >= high_score:
        return "Low"
    if medium_score >= low_score and medium_score >= high_score:
        return "Medium"
    return "High"


def get_budget_match_score(destination_cost: float, user_budget: float) -> float:
    if destination_cost > user_budget:
        ratio = user_budget / destination_cost
        return ratio ** 1.5
    ratio = 0.8 + 0.2 * (destination_cost / user_budget)
    return min(1.0, ratio)


def _apply_prefilter(
    df,
    destination_type: Optional[str],
    activity_type: Optional[str],
):
    candidates = df.copy()
    filtered = candidates

    dest_tokens = destination_filter_tokens(destination_type)
    if dest_tokens:
        mask = filtered["destination_type_keywords"].apply(
            lambda text: any(token in text for token in dest_tokens)
        )
        subset = filtered[mask]
        if not subset.empty:
            filtered = subset

    activity_tokens = activity_filter_tokens(activity_type)
    if activity_tokens:
        mask = filtered["activities_keywords"].apply(
            lambda text: any(token in text for token in activity_tokens)
        )
        subset = filtered[mask]
        if not subset.empty:
            filtered = subset

    if len(filtered) >= 5:
        return filtered
    return candidates


def get_recommendations(
    budget: float,
    destination_type: Optional[str] = None,
    activity_type: Optional[str] = None,
) -> Dict[str, Any]:
    df = get_catalog_df()
    budget = float(budget)

    if df.empty:
        return {"user_budget_category": classify_budget(budget), "recommendations": []}

    candidates_df = _apply_prefilter(df, destination_type, activity_type)
    budget_category = classify_budget(budget)

    norm_destination = normalize_destination_type(destination_type)
    norm_activity = normalize_activity_type(activity_type)

    user_preferences = {
        "destination_type": norm_destination,
        "activities": norm_activity,
    }
    if not user_preferences["destination_type"] and not user_preferences["activities"]:
        user_preferences["activities"] = "nature adventure leisure"

    feature_similarities: Dict[str, np.ndarray] = {}
    for feature in ["destination_type", "activities"]:
        if user_preferences[feature]:
            combined_texts = list(candidates_df[feature].fillna("")) + [user_preferences[feature]]
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                matrix = vectorizer.fit_transform(combined_texts)
                user_vector = matrix[-1]
                destination_vectors = matrix[:-1]
                feature_similarities[feature] = cosine_similarity(user_vector, destination_vectors).flatten()
            except Exception:
                feature_similarities[feature] = np.ones(len(candidates_df))
        else:
            feature_similarities[feature] = np.ones(len(candidates_df))

    recommendations: List[Dict[str, Any]] = []
    for index in range(len(candidates_df)):
        row = candidates_df.iloc[index]
        try:
            avg_cost = float(row["avg_cost_per_day"])
            budget_score = get_budget_match_score(avg_cost, budget)
            content_score = sum(
                0.5 * feature_similarities[feature][index]
                for feature in feature_similarities
            )
            final_score = 0.4 * budget_score + 0.6 * content_score

            recommendations.append(
                {
                    "destination": str(row["destination"]),
                    "state": str(row["state"]),
                    "city": str(row["city"]),
                    "destination_type": str(row["destination_type"]),
                    "activities": str(row["activities"]),
                    "climate": str(row["climate"]),
                    "avg_cost_per_day": avg_cost,
                    "best_season": str(row["best_season"]),
                    "accommodation_type": str(row["accommodation_type"]),
                    "nearby_hotel": str(row["nearby_hotel"]),
                    "hotel_price_range": str(row["hotel_price_range"]),
                    "feeding_cost_range": str(row["feeding_cost_range"]),
                    "necessities_range": str(row["necessities_range"]),
                    "budget_category": classify_budget(avg_cost),
                    "score": float(final_score),
                }
            )
        except Exception as exc:
            logger.warning("Skipping row %s: %s", index, exc)

    recommendations_sorted = sorted(recommendations, key=lambda item: item["score"], reverse=True)

    state_counts: Dict[str, int] = {}
    diversified: List[Dict[str, Any]] = []
    for rec in recommendations_sorted:
        state = rec["state"]
        if state_counts.get(state, 0) < 2:
            state_counts[state] = state_counts.get(state, 0) + 1
            diversified.append(rec)
        if len(diversified) >= 5:
            break

    if len(diversified) < 5:
        for rec in recommendations_sorted:
            if rec not in diversified:
                diversified.append(rec)
            if len(diversified) >= 5:
                break

    return {"user_budget_category": budget_category, "recommendations": diversified}
