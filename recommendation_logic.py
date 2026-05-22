"""Backward-compatible entry point — logic lives in services.recommendation_engine."""
from services.recommendation_engine import classify_budget, get_recommendations

__all__ = ["classify_budget", "get_recommendations"]
