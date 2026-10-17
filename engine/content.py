"""Content-based matching over the shared interest taxonomy.

A destination is a weighted vector over the taxonomy; so is the user, built
from the interests they marked "love" or "like" and their travel pace. Cosine
similarity between the two is the content score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from engine.taxonomy import TAXONOMY

VOCABULARY: tuple[str, ...] = tuple(tag.slug for tag in TAXONOMY)
_INDEX = {slug: i for i, slug in enumerate(VOCABULARY)}
INTEREST_SLUGS = frozenset(tag.slug for tag in TAXONOMY if tag.is_interest)

LOVE_WEIGHT = 1.0
LIKE_WEIGHT = 0.6
# Pace nudges the profile towards slower or busier places without overriding interests.
PACE_TAGS: Mapping[str, Mapping[str, float]] = {
    "relaxed": {"relaxation": 0.5},
    "balanced": {},
    "packed": {"adventure": 0.4, "hiking": 0.3},
}
FAMILY_TAG_WEIGHT = 0.5
# A destination tag at or above this weight counts as a real match in explanations.
MATCH_THRESHOLD = 0.5


@dataclass(frozen=True)
class Preferences:
    interests: Mapping[str, str]  # slug -> "love" | "like"
    pace: str = "balanced"
    with_children: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.interests) - INTEREST_SLUGS
        if unknown:
            raise ValueError(f"Unknown interests: {sorted(unknown)}")
        if any(level not in ("love", "like") for level in self.interests.values()):
            raise ValueError("Interest level must be 'love' or 'like'")
        if self.pace not in PACE_TAGS:
            raise ValueError(f"Unknown pace {self.pace}")


def tag_vector(tags: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros(len(VOCABULARY))
    for slug, weight in tags.items():
        index = _INDEX.get(slug)
        if index is not None:
            vector[index] = float(weight)
    return vector


def profile_vector(preferences: Preferences) -> np.ndarray:
    weights: dict[str, float] = {}
    for slug, level in preferences.interests.items():
        weights[slug] = LOVE_WEIGHT if level == "love" else LIKE_WEIGHT
    for slug, weight in PACE_TAGS[preferences.pace].items():
        weights[slug] = max(weights.get(slug, 0.0), weight)
    if preferences.with_children:
        weights["family-friendly"] = max(weights.get("family-friendly", 0.0), FAMILY_TAG_WEIGHT)
    return tag_vector(weights)


def cosine_many(profile: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one profile against every row; 0 where either side is empty."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(profile)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = (matrix @ profile) / norms
    return np.nan_to_num(scores, nan=0.0)


def matched_interests(preferences: Preferences, tags: Mapping[str, float]) -> list[str]:
    """User interests this destination genuinely offers, loved ones first."""
    matches = [slug for slug in preferences.interests if tags.get(slug, 0.0) >= MATCH_THRESHOLD]
    return sorted(matches, key=lambda s: (preferences.interests[s] != "love", -tags[s]))


def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    unit = matrix / norms[:, None]
    return unit @ unit.T


def has_interests(preferences: Preferences) -> bool:
    return bool(preferences.interests)


def labels(slugs: Sequence[str]) -> list[str]:
    by_slug = {tag.slug: tag.label for tag in TAXONOMY}
    return [by_slug[s] for s in slugs]
