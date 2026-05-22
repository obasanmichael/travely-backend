from __future__ import annotations

from typing import Dict, List, Optional

# Quiz UI values -> CSV-aligned search tokens (lowercase)
DESTINATION_TYPE_MAP: Dict[str, List[str]] = {
    "Nature/Adventure": ["nature"],
    "Nature/Leisure": ["nature", "leisure"],
    "Wildlife/Safari": ["nature", "wildlife", "safari"],
    "Architecture/Adventure": ["urban leisure", "cultural"],
    "Leisure/Urban": ["urban leisure"],
    "Nature/Relaxation": ["nature", "leisure"],
    "Cultural/Adventure": ["cultural"],
    "Historical/Nature": ["historical", "nature"],
    "Historical/Cultural": ["historical", "cultural"],
    "Leisure/Resort": ["urban leisure", "leisure"],
}

ACTIVITY_TYPE_MAP: Dict[str, List[str]] = {
    "Hiking": ["hiking", "trekking", "walk"],
    "Swimming": ["swimming", "water"],
    "Safari": ["safari", "wildlife"],
    "Picnic": ["picnic", "relaxation"],
    "Tour": ["tour", "cultural"],
    "Relaxation": ["relaxation", "spa", "wellness"],
    "Shopping": ["shopping", "market"],
    "Boating": ["boating", "water"],
    "Photography": ["photography", "sightseeing"],
    "Horse Riding": ["horse", "riding", "adventure"],
}


def normalize_destination_type(value: Optional[str]) -> str:
    if not value:
        return ""
    tokens = DESTINATION_TYPE_MAP.get(value.strip(), [])
    if tokens:
        return " ".join(tokens)
    return value.lower().strip()


def normalize_activity_type(value: Optional[str]) -> str:
    if not value:
        return ""
    tokens = ACTIVITY_TYPE_MAP.get(value.strip(), [])
    if tokens:
        return " ".join(tokens)
    return value.lower().strip()


def destination_filter_tokens(value: Optional[str]) -> List[str]:
    """Tokens used for strict pre-filtering against catalog destination_type."""
    if not value:
        return []
    mapped = DESTINATION_TYPE_MAP.get(value.strip())
    if mapped:
        return mapped
    return [value.lower().strip()]


def activity_filter_tokens(value: Optional[str]) -> List[str]:
    if not value:
        return []
    mapped = ACTIVITY_TYPE_MAP.get(value.strip())
    if mapped:
        return mapped
    return [value.lower().strip()]
