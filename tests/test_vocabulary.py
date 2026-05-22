from services.vocabulary import (
    activity_filter_tokens,
    destination_filter_tokens,
    normalize_activity_type,
    normalize_destination_type,
)


def test_nature_adventure_maps_to_nature():
    assert normalize_destination_type("Nature/Adventure") == "nature"
    assert destination_filter_tokens("Nature/Adventure") == ["nature"]


def test_leisure_urban_maps_to_urban_leisure():
    assert normalize_destination_type("Leisure/Urban") == "urban leisure"
    assert destination_filter_tokens("Leisure/Urban") == ["urban leisure"]


def test_historical_cultural_maps_to_both_tokens():
    tokens = destination_filter_tokens("Historical/Cultural")
    assert "historical" in tokens
    assert "cultural" in tokens


def test_hiking_activity_normalization():
    assert "hiking" in normalize_activity_type("Hiking")
    assert "hiking" in activity_filter_tokens("Hiking")


def test_unknown_value_falls_back_to_lower():
    assert normalize_destination_type("Custom/Type") == "custom/type"
