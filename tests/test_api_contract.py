from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from core.config import get_settings

    get_settings.cache_clear()
    from main import app

    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_openapi_schema_generates(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/recommendations" in response.json()["paths"]


def test_recommendations_requires_valid_budget(client):
    response = client.post(
        "/recommendations",
        json={"budget": 0, "destination_type": "Nature/Adventure", "activity_type": "Hiking"},
    )
    assert response.status_code == 422


def test_recommendations_returns_ranked_results(client):
    response = client.post(
        "/recommendations",
        json={
            "budget": 20000,
            "destination_type": "Leisure/Urban",
            "activity_type": "Shopping",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "user_budget_category" in body
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) > 0
    assert "destination" in body["recommendations"][0]


def test_destinations_list_returns_catalog(client):
    response = client.get("/destinations")
    assert response.status_code == 200
    destinations = response.json()
    assert len(destinations) > 0
    assert {"destination", "state", "city", "avg_cost_per_day"} <= set(destinations[0].keys())


def test_destinations_filter_by_state(client):
    all_destinations = client.get("/destinations").json()
    sample_state = all_destinations[0]["state"]
    filtered = client.get("/destinations", params={"state": sample_state}).json()
    assert all(item["state"].lower() == sample_state.lower() for item in filtered)


def test_recommendations_without_auth_when_disabled(client):
    response = client.post(
        "/recommendations",
        json={"budget": 15000, "destination_type": "Nature/Adventure", "activity_type": "Hiking"},
    )
    assert response.status_code == 200


def test_request_id_header_present(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
