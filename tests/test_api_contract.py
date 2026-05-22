from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AUTH_DISABLED", "true")
os.environ.setdefault("ENV", "development")

from main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def load_catalog():
    from services.data_cache import load_catalog

    load_catalog()


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_recommendations_requires_valid_budget():
    response = client.post(
        "/recommendations",
        json={"budget": 0, "destination_type": "Nature/Adventure", "activity_type": "Hiking"},
    )
    assert response.status_code == 422


def test_recommendations_returns_ranked_results():
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


def test_destinations_list_returns_catalog():
    response = client.get("/destinations")
    assert response.status_code == 200
    destinations = response.json()
    assert len(destinations) > 0
    assert {"destination", "state", "city", "avg_cost_per_day"} <= set(destinations[0].keys())


def test_destinations_filter_by_state():
    all_destinations = client.get("/destinations").json()
    sample_state = all_destinations[0]["state"]
    filtered = client.get("/destinations", params={"state": sample_state}).json()
    assert all(item["state"].lower() == sample_state.lower() for item in filtered)


def test_recommendations_without_auth_when_disabled():
    response = client.post(
        "/recommendations",
        json={"budget": 15000, "destination_type": "Nature/Adventure", "activity_type": "Hiking"},
    )
    assert response.status_code == 200


def test_request_id_header_present():
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
