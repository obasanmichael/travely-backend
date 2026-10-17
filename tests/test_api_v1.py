from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pytest
from sqlalchemy import text

from tests.conftest import TEST_DATABASE_URL, requires_db

GEO = {
    "type": "FeatureCollection",
    "features": [
        {"properties": {"shapeISO": "NG-LA", "shapeName": "Lagos"},
         "geometry": {"type": "Polygon", "coordinates": [[[2.7, 6.3], [4.3, 6.3], [4.3, 6.8], [2.7, 6.8], [2.7, 6.3]]]}},
        {"properties": {"shapeISO": "NG-OG", "shapeName": "Ogun"},
         "geometry": {"type": "Polygon", "coordinates": [[[2.7, 6.8], [4.3, 6.8], [4.3, 7.9], [2.7, 7.9], [2.7, 6.8]]]}},
    ],
}
MAY = date(2026, 5, 1)


async def _seed():
    from engine.costs import MEAL_BASKET
    from pipelines.db import session_scope
    from pipelines.geography import seed_geography
    from pipelines.taxonomy import seed_taxonomy
    from pipelines.wikidata import Candidate, upsert_candidates

    async with session_scope(TEST_DATABASE_URL) as session:
        await seed_geography(session, GEO)
        await seed_taxonomy(session)
        await upsert_candidates(session, [
            Candidate("Q1", "Olumo Rock", 3.35, 7.16, tags={"hills-mountains": 1.0, "history": 0.8, "hiking": 0.6}),
            Candidate("Q2", "Lekki Conservation Centre", 3.54, 6.44, tags={"nature": 1.0, "wildlife": 0.6}),
        ])
        src = await session.scalar(text(
            "INSERT INTO sources (kind, url, retrieved_at) VALUES ('nbs', 'test', now()) RETURNING id"
        ))
        await session.execute(text(
            """
            INSERT INTO price_observations (component, mode, unit, state_id, amount, currency, observed_on, source_id)
            SELECT c, 'bus', 'per_trip', s.id, a, 'NGN', :d, :src FROM states s,
                   (VALUES ('intercity_transport', 9000), ('local_transport', 1500)) v(c, a)
            """
        ), {"d": MAY, "src": src})
        for zone in ("south_west",):
            for item in MEAL_BASKET:
                await session.execute(text(
                    "INSERT INTO food_prices (zone, item, month, price, source_id) VALUES (:z, :i, :m, 1500, :s)"
                ), {"z": zone, "i": item, "m": MAY, "s": src})
        await session.execute(text(
            """
            INSERT INTO price_observations (component, tier, unit, state_id, amount, currency, observed_on, source_id, subject)
            SELECT 'accommodation', 'mid', 'per_night', s.id, a, 'NGN', DATE '2026-09-01', :src, 'Hotel ' || a
            FROM states s, (VALUES (30000), (40000), (50000)) v(a)
            """
        ), {"src": src})


def _run(scenario):
    """Run an async scenario against the app with a real DB and engine snapshot."""
    from db import session as db_session
    from engine.snapshot import load_snapshot
    from main import app
    from services import snapshot_store

    async def _go():
        await _seed()
        db_session.init_engine(TEST_DATABASE_URL)
        try:
            async for session in db_session.get_session():
                snapshot_store.set_snapshot(await load_snapshot(session, ("candidate",), as_of=date(2026, 10, 1)))
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await scenario(client)
        finally:
            snapshot_store.set_snapshot(None)
            await db_session.dispose_engine()

    return asyncio.run(_go())


BODY = {
    "origin_state": "NG-LA", "month": 12, "days": 3, "travellers": 2, "budget": 600000,
    "tier": "mid", "interests": {"hiking": "love", "history": "like"},
}


@requires_db
def test_meta_lists_states_and_interests(empty_db):
    async def scenario(client):
        return (await client.get("/v1/meta")).json()

    meta = _run(scenario)
    assert {s["iso"] for s in meta["states"]} == {"NG-LA", "NG-OG"}
    assert "family-friendly" not in {i["slug"] for i in meta["interests"]}
    assert meta["catalog_size"] == 2


@requires_db
def test_recommendations_are_explained_and_priced(empty_db):
    async def scenario(client):
        return await client.post("/v1/recommendations", json=BODY)

    response = _run(scenario)
    assert response.status_code == 200
    body = response.json()
    top = body["recommendations"][0]
    assert top["destination"]["name"] == "Olumo Rock"
    assert top["cost"]["total"]["low"] <= top["cost"]["total"]["high"]
    assert {c["name"] for c in top["cost"]["components"]} >= {"food", "accommodation"}
    assert any(r["kind"] == "interests" for r in top["reasons"])
    assert body["budget_band"] in {"Low", "Medium", "High"}
    assert body["run_id"] is None  # nothing stored without an idempotency key
    assert top["destination"]["month_scores"] == []  # fixture has no climate


@requires_db
def test_idempotency_key_replays_the_same_run(empty_db):
    headers = {"Idempotency-Key": "quiz-123456"}

    async def scenario(client):
        first = await client.post("/v1/recommendations", json=BODY, headers=headers)
        second = await client.post("/v1/recommendations", json=BODY, headers=headers)
        conflict = await client.post("/v1/recommendations", json={**BODY, "days": 2}, headers=headers)
        from db import session as db_session

        async for session in db_session.get_session():
            count = await session.scalar(text("SELECT count(*) FROM recommendation_runs"))
        return first, second, conflict, count

    first, second, conflict, count = _run(scenario)
    assert first.status_code == second.status_code == 200
    assert first.json()["run_id"] and first.json()["run_id"] == second.json()["run_id"]
    assert first.json()["recommendations"] == second.json()["recommendations"]
    assert conflict.status_code == 409
    assert count == 1


@requires_db
@pytest.mark.parametrize(
    "change, status",
    [
        ({"interests": {"teleporting": "love"}}, 422),
        ({"origin_state": "NG-ZZ"}, 422),
        ({"budget": 0}, 422),
        ({"surprise": True}, 422),
    ],
)
def test_bad_requests_are_rejected(empty_db, change, status):
    async def scenario(client):
        return await client.post("/v1/recommendations", json={**BODY, **change})

    assert _run(scenario).status_code == status


def test_service_unavailable_until_catalog_loads():
    from fastapi.testclient import TestClient

    from main import app
    from services import snapshot_store

    snapshot_store.set_snapshot(None)
    with TestClient(app) as client:
        assert client.get("/v1/meta").status_code == 503
        assert client.post("/v1/recommendations", json=BODY).status_code == 503
