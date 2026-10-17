from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import text

from tests.conftest import TEST_DATABASE_URL, requires_db

pytestmark = requires_db

GEO = {
    "type": "FeatureCollection",
    "features": [{
        "properties": {"shapeISO": "NG-PL", "shapeName": "Plateau"},
        "geometry": {"type": "Polygon", "coordinates": [[[8, 8.5], [10, 8.5], [10, 10.5], [8, 10.5], [8, 8.5]]]},
    }],
}


async def _seed():
    from pipelines.db import session_scope
    from pipelines.geography import seed_geography
    from pipelines.taxonomy import seed_taxonomy
    from pipelines.wikidata import Candidate, upsert_candidates

    async with session_scope(TEST_DATABASE_URL) as session:
        await seed_geography(session, GEO)
        await seed_taxonomy(session)
        await upsert_candidates(session, [
            Candidate("Q10", "Shere Hills", 9.0, 9.9, tags={"hills-mountains": 1.0, "hiking": 0.8},
                      wikipedia_url="https://en.wikipedia.org/wiki/Shere_Hills"),
            Candidate("Q11", "Victoria Roundabout", 8.9, 9.8, tags={"city-life": 0.6}),
        ])


def _run(scenario, admin=True):
    from core.security import verify_firebase_token
    from db import session as db_session
    from main import app

    async def fake_token():
        return {"uid": "reviewer-1", "admin": admin} if admin else {"uid": "someone"}

    async def _go():
        await _seed()
        db_session.init_engine(TEST_DATABASE_URL)
        app.dependency_overrides[verify_firebase_token] = fake_token
        import core.config as config

        config.get_settings.cache_clear()
        settings = config.get_settings()
        original = settings.auth_disabled
        settings.auth_disabled = False  # exercise the real admin check
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await scenario(client)
        finally:
            settings.auth_disabled = original
            app.dependency_overrides.clear()
            await db_session.dispose_engine()

    return asyncio.run(_go())


async def _id_of(client, name):
    page = (await client.get("/v1/admin/destinations", params={"q": name})).json()
    return page["items"][0]["id"]


def test_non_admins_are_refused(empty_db):
    async def scenario(client):
        return (await client.get("/v1/admin/destinations")).status_code

    assert _run(scenario, admin=False) == 403


def test_review_queue_puts_documented_candidates_first(empty_db):
    async def scenario(client):
        return (await client.get("/v1/admin/destinations")).json()

    page = _run(scenario)
    assert page["total"] == 2
    assert page["items"][0]["name"] == "Shere Hills"


def test_approve_fix_reject_flow_is_audited(empty_db):
    async def scenario(client):
        shere = await _id_of(client, "Shere")
        roundabout = await _id_of(client, "Roundabout")
        bad_tag = await client.post(f"/v1/admin/destinations/{shere}/review", json={
            "action": "fix", "changes": {"tags": [{"slug": "skiing", "weight": 1}]},
        })
        approved = await client.post(f"/v1/admin/destinations/{shere}/review", json={
            "action": "approve", "note": "Checked on the map",
            "changes": {"description": "Rocky hills east of Jos", "tags": [
                {"slug": "hills-mountains", "weight": 1}, {"slug": "hiking", "weight": 0.9},
                {"slug": "photography", "weight": 0.6}]},
        })
        rejected = await client.post(f"/v1/admin/destinations/{roundabout}/review", json={"action": "reject"})
        approve_rejected = await client.post(f"/v1/admin/destinations/{roundabout}/review", json={"action": "approve"})
        from db import session as db_session

        async for session in db_session.get_session():
            actions = (await session.execute(text(
                "SELECT action, reviewer_uid FROM review_actions ORDER BY id"
            ))).all()
        return bad_tag, approved, rejected, approve_rejected, actions

    bad_tag, approved, rejected, approve_rejected, actions = _run(scenario)
    assert bad_tag.status_code == 422
    body = approved.json()
    assert body["status"] == "published" and body["last_verified_at"]
    assert body["description"] == "Rocky hills east of Jos"
    assert {t["slug"] for t in body["tags"]} == {"hills-mountains", "hiking", "photography"}
    assert rejected.json()["status"] == "rejected"
    assert approve_rejected.status_code == 409
    assert [a.action for a in actions] == ["approve", "reject"]
    assert all(a.reviewer_uid == "reviewer-1" for a in actions)


def test_prices_coverage_and_publishing(empty_db):
    async def scenario(client):
        shere = await _id_of(client, "Shere")
        nothing_published = await client.post("/v1/admin/catalog/publish")
        await client.post(f"/v1/admin/destinations/{shere}/review", json={"action": "approve"})
        coverage_before = (await client.get("/v1/admin/coverage")).json()
        rate = {
            "component": "accommodation", "tier": "mid", "unit": "per_night", "destination_id": shere,
            "subject": "Hill Station Hotel", "amount": 45000, "observed_on": "2026-09-20",
            "source_url": "https://example-hotel.ng/rates",
        }
        first = await client.post("/v1/admin/price-observations", json=rate)
        duplicate = await client.post("/v1/admin/price-observations", json=rate)
        both = await client.post("/v1/admin/price-observations", json={**rate, "state_iso": "NG-PL"})
        no_tier = await client.post("/v1/admin/price-observations", json={**rate, "tier": None})
        coverage_after = (await client.get("/v1/admin/coverage")).json()
        published = await client.post("/v1/admin/catalog/publish")
        return nothing_published, coverage_before, first, duplicate, both, no_tier, coverage_after, published

    (nothing_published, before, first, duplicate, both, no_tier, after, published) = _run(scenario)
    assert nothing_published.status_code == 409
    assert before[0]["missing_tiers"] == ["budget", "mid", "premium"]
    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert both.status_code == 422 and no_tier.status_code == 422
    assert after[0]["missing_tiers"] == ["budget", "premium"]
    assert published.json() == {"catalog_version": 1, "destination_count": 1}
