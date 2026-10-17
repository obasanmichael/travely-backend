from __future__ import annotations

from sqlalchemy import text

from tests.conftest import requires_db
from tests.test_api_v1 import BODY, _run

pytestmark = requires_db


def test_featured_and_lookup_by_slug(empty_db):
    async def scenario(client):
        featured = await client.get("/v1/destinations/featured")
        found = await client.get("/v1/destinations/olumo-rock")
        missing = await client.get("/v1/destinations/atlantis")
        return featured, found, missing

    featured, found, missing = _run(scenario)
    # The fixture has no photos or climate, so nothing qualifies to be featured.
    assert featured.status_code == 200 and featured.json() == []
    assert found.json()["name"] == "Olumo Rock"
    assert missing.status_code == 404


def test_saving_is_idempotent_and_scoped(empty_db):
    async def scenario(client):
        place = (await client.get("/v1/destinations/olumo-rock")).json()["id"]
        first = await client.put(f"/v1/me/saved/{place}")
        again = await client.put(f"/v1/me/saved/{place}")
        unknown = await client.put("/v1/me/saved/00000000-0000-0000-0000-000000000000")
        listed = (await client.get("/v1/me/saved")).json()
        removed = await client.delete(f"/v1/me/saved/{place}")
        after = (await client.get("/v1/me/saved")).json()
        return first, again, unknown, listed, removed, after

    first, again, unknown, listed, removed, after = _run(scenario)
    assert first.status_code == again.status_code == 204
    assert unknown.status_code == 404
    assert [d["name"] for d in listed] == ["Olumo Rock"]
    assert removed.status_code == 204 and after == []


def test_history_lists_stored_runs(empty_db):
    async def scenario(client):
        await client.post("/v1/recommendations", json=BODY, headers={"Idempotency-Key": "history-key-1"})
        return (await client.get("/v1/me/runs")).json()

    runs = _run(scenario)
    assert len(runs) == 1
    assert runs[0]["inputs"]["origin_state"] == "NG-LA"
    assert runs[0]["top_places"][0] == "Olumo Rock"


def test_guest_session_merges_into_account(empty_db, monkeypatch):
    import firebase_admin.auth

    monkeypatch.setattr(
        firebase_admin.auth, "verify_id_token",
        lambda token, **_: {"uid": "guest-1", "firebase": {"sign_in_provider": "anonymous"}},
    )

    async def scenario(client):
        from db import session as db_session

        place = (await client.get("/v1/destinations/olumo-rock")).json()["id"]
        async for session in db_session.get_session():
            async with session.begin():
                await session.execute(text("INSERT INTO users (firebase_uid, is_anonymous) VALUES ('guest-1', true)"))
                await session.execute(text(
                    "INSERT INTO saved_destinations (uid, destination_id) VALUES ('guest-1', :d)"
                ), {"d": place})
        merged = await client.post("/v1/me/merge", json={"anonymous_token": "x" * 40})
        saved = (await client.get("/v1/me/saved")).json()
        async for session in db_session.get_session():
            guest_left = await session.scalar(text("SELECT count(*) FROM users WHERE firebase_uid = 'guest-1'"))
        return merged, saved, guest_left

    merged, saved, guest_left = _run(scenario)
    assert merged.status_code == 204
    assert [d["name"] for d in saved] == ["Olumo Rock"]
    assert guest_left == 0
