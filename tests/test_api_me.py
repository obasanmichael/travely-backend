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


def test_a_trip_can_be_reopened_by_id_but_only_by_its_owner(empty_db):
    async def scenario(client):
        created = (await client.post("/v1/recommendations", json=BODY, headers={"Idempotency-Key": "reopen-key-1"})).json()
        run_id = created["run_id"]
        mine = await client.get(f"/v1/me/runs/{run_id}")
        missing = await client.get("/v1/me/runs/00000000-0000-0000-0000-000000000000")
        # A different user can't read it.
        from core.security import verify_firebase_token
        from main import app

        async def other():
            return {"uid": "someone-else"}

        app.dependency_overrides[verify_firebase_token] = other
        try:
            theirs = await client.get(f"/v1/me/runs/{run_id}")
        finally:
            app.dependency_overrides.pop(verify_firebase_token, None)
        return run_id, mine, missing, theirs

    run_id, mine, missing, theirs = _run(scenario)
    assert mine.status_code == 200
    body = mine.json()
    assert body["id"] == run_id and body["outputs"]["run_id"] == run_id
    assert body["inputs"]["origin_state"] == "NG-LA"
    assert body["outputs"]["recommendations"][0]["destination"]["name"] == "Olumo Rock"
    assert missing.status_code == 404 and theirs.status_code == 404


def test_budget_band_label_matches_the_engines_definition(empty_db):
    async def scenario(client):
        return [(await client.get("/v1/meta/budget-band", params={"per_person_per_day": v})).json()["band"]
                for v in (10_000, 50_000, 150_000)], await client.get("/v1/meta/budget-band", params={"per_person_per_day": -1})

    bands, invalid = _run(scenario)
    assert bands == ["Low", "Medium", "High"]
    assert invalid.status_code == 422


def test_browse_filters_and_pages_the_catalog(empty_db):
    async def scenario(client):
        everything = (await client.get("/v1/destinations")).json()
        by_state = (await client.get("/v1/destinations", params={"state": "NG-OG"})).json()
        by_name = (await client.get("/v1/destinations", params={"q": "lekki"})).json()
        by_interest = (await client.get("/v1/destinations", params={"interest": "wildlife"})).json()
        paged = (await client.get("/v1/destinations", params={"page_size": 1, "page": 2})).json()
        bad = await client.get("/v1/destinations", params={"state": "Ogun"})
        return everything, by_state, by_name, by_interest, paged, bad

    everything, by_state, by_name, by_interest, paged, bad = _run(scenario)
    assert everything["total"] == 2
    assert [d["name"] for d in by_state["items"]] == ["Olumo Rock"]
    assert [d["name"] for d in by_name["items"]] == ["Lekki Conservation Centre"]
    assert [d["name"] for d in by_interest["items"]] == ["Lekki Conservation Centre"]  # wildlife weight 0.6 >= 0.5
    assert paged["page"] == 2 and len(paged["items"]) == 1 and paged["total"] == 2
    assert bad.status_code == 422


def test_a_new_visitor_sees_none_of_another_users_trips(empty_db):
    async def scenario(client):
        from core.security import verify_firebase_token
        from main import app

        await client.post("/v1/recommendations", json=BODY, headers={"Idempotency-Key": "owner-key-1"})
        owner = (await client.get("/v1/me/runs")).json()

        async def stranger():
            return {"uid": "a-new-guest"}

        app.dependency_overrides[verify_firebase_token] = stranger
        try:
            theirs = (await client.get("/v1/me/runs")).json()
        finally:
            app.dependency_overrides.pop(verify_firebase_token, None)
        return owner, theirs

    owner, theirs = _run(scenario)
    assert len(owner) == 1
    assert theirs == []
