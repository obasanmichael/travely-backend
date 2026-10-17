from __future__ import annotations

from sqlalchemy import text

from tests.conftest import requires_db
from tests.test_api_v1 import BODY, _run

pytestmark = requires_db


def test_profile_round_trip_and_validation(empty_db):
    async def scenario(client):
        empty = (await client.get("/v1/me/profile")).json()
        saved = await client.patch("/v1/me/profile", json={
            "display_name": " Ada Obi ", "home_state": "NG-LA",
            "preferences": {"tier": "premium", "pace": "packed", "interests": {"hiking": "love", "history": "like"}},
            "marketing_consent": True,
        })
        again = (await client.get("/v1/me/profile")).json()
        cleared = (await client.patch("/v1/me/profile", json={"clear_home_state": True})).json()
        bad_interest = await client.patch("/v1/me/profile", json={"preferences": {"interests": {"skiing": "love"}}})
        bad_state = await client.patch("/v1/me/profile", json={"home_state": "NG-ZZ"})
        extra = await client.patch("/v1/me/profile", json={"is_admin": True})
        return empty, saved, again, cleared, bad_interest, bad_state, extra

    empty, saved, again, cleared, bad_interest, bad_state, extra = _run(scenario)
    assert empty["display_name"] is None and empty["preferences"]["interests"] == {}
    assert saved.status_code == 200
    assert again["display_name"] == "Ada Obi" and again["home_state"] == "NG-LA"
    assert again["preferences"] == {"tier": "premium", "pace": "packed", "interests": {"hiking": "love", "history": "like"}}
    assert again["marketing_consent"] is True and again["analytics_consent"] is False
    assert cleared["home_state"] is None
    assert bad_interest.status_code == 422 and bad_state.status_code == 422 and extra.status_code == 422


def test_export_contains_everything_stored(empty_db):
    async def scenario(client):
        place = (await client.get("/v1/destinations/olumo-rock")).json()["id"]
        await client.put(f"/v1/me/saved/{place}")
        await client.post("/v1/recommendations", json=BODY, headers={"Idempotency-Key": "export-key-1"})
        return await client.get("/v1/me/export")

    response = _run(scenario)
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    data = response.json()
    assert len(data["saved_places"]) == 1 and len(data["trips"]) == 1
    assert data["trips"][0]["answers"]["origin_state"] == "NG-LA"


def test_deleting_an_account_removes_data_and_the_sign_in(empty_db, monkeypatch):
    import firebase_admin.auth

    deleted = []
    monkeypatch.setattr(firebase_admin.auth, "delete_user", lambda uid: deleted.append(uid))

    async def scenario(client):
        place = (await client.get("/v1/destinations/olumo-rock")).json()["id"]
        await client.put(f"/v1/me/saved/{place}")
        await client.post("/v1/recommendations", json=BODY, headers={"Idempotency-Key": "delete-key-1"})
        gone = await client.delete("/v1/me")
        from db import session as db_session

        async for session in db_session.get_session():
            counts = [
                await session.scalar(text(f"SELECT count(*) FROM {table}"))
                for table in ("users", "saved_destinations", "recommendation_runs")
            ]
        return gone, counts

    gone, counts = _run(scenario)
    assert gone.status_code == 204
    assert counts == [0, 0, 0]
    assert deleted == ["dev-user"]


def test_failing_to_remove_the_sign_in_is_reported_and_retryable(empty_db, monkeypatch):
    import firebase_admin.auth

    def boom(uid):
        raise RuntimeError("firebase down")

    monkeypatch.setattr(firebase_admin.auth, "delete_user", boom)

    async def scenario(client):
        return await client.delete("/v1/me")

    response = _run(scenario)
    assert response.status_code == 502
    assert "Try again" in response.json()["detail"]
