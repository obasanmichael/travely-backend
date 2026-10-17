from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from main import app

    with TestClient(app) as test_client:
        yield test_client


def test_livez(client):
    assert client.get("/livez").json() == {"status": "alive"}


def test_readyz_ready_when_catalog_loaded(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["checks"]["catalog"] is True


def test_readyz_not_ready_when_catalog_empty(client, monkeypatch):
    import services.data_cache as data_cache

    monkeypatch.setattr(data_cache, "_CATALOG_DF", pd.DataFrame())
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_production_rejects_auth_disabled(monkeypatch):
    from core.config import Settings

    with pytest.raises(ValueError, match="AUTH_DISABLED"):
        Settings(env="production", auth_disabled=True, _env_file=None)


def test_production_requires_database_url():
    from core.config import Settings

    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(env="production", auth_disabled=False, database_url="", _env_file=None)


def test_readyz_not_ready_when_database_unreachable(client, monkeypatch):
    from db import session as db_session

    async def _down() -> bool:
        return False

    monkeypatch.setattr(db_session, "engine_configured", lambda: True)
    monkeypatch.setattr(db_session, "ping", _down)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["database"] is False
