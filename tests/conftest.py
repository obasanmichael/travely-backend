from __future__ import annotations

import os

import pytest

# Ensure tests never pick up a developer .env file from the repo root.
os.environ["AUTH_DISABLED"] = "true"
os.environ["ENV"] = "development"


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _load_catalog():
    from services.data_cache import load_catalog

    load_catalog()
