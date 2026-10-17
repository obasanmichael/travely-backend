from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import func, select

from db.models import DestinationClimate
from pipelines.climate import MONTH_KEYS, MonthlyClimate, fill_climate, grid_cell, parse_power
from pipelines.db import session_scope
from pipelines.geography import seed_geography
from pipelines.taxonomy import seed_taxonomy
from pipelines.wikidata import Candidate, upsert_candidates
from tests.conftest import TEST_DATABASE_URL, requires_db


def _payload(temp=25.0, rain=2.0):
    return {
        "properties": {
            "parameter": {
                "T2M": {k: temp for k in MONTH_KEYS} | {"ANN": temp},
                "PRECTOTCORR": {k: rain for k in MONTH_KEYS} | {"ANN": rain},
            }
        }
    }


def test_grid_cell_snaps_to_half_degree_centres():
    assert grid_cell(9.36, 6.39) == (9.25, 6.25)
    assert grid_cell(9.6, 6.6) == (9.75, 6.75)


def test_parse_converts_daily_rain_to_monthly_totals():
    months = parse_power(_payload(rain=2.0))
    assert len(months) == 12
    assert months[0] == MonthlyClimate(1, 25.0, 62.0)
    assert months[1].precip_mm == 56.0


def test_parse_rejects_missing_values():
    payload = _payload()
    payload["properties"]["parameter"]["T2M"]["MAR"] = -999.0
    with pytest.raises(ValueError, match="MAR"):
        parse_power(payload)


@requires_db
def test_fill_climate_groups_by_cell_and_is_idempotent(empty_db):
    geo = {
        "type": "FeatureCollection",
        "features": [{
            "properties": {"shapeISO": "NG-CR", "shapeName": "Cross River"},
            "geometry": {"type": "Polygon", "coordinates": [[[8, 5], [10, 5], [10, 7], [8, 7], [8, 5]]]},
        }],
    }
    calls = []

    def fetcher(cell):
        calls.append(cell)
        return parse_power(_payload())

    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_geography(session, geo)
            await seed_taxonomy(session)
            await upsert_candidates(session, [
                Candidate("Q701", "Obudu A", 9.36, 6.39, tags={"nature": 1.0}),
                Candidate("Q702", "Obudu B", 9.40, 6.30, tags={"nature": 1.0}),
            ])
            first = await fill_climate(session, fetcher, pause_seconds=0)
        async with session_scope(TEST_DATABASE_URL) as session:
            second = await fill_climate(session, fetcher, pause_seconds=0)
            rows = await session.scalar(select(func.count()).select_from(DestinationClimate))
        return first, second, rows

    first, second, rows = asyncio.run(_run())
    assert first.cells_fetched == 1
    assert first.destinations >= 2
    assert second.destinations == 0
    assert rows % 12 == 0
