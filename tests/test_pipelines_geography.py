from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from db.models import Country, State
from pipelines.db import session_scope
from pipelines.geography import resolve_state_id, seed_geography
from tests.conftest import TEST_DATABASE_URL, requires_db

pytestmark = requires_db


def _square(x0: float, y0: float, size: float = 1.0) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]],
    }


FIXTURE = {
    "type": "FeatureCollection",
    "features": [
        {"properties": {"shapeISO": "NG-LA", "shapeName": "Lagos"}, "geometry": _square(3.0, 6.0)},
        {"properties": {"shapeISO": "NG-OG", "shapeName": "Ogun"}, "geometry": _square(3.0, 7.0)},
        {
            "properties": {"shapeISO": "NG-FC", "shapeName": "Abuja Federal Capital Territory"},
            "geometry": _square(7.0, 8.5),
        },
    ],
}


async def _seed_and_read():
    async with session_scope(TEST_DATABASE_URL) as session:
        await seed_geography(session, FIXTURE)
    async with session_scope(TEST_DATABASE_URL) as session:
        await seed_geography(session, FIXTURE)
        states = (await session.scalars(select(State).where(State.iso_code.in_(
            ["NG-LA", "NG-OG", "NG-FC"]
        )))).all()
        country = await session.get(Country, "NG")
        lagos = await resolve_state_id(session, 3.4, 6.5)
        ogun = await resolve_state_id(session, 3.35, 7.15)
        nowhere = await resolve_state_id(session, -10.0, 50.0)
        by_iso = {s.iso_code: s for s in states}
        return by_iso, country, lagos, ogun, nowhere


def test_seed_is_idempotent_and_resolves_points(empty_db):
    by_iso, country, lagos, ogun, nowhere = asyncio.run(_seed_and_read())

    assert country.currency == "NGN"
    assert set(by_iso) == {"NG-LA", "NG-OG", "NG-FC"}
    assert by_iso["NG-FC"].name == "Federal Capital Territory"
    assert by_iso["NG-FC"].slug == "federal-capital-territory"
    assert by_iso["NG-LA"].boundary_source_id is not None
    assert by_iso["NG-LA"].zone == "south_west"
    assert by_iso["NG-FC"].zone == "north_central"
    assert lagos == by_iso["NG-LA"].id
    assert ogun == by_iso["NG-OG"].id
    assert nowhere is None


def test_rejects_features_without_nigerian_iso_codes(empty_db):
    bad = {"type": "FeatureCollection", "features": [{"properties": {"shapeISO": "GH-AA"}}]}

    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_geography(session, bad)

    with pytest.raises(ValueError, match="NG ISO"):
        asyncio.run(_run())


def test_every_state_belongs_to_exactly_one_zone():
    from pipelines.geography import _ZONE_MEMBERS, ZONE_BY_ISO

    members = [iso for group in _ZONE_MEMBERS.values() for iso in group]
    assert len(members) == len(set(members)) == len(ZONE_BY_ISO) == 37
