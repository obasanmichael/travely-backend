from __future__ import annotations

import asyncio
from datetime import date

from sqlalchemy import text

from engine.snapshot import load_snapshot
from pipelines.db import session_scope
from pipelines.geography import seed_geography
from pipelines.taxonomy import seed_taxonomy
from pipelines.wikidata import Candidate, upsert_candidates
from tests.conftest import TEST_DATABASE_URL, requires_db

pytestmark = requires_db

GEO = {
    "type": "FeatureCollection",
    "features": [{
        "properties": {"shapeISO": "NG-OS", "shapeName": "Osun"},
        "geometry": {"type": "Polygon", "coordinates": [[[4, 7], [5.5, 7], [5.5, 8.5], [4, 8.5], [4, 7]]]},
    }],
}


def test_snapshot_reads_catalog_and_latest_cost_inputs(empty_db):
    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_geography(session, GEO)
            await seed_taxonomy(session)
            await upsert_candidates(session, [
                Candidate("Q1", "Osun Grove", 4.55, 7.75, tags={"sacred-sites": 1.0, "nature": 0.6},
                          wikipedia_url="https://en.wikipedia.org/wiki/Osun-Osogbo"),
            ])
            await session.execute(text(
                "INSERT INTO sources (kind, url, retrieved_at) VALUES ('nbs', 'u', now())"
            ))
            await session.execute(text(
                """
                INSERT INTO price_observations
                  (component, mode, unit, state_id, amount, currency, observed_on, source_id)
                SELECT 'intercity_transport', 'bus', 'per_trip', s.id, a, 'NGN', d, src.id
                FROM states s, sources src,
                     (VALUES (8000, DATE '2026-04-01'), (9000, DATE '2026-05-01')) AS v(a, d)
                WHERE s.iso_code = 'NG-OS' AND src.kind = 'nbs'
                """
            ))
        async with session_scope(TEST_DATABASE_URL) as session:
            candidates = await load_snapshot(session, ("candidate",), as_of=date(2026, 10, 1))
            published = await load_snapshot(session, ("published",))
        return candidates, published

    candidates, published = asyncio.run(_run())
    assert published.size == 0
    assert candidates.size == 1
    grove = candidates.catalog[0]
    assert grove.place.state_iso == "NG-OS" and grove.place.zone == "south_west"
    assert grove.tags == {"sacred-sites": 1.0, "nature": 0.6}
    assert grove.quality == 0.5  # has a Wikipedia article; no description or photo
    assert candidates.inputs.fares["NG-OS"]["bus"].amount == 9000
    assert candidates.states["NG-OS"].name == "Osun"
