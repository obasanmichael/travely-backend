from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from db.models import Destination, DestinationTag
from pipelines.db import session_scope
from pipelines.geography import seed_geography
from pipelines.taxonomy import seed_taxonomy
from pipelines.wikidata import CLASSES, Candidate, merge, parse_bindings, upsert_candidates
from tests.conftest import TEST_DATABASE_URL, requires_db

WATERFALL = next(c for c in CLASSES if c.label == "waterfall")
HILL = next(c for c in CLASSES if c.label == "hill")


def _row(qid, label, coord, article=None, image=None, heritage=None, description=None):
    row = {
        "item": {"value": f"http://www.wikidata.org/entity/{qid}"},
        "itemLabel": {"value": label},
        "coord": {"value": coord},
    }
    if article:
        row["article"] = {"value": article}
    if image:
        row["image"] = {"value": image}
    if heritage:
        row["heritage"] = {"value": f"http://www.wikidata.org/entity/{heritage}"}
    if description:
        row["itemDescription"] = {"value": description}
    return row


def test_parse_collapses_duplicate_rows_and_reads_point_order():
    rows = [
        _row("Q1", "Erin Ijesha Waterfalls", "Point(4.87 7.6)", image="http://x/a.jpg"),
        _row("Q1", "Erin Ijesha Waterfalls", "Point(4.87 7.6)", article="https://en.wikipedia.org/wiki/E"),
    ]
    found = parse_bindings(rows, WATERFALL)
    assert list(found) == ["Q1"]
    candidate = found["Q1"]
    assert (candidate.lon, candidate.lat) == (4.87, 7.6)
    assert candidate.image_url == "http://x/a.jpg"
    assert candidate.wikipedia_url.endswith("/E")
    assert candidate.tags["waterfalls"] == 1.0


def test_parse_skips_items_without_english_label_or_coords():
    rows = [_row("Q2", "Q2", "Point(4 7)"), _row("Q3", "Ok", "not a point")]
    assert parse_bindings(rows, WATERFALL) == {}


def test_noisy_classes_require_a_wikipedia_article():
    rows = [
        _row("Q4", "Unknown Hill", "Point(8 9)"),
        _row("Q5", "Shere Hills", "Point(9 9.9)", article="https://en.wikipedia.org/wiki/Shere_Hills"),
    ]
    assert list(parse_bindings(rows, HILL)) == ["Q5"]


def test_world_heritage_adds_heritage_tags():
    rows = [_row("Q6", "Sukur", "Point(13.57 10.74)", heritage="Q9259")]
    candidate = parse_bindings(rows, WATERFALL)["Q6"]
    assert candidate.world_heritage
    assert candidate.tags["history"] == 1.0


def test_merge_keeps_the_strongest_tag_weights():
    a = {"Q7": Candidate("Q7", "X", 1, 1, tags={"nature": 0.5})}
    b = {"Q7": Candidate("Q7", "X", 1, 1, tags={"nature": 0.9, "hiking": 0.4})}
    merge(a, b)
    assert a["Q7"].tags == {"nature": 0.9, "hiking": 0.4}


def _square(x0, y0):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]],
    }


GEO = {
    "type": "FeatureCollection",
    "features": [
        {"properties": {"shapeISO": "NG-OS", "shapeName": "Osun"}, "geometry": _square(4.5, 7.3)},
    ],
}


@requires_db
def test_upsert_respects_reviewer_decisions(migrated_db):
    inside = Candidate("Q901", "Test Falls", 4.8, 7.6, tags={"waterfalls": 1.0, "nature": 0.8})
    outside = Candidate("Q902", "Abroad", -10.0, 50.0, tags={"nature": 1.0})

    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_geography(session, GEO)
            await seed_taxonomy(session)
            first = await upsert_candidates(session, [inside, outside])

        async with session_scope(TEST_DATABASE_URL) as session:
            again = await upsert_candidates(session, [inside])
            dest = await session.scalar(select(Destination).where(Destination.wikidata_id == "Q901"))
            tags = (await session.scalars(
                select(DestinationTag.tag_slug).where(DestinationTag.destination_id == dest.id)
            )).all()
            # A reviewer publishes it and renames it.
            dest.status = "published"
            dest.name = "Reviewed Falls"
            dest.verified_by = "reviewer"
            dest.last_verified_at = datetime.now(timezone.utc)

        async with session_scope(TEST_DATABASE_URL) as session:
            after_review = await upsert_candidates(session, [inside])
            name = await session.scalar(
                select(Destination.name).where(Destination.wikidata_id == "Q901")
            )
        return first, again, sorted(tags), after_review, dest.slug, name

    first, again, tags, after_review, slug, name = asyncio.run(_run())
    assert (first.inserted, first.outside_nigeria) == (1, 1)
    assert (again.inserted, again.refreshed) == (0, 1)
    assert tags == ["nature", "waterfalls"]
    assert after_review.skipped_reviewed == 1
    assert name == "Reviewed Falls"
    assert slug == "test-falls"
