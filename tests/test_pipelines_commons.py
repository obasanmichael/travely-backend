from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import func, select

from db.models import DestinationMedia
from pipelines.commons import (
    CommonsImage,
    attach_primary_images,
    clean_author,
    licence_allowed,
    parse_imageinfo,
    parse_p18,
)
from pipelines.db import session_scope
from pipelines.geography import seed_geography
from pipelines.taxonomy import seed_taxonomy
from pipelines.wikidata import Candidate, upsert_candidates
from tests.conftest import TEST_DATABASE_URL, requires_db


@pytest.mark.parametrize(
    "licence, allowed",
    [
        ("CC BY-SA 4.0", True),
        ("CC BY 2.0", True),
        ("CC0", True),
        ("Public domain", True),
        ("PD-old-70", True),
        ("CC BY-SA 4.0 or GFDL", False),
        ("CC BY-NC 4.0", False),
        ("CC BY-ND 3.0", False),
        ("All rights reserved", False),
        ("", False),
    ],
)
def test_only_commercial_reuse_licences_are_allowed(licence, allowed):
    assert licence_allowed(licence) is allowed


def test_author_html_is_flattened():
    raw = '<a href="//commons.wikimedia.org/wiki/User:X">Ade&nbsp;Bola</a>\n (talk)'
    assert clean_author(raw) == "Ade Bola (talk)"


def test_parse_p18_takes_the_first_image():
    entities = {
        "Q1": {"claims": {"P18": [{"mainsnak": {"datavalue": {"value": "Olumo Rock.jpg"}}}]}},
        "Q2": {"claims": {}},
    }
    assert parse_p18(entities) == {"Q1": "File:Olumo Rock.jpg"}


def test_parse_imageinfo_drops_unlicensed_and_anonymous_files():
    def page(title, licence, artist):
        return {
            "title": title,
            "imageinfo": [{
                "url": "https://u/full.jpg",
                "thumburl": "https://u/1280.jpg",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}",
                "thumbwidth": 1280,
                "thumbheight": 853,
                "extmetadata": {
                    "LicenseShortName": {"value": licence},
                    "Artist": {"value": artist},
                },
            }],
        }

    pages = {
        "1": page("File:A.jpg", "CC BY-SA 4.0", "Ada"),
        "2": page("File:B.jpg", "CC BY-NC 2.0", "Bo"),
        "3": page("File:C.jpg", "CC0", ""),
        "-1": {"title": "File:Missing.jpg", "missing": ""},
    }
    images = parse_imageinfo(pages)
    assert list(images) == ["File:A.jpg"]
    assert images["File:A.jpg"].url == "https://u/1280.jpg"
    assert images["File:A.jpg"].width == 1280


@requires_db
def test_attach_is_idempotent_and_skips_bad_licences(migrated_db):
    geo = {
        "type": "FeatureCollection",
        "features": [{
            "properties": {"shapeISO": "NG-EK", "shapeName": "Ekiti"},
            "geometry": {"type": "Polygon", "coordinates": [[[5, 7.5], [6, 7.5], [6, 8], [5, 8], [5, 7.5]]]},
        }],
    }
    candidates = [
        Candidate("Q801", "Ikogosi Warm Springs", 5.2, 7.6, tags={"relaxation": 0.9}),
        Candidate("Q802", "No Photo Hill", 5.3, 7.6, tags={"hiking": 0.8}),
        Candidate("Q803", "NC Photo Falls", 5.4, 7.6, tags={"waterfalls": 1.0}),
    ]
    image = CommonsImage(
        "File:Ikogosi.jpg", "https://u/1280.jpg", "https://c/File:Ikogosi.jpg", "Ada", "CC BY-SA 4.0", 1280, 853
    )

    def p18(qids):
        return {"Q801": "File:Ikogosi.jpg", "Q803": "File:NC.jpg"}

    def imageinfo(titles):
        return {"File:Ikogosi.jpg": image}

    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_geography(session, geo)
            await seed_taxonomy(session)
            await upsert_candidates(session, candidates)
            first = await attach_primary_images(session, p18, imageinfo)
        async with session_scope(TEST_DATABASE_URL) as session:
            second = await attach_primary_images(session, p18, imageinfo)
            count = await session.scalar(
                select(func.count()).select_from(DestinationMedia).where(DestinationMedia.author == "Ada")
            )
        return first, second, count

    first, second, count = asyncio.run(_run())
    assert (first.attached, first.no_image, first.rejected_licence) == (1, 1, 1)
    assert second.attached == 0
    assert count == 1
