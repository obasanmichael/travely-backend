"""Seed Nigeria and its 37 first-level divisions (36 states + FCT) with boundaries."""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Country, State
from pipelines.db import get_or_create_source
from pipelines.http import client
from pipelines.text import slugify

# Pinned to a release commit so a re-run never silently changes boundaries.
GEOBOUNDARIES_NGA_ADM1_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/NGA/ADM1/"
    "geoBoundaries-NGA-ADM1_simplified.geojson"
)

_ZONE_MEMBERS = {
    "north_central": ("NG-BE", "NG-FC", "NG-KO", "NG-KW", "NG-NA", "NG-NI", "NG-PL"),
    "north_east": ("NG-AD", "NG-BA", "NG-BO", "NG-GO", "NG-TA", "NG-YO"),
    "north_west": ("NG-JI", "NG-KD", "NG-KE", "NG-KN", "NG-KT", "NG-SO", "NG-ZA"),
    "south_east": ("NG-AB", "NG-AN", "NG-EB", "NG-EN", "NG-IM"),
    "south_south": ("NG-AK", "NG-BY", "NG-CR", "NG-DE", "NG-ED", "NG-RI"),
    "south_west": ("NG-EK", "NG-LA", "NG-OG", "NG-ON", "NG-OS", "NG-OY"),
}
ZONE_BY_ISO = {iso: zone for zone, members in _ZONE_MEMBERS.items() for iso in members}

# geoBoundaries names that differ from the name people use.
NAME_OVERRIDES = {"NG-FC": "Federal Capital Territory"}


def fetch_adm1() -> dict[str, Any]:
    with client() as http:
        response = http.get(GEOBOUNDARIES_NGA_ADM1_URL)
        response.raise_for_status()
        return response.json()


def _features(collection: dict[str, Any]) -> list[dict[str, Any]]:
    features = collection.get("features") or []
    if not features:
        raise ValueError("Boundary file has no features")
    for feature in features:
        props = feature.get("properties") or {}
        if not props.get("shapeISO", "").startswith("NG-"):
            raise ValueError(f"Unexpected feature without an NG ISO code: {props}")
    return features


async def seed_geography(session: AsyncSession, collection: dict[str, Any]) -> int:
    features = _features(collection)

    await session.execute(
        insert(Country)
        .values(code="NG", name="Nigeria", currency="NGN")
        .on_conflict_do_nothing(index_elements=["code"])
    )
    source = await get_or_create_source(
        session,
        kind="geoboundaries",
        url=GEOBOUNDARIES_NGA_ADM1_URL,
        title="geoBoundaries NGA ADM1 (source: GRID3, 2022)",
        licence="CC BY 4.0",
    )

    for feature in features:
        iso = feature["properties"]["shapeISO"]
        name = NAME_OVERRIDES.get(iso, feature["properties"]["shapeName"])
        boundary = func.ST_Multi(
            func.ST_SetSRID(func.ST_GeomFromGeoJSON(json.dumps(feature["geometry"])), 4326)
        )
        values = {
            "country_code": "NG",
            "iso_code": iso,
            "name": name,
            "slug": slugify(name),
            "boundary": func.Geography(boundary),
            "centroid": func.Geography(func.ST_PointOnSurface(boundary)),
            "boundary_source_id": source.id,
            "zone": ZONE_BY_ISO.get(iso),
        }
        stmt = insert(State).values(**values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["iso_code"],
                set_={k: stmt.excluded[k] for k in values if k != "iso_code"},
            )
        )
    return len(features)


async def resolve_state_id(session: AsyncSession, lon: float, lat: float) -> Optional[int]:
    """The state whose boundary contains the point, or None if it falls outside them all."""
    point = func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326)
    return await session.scalar(
        select(State.id)
        .where(func.ST_Intersects(State.boundary, func.Geography(point)))
        .limit(1)
    )


async def count_states(session: AsyncSession) -> int:
    return (await session.execute(text("SELECT count(*) FROM states"))).scalar_one()
