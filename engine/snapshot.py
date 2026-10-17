"""Load an immutable, in-memory snapshot of everything the engine needs.

The API holds one snapshot and swaps it atomically when a new catalog version
is published, so every request ranks against a consistent view.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from engine.costs import CostInputs, Observation, Place
from engine.ranker import Destination
from engine.season import MonthClimate, best_months

# What makes a destination a "safer bet" when interests can't separate options.
QUALITY_WEIGHTS = {"wikipedia": 0.5, "photo": 0.3, "description": 0.2}


@dataclass(frozen=True)
class StateInfo:
    iso: str
    name: str
    place: Place


@dataclass(frozen=True)
class Photo:
    url: str
    file_page_url: str
    author: str
    licence: str


@dataclass(frozen=True)
class Details:
    """Display-only facts the ranker doesn't need."""

    description: Optional[str]
    wikipedia_url: Optional[str]
    photo: Optional[Photo]
    best_months: tuple[int, ...]


@dataclass(frozen=True)
class Snapshot:
    catalog: tuple[Destination, ...]
    inputs: CostInputs
    states: Mapping[str, StateInfo]
    catalog_version: Optional[int]
    statuses: tuple[str, ...]
    details: Mapping[str, Details] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.catalog)


async def _states(session: AsyncSession) -> dict[str, StateInfo]:
    rows = await session.execute(text(
        "SELECT iso_code, name, zone, ST_X(centroid::geometry), ST_Y(centroid::geometry) "
        "FROM states WHERE centroid IS NOT NULL AND zone IS NOT NULL"
    ))
    return {iso: StateInfo(iso, name, Place(iso, zone, lon, lat)) for iso, name, zone, lon, lat in rows}


async def _destinations(session: AsyncSession, statuses: Sequence[str]) -> list[Destination]:
    rows = (await session.execute(
        text(
            """
            SELECT d.id::text, d.slug, d.name, s.iso_code, s.zone, s.name,
                   ST_X(d.location::geometry), ST_Y(d.location::geometry),
                   d.city_id, d.wikipedia_url IS NOT NULL, d.description IS NOT NULL,
                   EXISTS (SELECT 1 FROM destination_media m WHERE m.destination_id = d.id AND m.is_primary)
            FROM destinations d JOIN states s ON s.id = d.state_id
            WHERE d.status = ANY(:statuses)
            ORDER BY d.slug
            """
        ),
        {"statuses": list(statuses)},
    )).all()

    tags: dict[str, dict[str, float]] = defaultdict(dict)
    for dest_id, slug, weight in await session.execute(text(
        "SELECT destination_id::text, tag_slug, weight FROM destination_tags"
    )):
        tags[dest_id][slug] = float(weight)

    climate: dict[str, list[MonthClimate]] = defaultdict(list)
    for dest_id, month, temp, rain in await session.execute(text(
        "SELECT destination_id::text, month, temp_mean_c, precip_mm FROM destination_climate"
    )):
        climate[dest_id].append(MonthClimate(int(month), float(temp), float(rain)))

    catalog = []
    for (dest_id, slug, name, iso, zone, state_name, lon, lat, city_id,
         has_wiki, has_description, has_photo) in rows:
        quality = (
            QUALITY_WEIGHTS["wikipedia"] * has_wiki
            + QUALITY_WEIGHTS["photo"] * has_photo
            + QUALITY_WEIGHTS["description"] * has_description
        )
        catalog.append(Destination(
            id=dest_id,
            slug=slug,
            name=name,
            place=Place(iso, zone, lon, lat),
            state_name=state_name,
            tags=tags.get(dest_id, {}),
            climate=tuple(sorted(climate.get(dest_id, []), key=lambda c: c.month)),
            quality=round(quality, 3),
            city_key=str(city_id) if city_id is not None else None,
        ))
    return catalog


async def _cost_inputs(session: AsyncSession, as_of: date) -> CostInputs:
    fares: dict[str, dict[str, Observation]] = defaultdict(dict)
    # Latest observation per state and mode; in-city bus is keyed "local_bus".
    for iso, component, mode, amount, observed_on in await session.execute(text(
        """
        SELECT DISTINCT ON (s.iso_code, o.component, o.mode)
               s.iso_code, o.component, o.mode, o.amount, o.observed_on
        FROM price_observations o JOIN states s ON s.id = o.state_id
        WHERE o.component IN ('intercity_transport', 'local_transport') AND o.mode = 'bus'
        ORDER BY s.iso_code, o.component, o.mode, o.observed_on DESC
        """
    )):
        key = "local_bus" if component == "local_transport" else "bus"
        fares[iso][key] = Observation(float(amount), observed_on)

    food: dict[str, dict[str, Observation]] = defaultdict(dict)
    for zone, item, price, month in await session.execute(text(
        "SELECT DISTINCT ON (zone, item) zone, item, price, month FROM food_prices "
        "ORDER BY zone, item, month DESC"
    )):
        food[zone][item] = Observation(float(price), month)

    accommodation: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for dest_id, city_id, iso, tier, amount, observed_on in await session.execute(text(
        """
        SELECT o.destination_id::text, o.city_id, s.iso_code, o.tier, o.amount, o.observed_on
        FROM price_observations o LEFT JOIN states s ON s.id = o.state_id
        WHERE o.component = 'accommodation' AND o.unit = 'per_night' AND o.tier IS NOT NULL
        """
    )):
        observation = Observation(float(amount), observed_on)
        if dest_id:
            accommodation[("destination", dest_id, tier)].append(observation)
        elif city_id is not None:
            accommodation[("city", str(city_id), tier)].append(observation)
        elif iso:
            accommodation[("state", iso, tier)].append(observation)

    entry_fees: dict[str, list[Observation]] = defaultdict(list)
    for dest_id, amount, observed_on in await session.execute(text(
        "SELECT destination_id::text, amount, observed_on FROM price_observations "
        "WHERE component = 'entry_fee' AND destination_id IS NOT NULL AND unit = 'per_person'"
    )):
        entry_fees[dest_id].append(Observation(float(amount), observed_on))

    return CostInputs(
        as_of=as_of,
        fares=dict(fares),
        food_prices=dict(food),
        accommodation={k: tuple(v) for k, v in accommodation.items()},
        entry_fees={k: tuple(v) for k, v in entry_fees.items()},
    )


async def _details(session: AsyncSession, catalog: Sequence[Destination]) -> dict[str, Details]:
    ids = [d.id for d in catalog]
    if not ids:
        return {}
    text_rows = {
        dest_id: (description, wikipedia_url)
        for dest_id, description, wikipedia_url in await session.execute(
            text("SELECT id::text, description, wikipedia_url FROM destinations WHERE id::text = ANY(:ids)"),
            {"ids": ids},
        )
    }
    photos = {
        dest_id: Photo(url, page, author, licence)
        for dest_id, url, page, author, licence in await session.execute(
            text(
                "SELECT destination_id::text, url, file_page_url, author, licence "
                "FROM destination_media WHERE is_primary AND destination_id::text = ANY(:ids)"
            ),
            {"ids": ids},
        )
    }
    return {
        d.id: Details(
            description=text_rows.get(d.id, (None, None))[0],
            wikipedia_url=text_rows.get(d.id, (None, None))[1],
            photo=photos.get(d.id),
            best_months=tuple(best_months(d.climate, d.tags)),
        )
        for d in catalog
    }


async def load_snapshot(
    session: AsyncSession, statuses: Sequence[str] = ("published",), as_of: Optional[date] = None
) -> Snapshot:
    version = await session.scalar(text("SELECT max(id) FROM catalog_versions"))
    catalog = tuple(await _destinations(session, statuses))
    return Snapshot(
        catalog=catalog,
        inputs=await _cost_inputs(session, as_of or date.today()),
        states=await _states(session),
        catalog_version=version,
        statuses=tuple(statuses),
        details=await _details(session, catalog),
    )
