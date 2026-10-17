"""Ingest candidate destinations in Nigeria from Wikidata (CC0).

Everything lands as status='candidate'. A human publishes or rejects it in
the admin review queue; re-running this job never touches reviewed rows.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Destination, DestinationTag, State
from pipelines.db import get_or_create_source
from pipelines.geography import resolve_state_id
from pipelines.http import client
from pipelines.text import slugify

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
NIGERIA_QID = "Q1033"
WORLD_HERITAGE_QID = "Q9259"


@dataclass(frozen=True)
class WikidataClass:
    qid: str
    label: str
    tags: dict[str, float]
    # Classes with lots of obscure entries (every hill in the country) only
    # yield candidates that have an English Wikipedia article.
    require_article: bool = False


CLASSES: tuple[WikidataClass, ...] = (
    WikidataClass("Q34038", "waterfall", {"waterfalls": 1.0, "nature": 0.8, "photography": 0.6, "hiking": 0.4}),
    WikidataClass("Q46169", "national park", {"nature": 1.0, "wildlife": 0.9, "hiking": 0.6, "photography": 0.6, "camping": 0.4}),
    WikidataClass("Q179049", "nature reserve", {"nature": 1.0, "wildlife": 0.7, "birdwatching": 0.4}),
    WikidataClass("Q473972", "protected area", {"nature": 0.9, "wildlife": 0.6}),
    WikidataClass("Q33506", "museum", {"museums": 1.0, "history": 0.8, "culture": 0.6}),
    WikidataClass("Q207694", "art museum", {"museums": 0.8, "art": 1.0, "culture": 0.5}),
    WikidataClass("Q40080", "beach", {"beaches": 1.0, "relaxation": 0.7, "swimming": 0.6, "photography": 0.4}),
    WikidataClass("Q35509", "cave", {"caves": 1.0, "nature": 0.6, "adventure": 0.6}),
    WikidataClass("Q23397", "lake", {"lakes-rivers": 1.0, "nature": 0.7, "boating": 0.5, "photography": 0.4}),
    WikidataClass("Q177380", "hot spring", {"lakes-rivers": 0.8, "relaxation": 0.9, "swimming": 0.7, "nature": 0.6}),
    WikidataClass("Q22698", "park", {"nature": 0.6, "city-life": 0.6, "relaxation": 0.5, "family-friendly": 0.6}),
    WikidataClass("Q1107656", "garden", {"nature": 0.7, "relaxation": 0.6}),
    WikidataClass("Q43501", "zoo", {"wildlife": 0.8, "family-friendly": 0.9}),
    WikidataClass("Q194195", "amusement park", {"amusement": 1.0, "family-friendly": 0.9}),
    WikidataClass("Q839954", "archaeological site", {"history": 1.0, "culture": 0.5}),
    WikidataClass("Q4989906", "monument", {"history": 0.8, "architecture": 0.5}),
    WikidataClass("Q16560", "palace", {"culture": 0.8, "history": 0.8, "architecture": 0.6}),
    WikidataClass("Q8502", "mountain", {"hills-mountains": 1.0, "hiking": 0.8, "photography": 0.5}, require_article=True),
    WikidataClass("Q54050", "hill", {"hills-mountains": 1.0, "hiking": 0.8, "photography": 0.5}, require_article=True),
    WikidataClass("Q23442", "island", {"nature": 0.7, "beaches": 0.5, "boating": 0.5}, require_article=True),
    WikidataClass("Q2319498", "landmark", {"city-life": 0.6, "photography": 0.6}, require_article=True),
)

WORLD_HERITAGE_TAGS = {"history": 1.0, "culture": 1.0, "sacred-sites": 0.5}

_POINT_RE = re.compile(r"Point\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")
_QID_RE = re.compile(r"/entity/(Q\d+)$")


@dataclass
class Candidate:
    qid: str
    name: str
    lon: float
    lat: float
    description: Optional[str] = None
    wikipedia_url: Optional[str] = None
    image_url: Optional[str] = None
    world_heritage: bool = False
    tags: dict[str, float] = field(default_factory=dict)
    classes: set[str] = field(default_factory=set)

    def add_tags(self, tags: dict[str, float]) -> None:
        for slug, weight in tags.items():
            self.tags[slug] = max(weight, self.tags.get(slug, 0.0))


def build_query(wd_class: WikidataClass) -> str:
    return f"""
SELECT ?item ?itemLabel ?itemDescription ?coord ?article ?image ?heritage WHERE {{
  ?item wdt:P31 wd:{wd_class.qid} ;
        wdt:P17 wd:{NIGERIA_QID} ;
        wdt:P625 ?coord .
  OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> }}
  OPTIONAL {{ ?item wdt:P18 ?image }}
  OPTIONAL {{ ?item wdt:P1435 ?heritage }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def _value(binding: dict[str, Any], key: str) -> Optional[str]:
    entry = binding.get(key)
    return entry.get("value") if entry else None


def parse_bindings(bindings: Iterable[dict[str, Any]], wd_class: WikidataClass) -> dict[str, Candidate]:
    """Collapse SPARQL rows (one per coord/image/heritage combination) into candidates."""
    candidates: dict[str, Candidate] = {}
    for row in bindings:
        qid_match = _QID_RE.search(_value(row, "item") or "")
        point = _POINT_RE.match(_value(row, "coord") or "")
        name = (_value(row, "itemLabel") or "").strip()
        if not qid_match or not point:
            continue
        qid = qid_match.group(1)
        # Items without an English label come back labelled with their QID.
        if not name or name == qid:
            continue

        candidate = candidates.get(qid)
        if candidate is None:
            candidate = Candidate(
                qid=qid, name=name, lon=float(point.group(1)), lat=float(point.group(2))
            )
            candidates[qid] = candidate
        candidate.classes.add(wd_class.label)
        candidate.add_tags(wd_class.tags)
        candidate.description = candidate.description or _value(row, "itemDescription")
        candidate.wikipedia_url = candidate.wikipedia_url or _value(row, "article")
        candidate.image_url = candidate.image_url or _value(row, "image")
        if (_value(row, "heritage") or "").endswith(f"/{WORLD_HERITAGE_QID}"):
            candidate.world_heritage = True
            candidate.add_tags(WORLD_HERITAGE_TAGS)

    if wd_class.require_article:
        candidates = {qid: c for qid, c in candidates.items() if c.wikipedia_url}
    return candidates


def merge(into: dict[str, Candidate], new: dict[str, Candidate]) -> None:
    for qid, candidate in new.items():
        existing = into.get(qid)
        if existing is None:
            into[qid] = candidate
            continue
        existing.classes |= candidate.classes
        existing.add_tags(candidate.tags)
        existing.description = existing.description or candidate.description
        existing.wikipedia_url = existing.wikipedia_url or candidate.wikipedia_url
        existing.image_url = existing.image_url or candidate.image_url
        existing.world_heritage = existing.world_heritage or candidate.world_heritage


def fetch_candidates(pause_seconds: float = 1.0) -> dict[str, Candidate]:
    """One query per class keeps each request well inside the endpoint's timeout."""
    collected: dict[str, Candidate] = {}
    with client(timeout=90) as http:
        for wd_class in CLASSES:
            response = http.get(
                SPARQL_ENDPOINT,
                params={"query": build_query(wd_class), "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            response.raise_for_status()
            found = parse_bindings(response.json()["results"]["bindings"], wd_class)
            logger.info("wikidata %s: %d candidates", wd_class.label, len(found))
            merge(collected, found)
            time.sleep(pause_seconds)
    return collected


@dataclass
class IngestReport:
    inserted: int = 0
    refreshed: int = 0
    skipped_reviewed: int = 0
    outside_nigeria: int = 0


async def _unique_slug(session: AsyncSession, name: str, state_slug: str, qid: str) -> str:
    base = slugify(name)
    for slug in (base, f"{base}-{state_slug}", f"{base}-{state_slug}-{qid.lower()}"):
        owner = await session.scalar(select(Destination.wikidata_id).where(Destination.slug == slug))
        if owner is None or owner == qid:
            return slug
    raise RuntimeError(f"No free slug for {qid}")


async def upsert_candidates(session: AsyncSession, candidates: Iterable[Candidate]) -> IngestReport:
    report = IngestReport()
    for candidate in candidates:
        existing = await session.scalar(
            select(Destination).where(Destination.wikidata_id == candidate.qid)
        )
        if existing is not None and existing.status != "candidate":
            # Published or rejected by a reviewer: their decision stands.
            report.skipped_reviewed += 1
            continue

        state_id = await resolve_state_id(session, candidate.lon, candidate.lat)
        if state_id is None:
            report.outside_nigeria += 1
            continue
        state_slug = await session.scalar(select(State.slug).where(State.id == state_id))

        source = await get_or_create_source(
            session,
            kind="wikidata",
            url=f"https://www.wikidata.org/wiki/{candidate.qid}",
            title=candidate.name,
            licence="CC0 1.0",
        )
        values = {
            "name": candidate.name,
            "country_code": "NG",
            "state_id": state_id,
            "location": f"SRID=4326;POINT({candidate.lon} {candidate.lat})",
            "wikipedia_url": candidate.wikipedia_url,
            "description": candidate.description,
            "description_source_id": source.id if candidate.description else None,
            "primary_source_id": source.id,
        }
        if existing is None:
            values["slug"] = await _unique_slug(session, candidate.name, state_slug, candidate.qid)
            stmt = insert(Destination).values(wikidata_id=candidate.qid, status="candidate", **values)
            destination_id = await session.scalar(stmt.returning(Destination.id))
            report.inserted += 1
        else:
            for key, value in values.items():
                setattr(existing, key, value)
            destination_id = existing.id
            report.refreshed += 1

        # Candidate tags are fully machine-derived, so replace rather than merge.
        await session.execute(
            delete(DestinationTag).where(DestinationTag.destination_id == destination_id)
        )
        for slug, weight in candidate.tags.items():
            session.add(DestinationTag(destination_id=destination_id, tag_slug=slug, weight=weight))
        await session.flush()
    return report
