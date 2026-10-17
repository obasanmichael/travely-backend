"""Admin review: approve, fix or reject candidates, record verified prices, publish.

Every route requires the admin custom claim, and every decision is written to
review_actions so the catalog has an audit trail.
"""
from datetime import date, datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import require_admin
from db import session as db_session
from db.models import (
    CatalogVersion,
    Destination,
    DestinationTag,
    PriceObservation,
    ReviewAction,
    Source,
    State,
)
from engine.taxonomy import TAXONOMY

router = APIRouter(prefix="/v1/admin", tags=["Admin"])

TAG_SLUGS = frozenset(t.slug for t in TAXONOMY)


async def _session():
    if not db_session.engine_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    async for session in db_session.get_session():
        yield session


class TagWeight(BaseModel):
    slug: str
    weight: float = Field(gt=0, le=1)


class AdminDestination(BaseModel):
    id: UUID
    slug: str
    name: str
    status: str
    state: str
    state_iso: str
    latitude: float
    longitude: float
    description: Optional[str]
    access_notes: Optional[str]
    wikipedia_url: Optional[str]
    wikidata_id: Optional[str]
    photo_url: Optional[str]
    photo_credit: Optional[str]
    tags: list[TagWeight]
    accommodation_rates: int
    last_verified_at: Optional[datetime]


class DestinationPage(BaseModel):
    items: list[AdminDestination]
    total: int


class Changes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)
    access_notes: Optional[str] = Field(default=None, max_length=1000)
    latitude: Optional[float] = Field(default=None, ge=4.0, le=14.0)
    longitude: Optional[float] = Field(default=None, ge=2.6, le=14.7)
    tags: Optional[list[TagWeight]] = Field(default=None, max_length=12)

    @field_validator("tags")
    @classmethod
    def _known_tags(cls, value: Optional[list[TagWeight]]) -> Optional[list[TagWeight]]:
        if value is None:
            return value
        unknown = sorted({t.slug for t in value} - TAG_SLUGS)
        if unknown:
            raise ValueError(f"Unknown tags: {', '.join(unknown)}")
        if len({t.slug for t in value}) != len(value):
            raise ValueError("Duplicate tags")
        return value


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "fix", "reject", "unpublish"]
    note: Optional[str] = Field(default=None, max_length=1000)
    changes: Optional[Changes] = None


class PriceObservationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: Literal["accommodation", "entry_fee", "activity", "food"]
    tier: Optional[Literal["budget", "mid", "premium"]] = None
    unit: Literal["per_night", "per_person", "per_meal", "per_trip"]
    destination_id: Optional[UUID] = None
    state_iso: Optional[str] = Field(default=None, pattern=r"^NG-[A-Z]{2}$")
    subject: str = Field(min_length=2, max_length=200, description="What was priced, e.g. the hotel's name")
    amount: float = Field(gt=0, le=50_000_000)
    observed_on: date
    source_url: HttpUrl
    source_kind: Literal["official_site", "manual"] = "official_site"
    notes: Optional[str] = Field(default=None, max_length=500)

    @field_validator("observed_on")
    @classmethod
    def _not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("Observation date can't be in the future")
        return value


class CoverageRow(BaseModel):
    id: UUID
    name: str
    state: str
    missing_tiers: list[str]


class PublishResponse(BaseModel):
    catalog_version: int
    destination_count: int


async def _load(session: AsyncSession, destination_id: UUID) -> Destination:
    destination = await session.get(Destination, destination_id)
    if destination is None:
        raise HTTPException(status_code=404, detail="Destination not found")
    return destination


_LIST_SQL = """
SELECT d.id, d.slug, d.name, d.status, s.name AS state, s.iso_code,
       ST_Y(d.location::geometry) AS lat, ST_X(d.location::geometry) AS lon,
       d.description, d.access_notes, d.wikipedia_url, d.wikidata_id, d.last_verified_at,
       m.url AS photo_url, m.author || ' (' || m.licence || ')' AS photo_credit,
       COALESCE((SELECT json_agg(json_build_object('slug', t.tag_slug, 'weight', t.weight)
                 ORDER BY t.weight DESC) FROM destination_tags t WHERE t.destination_id = d.id),
                '[]') AS tags,
       (SELECT count(*) FROM price_observations o
        WHERE o.destination_id = d.id AND o.component = 'accommodation') AS accommodation_rates
FROM destinations d
JOIN states s ON s.id = d.state_id
LEFT JOIN destination_media m ON m.destination_id = d.id AND m.is_primary
WHERE d.status = :status
  AND (CAST(:state AS text) IS NULL OR s.iso_code = :state)
  AND (CAST(:q AS text) IS NULL OR d.name ILIKE '%' || :q || '%')
ORDER BY (d.wikipedia_url IS NULL), (m.url IS NULL), d.name
LIMIT :limit OFFSET :offset
"""


def _row_to_destination(row: Any) -> AdminDestination:
    return AdminDestination(
        id=row.id, slug=row.slug, name=row.name, status=row.status, state=row.state,
        state_iso=row.iso_code, latitude=row.lat, longitude=row.lon, description=row.description,
        access_notes=row.access_notes, wikipedia_url=row.wikipedia_url, wikidata_id=row.wikidata_id,
        photo_url=row.photo_url, photo_credit=row.photo_credit,
        tags=[TagWeight(slug=t["slug"], weight=float(t["weight"])) for t in row.tags],
        accommodation_rates=row.accommodation_rates, last_verified_at=row.last_verified_at,
    )


@router.get("/destinations", response_model=DestinationPage)
async def list_destinations(
    status: Literal["candidate", "published", "rejected"] = "candidate",
    state: Optional[str] = Query(default=None, pattern=r"^NG-[A-Z]{2}$"),
    q: Optional[str] = Query(default=None, min_length=2, max_length=60),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _admin: dict = Depends(require_admin),
    session: AsyncSession = Depends(_session),
) -> DestinationPage:
    """The review queue: best-documented candidates (Wikipedia, photo) first."""
    params = {"status": status, "state": state, "q": q, "limit": limit, "offset": offset}
    rows = (await session.execute(text(_LIST_SQL), params)).all()
    total = await session.scalar(
        text(
            "SELECT count(*) FROM destinations d JOIN states s ON s.id = d.state_id "
            "WHERE d.status = :status AND (CAST(:state AS text) IS NULL OR s.iso_code = :state) "
            "AND (CAST(:q AS text) IS NULL OR d.name ILIKE '%' || :q || '%')"
        ),
        {"status": status, "state": state, "q": q},
    )
    return DestinationPage(items=[_row_to_destination(r) for r in rows], total=total or 0)


def _apply_changes(destination: Destination, changes: Changes) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for field in ("name", "description", "access_notes"):
        value = getattr(changes, field)
        if value is not None:
            applied[field] = {"from": getattr(destination, field), "to": value}
            setattr(destination, field, value)
    if (changes.latitude is None) != (changes.longitude is None):
        raise HTTPException(status_code=422, detail="Send latitude and longitude together")
    if changes.latitude is not None:
        destination.location = f"SRID=4326;POINT({changes.longitude} {changes.latitude})"
        applied["location"] = {"to": [changes.latitude, changes.longitude]}
    return applied


@router.post("/destinations/{destination_id}/review", response_model=AdminDestination)
async def review_destination(
    destination_id: UUID,
    review: ReviewRequest,
    admin: dict = Depends(require_admin),
    session: AsyncSession = Depends(_session),
) -> AdminDestination:
    async with session.begin():
        destination = await _load(session, destination_id)
        applied: dict[str, Any] = {}
        if review.changes is not None:
            if review.action in ("reject", "unpublish"):
                raise HTTPException(status_code=422, detail="Changes can only accompany approve or fix")
            applied = _apply_changes(destination, review.changes)
            if review.changes.tags is not None:
                await session.execute(
                    delete(DestinationTag).where(DestinationTag.destination_id == destination.id)
                )
                for tag in review.changes.tags:
                    session.add(DestinationTag(destination_id=destination.id, tag_slug=tag.slug, weight=tag.weight))
                applied["tags"] = {"to": {t.slug: t.weight for t in review.changes.tags}}

        now = datetime.now(timezone.utc)
        if review.action == "approve":
            if destination.status == "rejected":
                raise HTTPException(status_code=409, detail="Rejected places must be fixed before approval")
            destination.status = "published"
            destination.verified_by = admin["uid"]
            destination.last_verified_at = now
            destination.published_at = destination.published_at or now
        elif review.action == "reject":
            destination.status = "rejected"
        elif review.action == "unpublish":
            if destination.status != "published":
                raise HTTPException(status_code=409, detail="Only published places can be unpublished")
            destination.status = "candidate"
        elif review.action == "fix" and destination.status == "rejected":
            destination.status = "candidate"

        session.add(ReviewAction(
            destination_id=destination.id, action=review.action, reviewer_uid=admin["uid"],
            note=review.note, changes=applied or None,
        ))
        await session.flush()

    rows = (await session.execute(
        text(_LIST_SQL.replace("WHERE d.status = :status", "WHERE d.id = :id")),
        {"id": destination_id, "state": None, "q": None, "limit": 1, "offset": 0},
    )).all()
    return _row_to_destination(rows[0])


@router.post("/price-observations", status_code=201)
async def add_price_observation(
    observation: PriceObservationIn,
    admin: dict = Depends(require_admin),
    session: AsyncSession = Depends(_session),
) -> dict[str, Any]:
    """Record one verified price, with the page it was read from."""
    if (observation.destination_id is None) == (observation.state_iso is None):
        raise HTTPException(status_code=422, detail="Give either a destination or a state, not both")
    if observation.component == "accommodation" and (observation.tier is None or observation.unit != "per_night"):
        raise HTTPException(status_code=422, detail="Accommodation needs a tier and a per-night amount")

    async with session.begin():
        state_id = None
        if observation.state_iso:
            state_id = await session.scalar(select(State.id).where(State.iso_code == observation.state_iso))
            if state_id is None:
                raise HTTPException(status_code=422, detail="Unknown state")
        else:
            await _load(session, observation.destination_id)

        url = str(observation.source_url)
        source = await session.scalar(select(Source).where(Source.kind == observation.source_kind, Source.url == url))
        if source is None:
            source = Source(
                kind=observation.source_kind, url=url, title=observation.subject,
                retrieved_at=datetime.now(timezone.utc),
            )
            session.add(source)
            await session.flush()

        stmt = insert(PriceObservation).values(
            component=observation.component, tier=observation.tier, unit=observation.unit,
            destination_id=observation.destination_id, state_id=state_id, subject=observation.subject,
            amount=observation.amount, currency="NGN", observed_on=observation.observed_on,
            source_id=source.id, notes=observation.notes,
        )
        created = await session.scalar(
            stmt.on_conflict_do_nothing(constraint="uq_price_observations_natural_key").returning(PriceObservation.id)
        )
    if created is None:
        raise HTTPException(status_code=409, detail="This exact price was already recorded")
    return {"id": created, "recorded_by": admin["uid"]}


@router.get("/coverage", response_model=list[CoverageRow])
async def price_coverage(
    _admin: dict = Depends(require_admin),
    session: AsyncSession = Depends(_session),
) -> list[CoverageRow]:
    """Published places that can't be recommended for overnight trips yet: the rate-collection to-do list."""
    rows = (await session.execute(text(
        """
        SELECT d.id, d.name, s.name AS state,
               ARRAY(SELECT t FROM unnest(ARRAY['budget','mid','premium']) t
                     WHERE NOT EXISTS (
                       SELECT 1 FROM price_observations o
                       WHERE o.component = 'accommodation' AND o.tier = t
                         AND (o.destination_id = d.id OR (o.destination_id IS NULL AND o.state_id = d.state_id))
                     )) AS missing
        FROM destinations d JOIN states s ON s.id = d.state_id
        WHERE d.status = 'published'
        ORDER BY s.name, d.name
        """
    ))).all()
    return [CoverageRow(id=r.id, name=r.name, state=r.state, missing_tiers=list(r.missing)) for r in rows if r.missing]


@router.post("/catalog/publish", response_model=PublishResponse)
async def publish_catalog(
    admin: dict = Depends(require_admin),
    session: AsyncSession = Depends(_session),
) -> PublishResponse:
    """Snapshot the current published set as a new version; running APIs pick it up within a minute."""
    async with session.begin():
        count = await session.scalar(
            select(func.count()).select_from(Destination).where(Destination.status == "published")
        )
        if not count:
            raise HTTPException(status_code=409, detail="Nothing is published yet")
        version = CatalogVersion(published_by=admin["uid"], destination_count=count)
        session.add(version)
        await session.flush()
        return PublishResponse(catalog_version=version.id, destination_count=count)
