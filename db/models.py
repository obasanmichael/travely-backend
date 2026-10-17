from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from geoalchemy2 import Geography
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base

DESTINATION_STATUSES = ("candidate", "published", "rejected")
REVIEW_ACTIONS = ("approve", "fix", "reject", "unpublish")
PRICE_COMPONENTS = (
    "accommodation",
    "food",
    "local_transport",
    "intercity_transport",
    "entry_fee",
    "activity",
)
PRICE_UNITS = ("per_night", "per_meal", "per_day", "per_trip", "per_person")
TIERS = ("budget", "mid", "premium")
SOURCE_KINDS = (
    "osm",
    "geoboundaries",
    "wikidata",
    "wikipedia",
    "commons",
    "nasa_power",
    "nbs",
    "liteapi",
    "official_site",
    "manual",
    "user_report",
)
ZONES = ("north_central", "north_east", "north_west", "south_east", "south_south", "south_west")
TAG_CATEGORIES = ("nature", "activity", "culture", "leisure", "audience")
RUN_EVENTS = ("impression", "save", "dismiss", "detail_view")
REPORT_STATUSES = ("pending", "accepted", "rejected")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Country(Base):
    __tablename__ = "countries"

    code: Mapped[str] = mapped_column(String(2), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)


class State(Base):
    __tablename__ = "states"
    __table_args__ = (
        UniqueConstraint("country_code", "slug", name="uq_states_country_slug"),
        Index("ix_states_boundary", "boundary", postgresql_using="gist"),
        CheckConstraint(f"zone IS NULL OR {_in('zone', ZONES)}", name="zone"),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    country_code: Mapped[str] = mapped_column(ForeignKey("countries.code"), nullable=False)
    # ISO 3166-2 code (e.g. NG-LA): the stable key ingestion jobs match on.
    iso_code: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    centroid: Mapped[Optional[Any]] = mapped_column(
        Geography("POINT", srid=4326, spatial_index=False), nullable=True
    )
    boundary: Mapped[Optional[Any]] = mapped_column(
        Geography("MULTIPOLYGON", srid=4326, spatial_index=False), nullable=True
    )
    boundary_source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sources.id"))
    # Nigeria's six geopolitical zones; NBS publishes some series only at this level.
    zone: Mapped[Optional[str]] = mapped_column(Text)


class City(Base):
    __tablename__ = "cities"
    __table_args__ = (UniqueConstraint("state_id", "slug", name="uq_cities_state_slug"),)

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[Optional[Any]] = mapped_column(
        Geography("POINT", srid=4326, spatial_index=False), nullable=True
    )


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (CheckConstraint(_in("kind", SOURCE_KINDS), name="kind"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(Text)
    title: Mapped[Optional[str]] = mapped_column(Text)
    licence: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class Tag(Base):
    __tablename__ = "tags"
    __table_args__ = (CheckConstraint(_in("category", TAG_CATEGORIES), name="category"),)

    slug: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    # Selectable in the quiz; some tags (e.g. family-friendly) are filters, not interests.
    is_interest: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")


class Destination(Base):
    __tablename__ = "destinations"
    __table_args__ = (
        CheckConstraint(_in("status", DESTINATION_STATUSES), name="status"),
        # A published destination must have been verified by someone.
        CheckConstraint(
            "status <> 'published' OR (last_verified_at IS NOT NULL AND verified_by IS NOT NULL)",
            name="published_requires_verification",
        ),
        Index("ix_destinations_location", "location", postgresql_using="gist"),
        Index("ix_destinations_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    country_code: Mapped[str] = mapped_column(ForeignKey("countries.code"), nullable=False)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), nullable=False, index=True)
    city_id: Mapped[Optional[int]] = mapped_column(ForeignKey("cities.id"), index=True)
    location: Mapped[Any] = mapped_column(
        Geography("POINT", srid=4326, spatial_index=False), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="candidate")
    primary_source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sources.id"))
    osm_id: Mapped[Optional[str]] = mapped_column(Text, unique=True)
    wikidata_id: Mapped[Optional[str]] = mapped_column(Text, unique=True)
    wikipedia_url: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    description_source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sources.id"))
    access_notes: Mapped[Optional[str]] = mapped_column(Text)
    last_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    verified_by: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class DestinationTag(Base):
    __tablename__ = "destination_tags"
    __table_args__ = (CheckConstraint("weight > 0 AND weight <= 1", name="weight_range"),)

    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), primary_key=True
    )
    tag_slug: Mapped[str] = mapped_column(ForeignKey("tags.slug"), primary_key=True)
    weight: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)


class DestinationMedia(Base):
    __tablename__ = "destination_media"
    __table_args__ = (
        # At most one primary image per destination.
        Index(
            "uq_destination_media_primary",
            "destination_id",
            unique=True,
            postgresql_where="is_primary",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    file_page_url: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str] = mapped_column(Text, nullable=False)
    licence: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), nullable=False)


class DestinationClimate(Base):
    __tablename__ = "destination_climate"
    __table_args__ = (CheckConstraint("month BETWEEN 1 AND 12", name="month_range"),)

    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), primary_key=True
    )
    month: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    temp_mean_c: Mapped[Decimal] = mapped_column(Numeric(4, 1), nullable=False)
    precip_mm: Mapped[Decimal] = mapped_column(Numeric(6, 1), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), nullable=False)


class ReviewAction(Base):
    __tablename__ = "review_actions"
    __table_args__ = (CheckConstraint(_in("action", REVIEW_ACTIONS), name="action"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(Text, nullable=False)
    reviewer_uid: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text)
    changes: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()


class PriceObservation(Base):
    """A single observed price. Never updated in place; estimates are derived from these."""

    __tablename__ = "price_observations"
    __table_args__ = (
        CheckConstraint(_in("component", PRICE_COMPONENTS), name="component"),
        CheckConstraint(_in("unit", PRICE_UNITS), name="unit"),
        CheckConstraint(f"tier IS NULL OR {_in('tier', TIERS)}", name="tier"),
        CheckConstraint("amount > 0", name="amount_positive"),
        CheckConstraint(
            "state_id IS NOT NULL OR city_id IS NOT NULL OR destination_id IS NOT NULL",
            name="has_place",
        ),
        Index("ix_price_observations_lookup", "component", "state_id", "city_id", "tier"),
        # Re-importing the same release can't duplicate an observation.
        UniqueConstraint(
            "source_id",
            "component",
            "mode",
            "tier",
            "state_id",
            "city_id",
            "destination_id",
            "origin_state_id",
            "observed_on",
            name="uq_price_observations_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    component: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[Optional[str]] = mapped_column(Text)
    unit: Mapped[str] = mapped_column(Text, nullable=False)
    state_id: Mapped[Optional[int]] = mapped_column(ForeignKey("states.id"))
    city_id: Mapped[Optional[int]] = mapped_column(ForeignKey("cities.id"))
    destination_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE")
    )
    origin_state_id: Mapped[Optional[int]] = mapped_column(ForeignKey("states.id"))
    mode: Mapped[Optional[str]] = mapped_column(Text)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    observed_on: Mapped[date] = mapped_column(Date, nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class FoodPrice(Base):
    """NBS Selected Food Price Watch: one item's average price in a zone for a month."""

    __tablename__ = "food_prices"
    __table_args__ = (
        CheckConstraint("price > 0", name="price_positive"),
        CheckConstraint(_in("zone", ZONES), name="zone"),
    )

    zone: Mapped[str] = mapped_column(Text, primary_key=True)
    item: Mapped[str] = mapped_column(Text, primary_key=True)
    month: Mapped[date] = mapped_column(Date, primary_key=True)
    price: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), nullable=False)


class CpiIndex(Base):
    __tablename__ = "cpi_index"

    country_code: Mapped[str] = mapped_column(ForeignKey("countries.code"), primary_key=True)
    month: Mapped[date] = mapped_column(Date, primary_key=True)
    all_items: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    food: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3))
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), nullable=False)


class CatalogVersion(Base):
    __tablename__ = "catalog_versions"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    published_at: Mapped[datetime] = _created_at()
    published_by: Mapped[str] = mapped_column(Text, nullable=False)
    destination_count: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)


class User(Base):
    __tablename__ = "users"

    firebase_uid: Mapped[str] = mapped_column(Text, primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(Text)
    display_name: Mapped[Optional[str]] = mapped_column(Text)
    is_anonymous: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    home_state_id: Mapped[Optional[int]] = mapped_column(ForeignKey("states.id"))
    analytics_consent: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    marketing_consent: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SavedDestination(Base):
    __tablename__ = "saved_destinations"

    uid: Mapped[str] = mapped_column(
        ForeignKey("users.firebase_uid", ondelete="CASCADE"), primary_key=True
    )
    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = _created_at()


class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"
    __table_args__ = (
        # Makes a double-submitted quiz impossible to store twice, not merely detectable.
        UniqueConstraint("uid", "idempotency_key", name="uq_recommendation_runs_uid_idem"),
        Index("ix_recommendation_runs_uid_created", "uid", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    uid: Mapped[str] = mapped_column(
        ForeignKey("users.firebase_uid", ondelete="CASCADE"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    inputs: Mapped[dict] = mapped_column(JSONB, nullable=False)
    outputs: Mapped[dict] = mapped_column(JSONB, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    catalog_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("catalog_versions.id"))
    created_at: Mapped[datetime] = _created_at()


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (CheckConstraint(_in("event", RUN_EVENTS), name="event"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("recommendation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), nullable=False
    )
    event: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class PriceReport(Base):
    __tablename__ = "price_reports"
    __table_args__ = (
        CheckConstraint(_in("status", REPORT_STATUSES), name="status"),
        CheckConstraint(_in("component", PRICE_COMPONENTS), name="component"),
        CheckConstraint("amount > 0", name="amount_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # Kept after account deletion (uid nulled) so moderated price data survives.
    uid: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.firebase_uid", ondelete="SET NULL")
    )
    destination_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("destinations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    component: Mapped[str] = mapped_column(Text, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    travelled_on: Mapped[Optional[date]] = mapped_column(Date)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = _created_at()
