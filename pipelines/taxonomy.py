"""The controlled vocabulary shared by the quiz and the catalog.

Destinations carry weighted tags from this list; the quiz asks users to pick
from the same list, so the engine compares like with like.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Tag


@dataclass(frozen=True)
class TagDef:
    slug: str
    label: str
    category: str
    description: str
    is_interest: bool = True


TAXONOMY: tuple[TagDef, ...] = (
    TagDef("nature", "Nature", "nature", "Green spaces, landscapes and the outdoors"),
    TagDef("waterfalls", "Waterfalls", "nature", "Falls, cascades and plunge pools"),
    TagDef("hills-mountains", "Hills & mountains", "nature", "Highlands, plateaus, rock formations"),
    TagDef("lakes-rivers", "Lakes & rivers", "nature", "Lakes, rivers, springs and wetlands"),
    TagDef("beaches", "Beaches", "nature", "Coastline and sandy shores"),
    TagDef("caves", "Caves", "nature", "Caves and rock shelters"),
    TagDef("wildlife", "Wildlife", "nature", "Parks, reserves and animal viewing"),
    TagDef("hiking", "Hiking", "activity", "Walking and trekking trails"),
    TagDef("swimming", "Swimming", "activity", "Places you can swim"),
    TagDef("boating", "Boat rides", "activity", "Boat cruises, canoeing and river trips"),
    TagDef("birdwatching", "Birdwatching", "activity", "Notable bird habitats"),
    TagDef("camping", "Camping", "activity", "Overnight stays in nature"),
    TagDef("photography", "Photography", "activity", "Scenic and photogenic spots"),
    TagDef("history", "History", "culture", "Historic sites, colonial and pre-colonial heritage"),
    TagDef("culture", "Culture & traditions", "culture", "Living culture, palaces and crafts"),
    TagDef("museums", "Museums", "culture", "Museums and galleries"),
    TagDef("art", "Art", "culture", "Art galleries, sculpture and creative spaces"),
    TagDef("sacred-sites", "Sacred sites", "culture", "Shrines, groves and religious heritage"),
    TagDef("festivals", "Festivals", "culture", "Annual festivals and cultural events"),
    TagDef("architecture", "Architecture", "culture", "Notable buildings and city walls"),
    TagDef("relaxation", "Relaxation", "leisure", "Resorts, springs and slow days"),
    TagDef("nightlife", "Nightlife", "leisure", "Bars, clubs and evening entertainment"),
    TagDef("food", "Food", "leisure", "Local cuisine and food scenes"),
    TagDef("shopping", "Shopping", "leisure", "Markets, malls and crafts"),
    TagDef("amusement", "Amusement & theme parks", "leisure", "Rides, water parks, play"),
    TagDef("city-life", "City life", "leisure", "Urban parks, landmarks and city energy"),
    TagDef("adventure", "Adventure", "leisure", "Climbing, zip lines and adrenaline"),
    TagDef(
        "family-friendly",
        "Family-friendly",
        "audience",
        "Suitable for children and mixed-age groups",
        is_interest=False,
    ),
)


async def seed_taxonomy(session: AsyncSession) -> int:
    for tag in TAXONOMY:
        values = {
            "slug": tag.slug,
            "label": tag.label,
            "category": tag.category,
            "description": tag.description,
            "is_interest": tag.is_interest,
        }
        stmt = insert(Tag).values(**values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["slug"],
                set_={k: stmt.excluded[k] for k in values if k != "slug"},
            )
        )
    return len(TAXONOMY)
