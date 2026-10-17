"""Attach a freely-licensed Wikimedia Commons photo to destinations that have none.

The image comes from the destination's Wikidata P18 claim. Only licences that
allow commercial reuse are accepted, and the author and licence are stored so
the UI can credit them.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Destination, DestinationMedia
from pipelines.db import get_or_create_source
from pipelines.http import client

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
BATCH_SIZE = 50
THUMB_WIDTH = 1280

# Licences that permit commercial reuse with attribution (or none). NC/ND are excluded.
_ALLOWED_LICENCE = re.compile(
    r"^(cc0( 1\.0)?|cc by(-sa)?( \d\.\d)?|public domain|pd(-\S+)?)$", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class CommonsImage:
    file_title: str
    url: str
    file_page_url: str
    author: str
    licence: str
    width: Optional[int]
    height: Optional[int]


def licence_allowed(licence: str) -> bool:
    return bool(licence) and bool(_ALLOWED_LICENCE.match(licence.strip()))


def clean_author(raw: str) -> str:
    text = html.unescape(_TAG_RE.sub("", raw or "")).strip()
    return re.sub(r"\s+", " ", text)


def _batches(items: list[str], size: int = BATCH_SIZE) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_p18(entities: dict) -> dict[str, str]:
    """QID -> 'File:<name>' for entities that have an image claim."""
    files: dict[str, str] = {}
    for qid, entity in entities.items():
        claims = (entity.get("claims") or {}).get("P18") or []
        for claim in claims:
            value = (claim.get("mainsnak") or {}).get("datavalue", {}).get("value")
            if value:
                files[qid] = f"File:{value}"
                break
    return files


def parse_imageinfo(pages: dict) -> dict[str, CommonsImage]:
    """'File:<name>' -> image, for files with an allowed licence and an author."""
    images: dict[str, CommonsImage] = {}
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        meta = info.get("extmetadata") or {}
        licence = (meta.get("LicenseShortName") or {}).get("value", "")
        author = clean_author((meta.get("Artist") or {}).get("value", ""))
        if not licence_allowed(licence) or not author:
            continue
        images[page["title"]] = CommonsImage(
            file_title=page["title"],
            url=info.get("thumburl") or info["url"],
            file_page_url=info["descriptionurl"],
            author=author,
            licence=licence,
            width=info.get("thumbwidth") or info.get("width"),
            height=info.get("thumbheight") or info.get("height"),
        )
    return images


def fetch_p18(qids: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    with client() as http:
        for batch in _batches(qids):
            response = http.get(
                WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "claims",
                    "format": "json",
                },
            )
            response.raise_for_status()
            files.update(parse_p18(response.json().get("entities") or {}))
    return files


def fetch_imageinfo(file_titles: list[str]) -> dict[str, CommonsImage]:
    images: dict[str, CommonsImage] = {}
    with client() as http:
        for batch in _batches(file_titles):
            response = http.get(
                COMMONS_API,
                params={
                    "action": "query",
                    "titles": "|".join(batch),
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|size",
                    "iiurlwidth": THUMB_WIDTH,
                    "format": "json",
                },
            )
            response.raise_for_status()
            query = response.json().get("query") or {}
            # Commons normalises titles (underscores, capitalisation); map back.
            renamed = {n["to"]: n["from"] for n in query.get("normalized") or []}
            for title, image in parse_imageinfo(query.get("pages") or {}).items():
                images[renamed.get(title, title)] = image
    return images


@dataclass
class MediaReport:
    attached: int = 0
    no_image: int = 0
    rejected_licence: int = 0


async def attach_primary_images(
    session: AsyncSession,
    p18_fetcher: Callable[[list[str]], dict[str, str]] = fetch_p18,
    imageinfo_fetcher: Callable[[list[str]], dict[str, CommonsImage]] = fetch_imageinfo,
) -> MediaReport:
    has_primary = exists().where(
        DestinationMedia.destination_id == Destination.id, DestinationMedia.is_primary
    )
    rows = (
        await session.execute(
            select(Destination.id, Destination.wikidata_id).where(
                Destination.wikidata_id.is_not(None),
                Destination.status != "rejected",
                ~has_primary,
            )
        )
    ).all()
    if not rows:
        return MediaReport()

    files = p18_fetcher([qid for _, qid in rows])
    images = imageinfo_fetcher(sorted(set(files.values()))) if files else {}

    report = MediaReport()
    for destination_id, qid in rows:
        file_title = files.get(qid)
        if file_title is None:
            report.no_image += 1
            continue
        image = images.get(file_title)
        if image is None:
            report.rejected_licence += 1
            continue
        source = await get_or_create_source(
            session,
            kind="commons",
            url=image.file_page_url,
            title=image.file_title,
            licence=image.licence,
        )
        session.add(
            DestinationMedia(
                destination_id=destination_id,
                url=image.url,
                file_page_url=image.file_page_url,
                author=image.author,
                licence=image.licence,
                width=image.width,
                height=image.height,
                is_primary=True,
                source_id=source.id,
            )
        )
        report.attached += 1
    await session.flush()
    return report
