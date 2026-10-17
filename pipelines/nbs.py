"""Import Nigeria Bureau of Statistics price watches.

- Transport Fare Watch (catalog 161): state averages for intercity bus, in-city
  bus, okada, air and water fares, stored as price_observations.
- Selected Food Price Watch (catalog 162): zone averages for staple items,
  stored as food_prices.

Each monthly release is a zip holding a PDF and an .xlsx; only the .xlsx is read.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional

import openpyxl
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import FoodPrice, PriceObservation, State
from pipelines.db import get_or_create_source
from pipelines.http import client
from pipelines.text import slugify

logger = logging.getLogger(__name__)

CATALOG_URL = "https://microdata.nigerianstat.gov.ng/index.php/catalog/{catalog_id}"
TRANSPORT_CATALOG = 161
FOOD_CATALOG = 162

_MONTHS = {
    name: index
    for index, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}
# Letter boundaries, not \b: file names glue words with underscores ("Report_MAR26").
_MONTH_YEAR_RE = re.compile(r"(?<![A-Za-z])([A-Za-z]{3,9})[\s_\-.]*'?(\d{4}|\d{2})(?!\d)")

# NBS spells some states differently from the boundary data.
_STATE_ALIASES = {
    "abuja": "NG-FC",
    "fct": "NG-FC",
    "fct-abuja": "NG-FC",
    "federal-capital-territory": "NG-FC",
    "nassarawa": "NG-NA",
}

# (component, mode, keyword in the NBS column header)
TRANSPORT_COLUMNS = (
    ("intercity_transport", "bus", "intercity"),
    ("local_transport", "bus", "within"),
    ("local_transport", "okada", "okada"),
    ("intercity_transport", "air", "air"),
    ("intercity_transport", "water", "water"),
)

ZONE_HEADERS = {
    "north central": "north_central",
    "north east": "north_east",
    "north west": "north_west",
    "south east": "south_east",
    "south south": "south_south",
    "south west": "south_west",
}


def parse_month(*texts: str) -> date:
    """First day of the month named in a sheet title or file name, e.g. 'Transport May 2026'."""
    for text in texts:
        for word, year in _MONTH_YEAR_RE.findall(text or ""):
            month = _MONTHS.get(word.lower())
            if month:
                full_year = int(year) + 2000 if len(year) == 2 else int(year)
                return date(full_year, month, 1)
    raise ValueError(f"No month found in {texts!r}")


def _to_decimal(value) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    return number.quantize(Decimal("0.01")) if number > 0 else None


@dataclass
class TransportRow:
    state_name: str
    component: str
    mode: str
    label: str
    amount: Decimal


@dataclass
class ParsedRelease:
    month: date
    transport: list[TransportRow] = field(default_factory=list)
    food: list[tuple[str, str, Decimal]] = field(default_factory=list)  # zone, item, price
    skipped_cells: int = 0


def _sheet_rows(sheet) -> list[tuple]:
    return [row for row in sheet.iter_rows(values_only=True) if any(c is not None for c in row)]


def parse_transport_workbook(data: bytes, file_name: str = "") -> ParsedRelease:
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    month = parse_month(*(ws.title for ws in workbook.worksheets), file_name)
    state_sheet = next((ws for ws in workbook.worksheets if "state" in ws.title.lower()), None)
    if state_sheet is None:
        raise ValueError("Transport workbook has no state sheet")

    rows = _sheet_rows(state_sheet)
    header = [str(c or "").strip() for c in rows[0]]
    columns: list[tuple[int, str, str, str]] = []
    for component, mode, keyword in TRANSPORT_COLUMNS:
        index = next((i for i, h in enumerate(header) if keyword in h.lower()), None)
        if index is None:
            raise ValueError(f"Transport workbook has no '{keyword}' column: {header}")
        columns.append((index, component, mode, header[index]))

    release = ParsedRelease(month=month)
    for row in rows[1:]:
        state_name = str(row[0] or "").strip()
        lowered = state_name.lower()
        # Aggregate rows ("Grand Total", "National Average") aren't states.
        if not state_name or "total" in lowered or lowered in {"national", "nigeria", "average"}:
            continue
        for index, component, mode, label in columns:
            amount = _to_decimal(row[index] if index < len(row) else None)
            if amount is None:
                release.skipped_cells += 1
                continue
            release.transport.append(TransportRow(state_name, component, mode, label, amount))
    return release


def parse_food_workbook(data: bytes, file_name: str = "") -> ParsedRelease:
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    month = parse_month(*(ws.title for ws in workbook.worksheets), file_name)
    zone_sheet = next((ws for ws in workbook.worksheets if "zone" in ws.title.lower()), None)
    if zone_sheet is None:
        raise ValueError("Food workbook has no zone sheet")

    rows = _sheet_rows(zone_sheet)
    header = [str(c or "").strip().lower() for c in rows[0]]
    zone_columns = {i: ZONE_HEADERS[h] for i, h in enumerate(header) if h in ZONE_HEADERS}
    if len(zone_columns) != len(ZONE_HEADERS):
        raise ValueError(f"Food workbook zone columns incomplete: {header}")

    release = ParsedRelease(month=month)
    for row in rows[1:]:
        item = re.sub(r"\s+", " ", str(row[0] or "")).strip(" ,")
        if not item:
            continue
        for index, zone in zone_columns.items():
            price = _to_decimal(row[index] if index < len(row) else None)
            if price is None:
                release.skipped_cells += 1
                continue
            release.food.append((zone, item, price))
    return release


def xlsx_from_zip(data: bytes) -> tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        if len(names) != 1:
            raise ValueError(f"Expected one .xlsx in release, found {names}")
        return names[0], archive.read(names[0])


def latest_release_url(catalog_id: int, page_html: str) -> str:
    """The newest zipped release linked from a catalog page (download ids increase over time)."""
    pattern = re.compile(rf"catalog/{catalog_id}/download/(\d+)/([^\"'\s<>]+\.zip)", re.IGNORECASE)
    releases = {int(m.group(1)): m.group(2) for m in pattern.finditer(page_html)}
    if not releases:
        raise ValueError(f"No zipped releases found on catalog {catalog_id}")
    newest = max(releases)
    base = CATALOG_URL.format(catalog_id=catalog_id)
    return f"{base}/download/{newest}/{releases[newest]}"


def download_release(catalog_id: int, url: Optional[str] = None) -> tuple[str, str, bytes]:
    """(release url, xlsx file name, xlsx bytes)."""
    with client(timeout=180) as http:
        if url is None:
            page = http.get(CATALOG_URL.format(catalog_id=catalog_id))
            page.raise_for_status()
            url = latest_release_url(catalog_id, page.text)
        response = http.get(url)
        response.raise_for_status()
    name, data = xlsx_from_zip(response.content)
    return url, name, data


def state_iso_for(name: str, iso_by_slug: dict[str, str]) -> Optional[str]:
    slug = slugify(name)
    return _STATE_ALIASES.get(slug) or iso_by_slug.get(slug)


@dataclass
class NbsReport:
    month: Optional[date] = None
    stored: int = 0
    unknown_states: set[str] = field(default_factory=set)
    skipped_cells: int = 0


async def store_transport(session: AsyncSession, release: ParsedRelease, url: str) -> NbsReport:
    states = (await session.execute(select(State.id, State.iso_code, State.slug))).all()
    iso_by_slug = {slug: iso for _, iso, slug in states}
    id_by_iso = {iso: state_id for state_id, iso, _ in states}

    source = await get_or_create_source(
        session,
        kind="nbs",
        url=url,
        title=f"NBS Transport Fare Watch {release.month:%B %Y}",
        licence=None,
    )
    report = NbsReport(month=release.month, skipped_cells=release.skipped_cells)
    for row in release.transport:
        iso = state_iso_for(row.state_name, iso_by_slug)
        if iso is None or iso not in id_by_iso:
            report.unknown_states.add(row.state_name)
            continue
        stmt = insert(PriceObservation).values(
            component=row.component,
            mode=row.mode,
            unit="per_trip",
            state_id=id_by_iso[iso],
            amount=row.amount,
            currency="NGN",
            observed_on=release.month,
            source_id=source.id,
            notes=row.label,
        )
        result = await session.execute(
            stmt.on_conflict_do_nothing(constraint="uq_price_observations_natural_key")
        )
        report.stored += result.rowcount
    return report


async def store_food(session: AsyncSession, release: ParsedRelease, url: str) -> NbsReport:
    source = await get_or_create_source(
        session,
        kind="nbs",
        url=url,
        title=f"NBS Selected Food Price Watch {release.month:%B %Y}",
        licence=None,
    )
    report = NbsReport(month=release.month, skipped_cells=release.skipped_cells)
    for zone, item, price in release.food:
        stmt = insert(FoodPrice).values(
            zone=zone, item=item, month=release.month, price=price, source_id=source.id
        )
        result = await session.execute(stmt.on_conflict_do_nothing())
        report.stored += result.rowcount
    return report


def summarise(reports: Iterable[NbsReport]) -> str:
    return "; ".join(
        f"{r.month:%Y-%m} stored={r.stored} skipped={r.skipped_cells}"
        + (f" unknown_states={sorted(r.unknown_states)}" if r.unknown_states else "")
        for r in reports
    )
