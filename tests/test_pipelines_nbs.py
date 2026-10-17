from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import date
from decimal import Decimal

import openpyxl
import pytest
from sqlalchemy import func, select

from db.models import FoodPrice, PriceObservation
from pipelines.db import session_scope
from pipelines.geography import seed_geography
from pipelines.nbs import (
    latest_release_url,
    parse_food_workbook,
    parse_month,
    parse_transport_workbook,
    store_food,
    store_transport,
    xlsx_from_zip,
)
from tests.conftest import TEST_DATABASE_URL, requires_db

TRANSPORT_HEADER = [
    "State",
    "Air fare charg.for specified routes single journey",
    "Bus journey intercity, state route, charg. per per",
    "Bus journey within  city , per  drop constant  rou",
    "Journey by motorcycle (okada) per drop",
    "Water transport : water way passenger  transportat",
]


def _workbook(sheets: dict[str, list[list]]) -> bytes:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _transport_xlsx() -> bytes:
    return _workbook({
        "Transport May 2026": [["Zone", "Average of May-26"], ["NORTH CENTRAL", 1]],
        "State Transport": [
            TRANSPORT_HEADER,
            ["LAGOS", 160000, 9500.123, 1500, 1000, 2000],
            ["ABUJA", 160006.4, 11024.93, 1409.81, 1081.31, None],
            ["NASSARAWA", 150000, "n/a", 900, 700, 0],
            ["Atlantis", 1, 1, 1, 1, 1],
            ["Grand Total", 155000, 9700, 1400, 1000, 2000],
        ],
    })


def _food_xlsx() -> bytes:
    zones = ["NORTH CENTRAL", "NORTH EAST", "NORTH WEST", "SOUTH EAST", "SOUTH SOUTH", "SOUTH WEST"]
    return _workbook({
        "Selected Food May 2026": [["Item Label", "Average of May-26"]],
        "Zone All item": [
            ["Item Label", *zones],
            ["Rice Local, short-Grained", 1000, 1100, 900, 1200, 1250, 1300],
            ["Beef Boneless ", 7000, None, 6100, 8400, 7700, 6500],
        ],
    })


@pytest.mark.parametrize(
    "texts, expected",
    [
        (("Transport May 2026",), date(2026, 5, 1)),
        (("", "Transport_Report_MAR26.zip"), date(2026, 3, 1)),
        (("Selected Food Sept 2025",), date(2025, 9, 1)),
    ],
)
def test_parse_month(texts, expected):
    assert parse_month(*texts) == expected


def test_parse_transport_skips_blanks_and_non_numbers():
    release = parse_transport_workbook(_transport_xlsx())
    assert release.month == date(2026, 5, 1)
    lagos_bus = next(r for r in release.transport if r.state_name == "LAGOS" and r.mode == "bus"
                     and r.component == "intercity_transport")
    assert lagos_bus.amount == Decimal("9500.12")
    # ABUJA water (None), NASSARAWA intercity ("n/a") and water (0)
    assert release.skipped_cells == 3


def test_parse_transport_fails_loudly_when_a_column_disappears():
    data = _workbook({"State Transport May 2026": [["State", "Air fare"], ["LAGOS", 1]]})
    with pytest.raises(ValueError, match="intercity"):
        parse_transport_workbook(data)


def test_parse_food_reads_every_zone():
    release = parse_food_workbook(_food_xlsx())
    assert release.month == date(2026, 5, 1)
    assert ("south_west", "Rice Local, short-Grained", Decimal("1300.00")) in release.food
    assert ("north_east", "Beef Boneless", Decimal("0")) not in release.food
    assert release.skipped_cells == 1


def test_xlsx_is_extracted_from_the_release_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("report.pdf", b"%PDF")
        archive.writestr("TRANSPORT COST Watch MAY_2026_.xlsx", b"xlsx-bytes")
    assert xlsx_from_zip(buffer.getvalue()) == ("TRANSPORT COST Watch MAY_2026_.xlsx", b"xlsx-bytes")


def test_latest_release_is_the_highest_download_id():
    page = (
        '<a href="https://microdata.nigerianstat.gov.ng/index.php/catalog/161/download/1414/A.zip">'
        '<a href="https://microdata.nigerianstat.gov.ng/index.php/catalog/161/download/1429/B.zip">'
        '<a href="https://microdata.nigerianstat.gov.ng/index.php/catalog/161/download/1237/C.xlsx">'
    )
    assert latest_release_url(161, page).endswith("/download/1429/B.zip")


GEO = {
    "type": "FeatureCollection",
    "features": [
        {"properties": {"shapeISO": iso, "shapeName": name},
         "geometry": {"type": "Polygon", "coordinates": [[[x, 6], [x + 1, 6], [x + 1, 7], [x, 7], [x, 6]]]}}
        for x, (iso, name) in enumerate(
            [("NG-LA", "Lagos"), ("NG-FC", "Abuja Federal Capital Territory"), ("NG-NA", "Nasarawa")]
        )
    ],
}


@requires_db
def test_store_is_idempotent_and_reports_unknown_states(empty_db):
    transport = parse_transport_workbook(_transport_xlsx())
    food = parse_food_workbook(_food_xlsx())

    async def _run():
        async with session_scope(TEST_DATABASE_URL) as session:
            await seed_geography(session, GEO)
            first = await store_transport(session, transport, "https://nbs/t.zip")
            await store_food(session, food, "https://nbs/f.zip")
        async with session_scope(TEST_DATABASE_URL) as session:
            again = await store_transport(session, transport, "https://nbs/t.zip")
            food_again = await store_food(session, food, "https://nbs/f.zip")
            observations = await session.scalar(select(func.count()).select_from(PriceObservation))
            food_rows = await session.scalar(select(func.count()).select_from(FoodPrice))
        return first, again, food_again, observations, food_rows

    first, again, food_again, observations, food_rows = asyncio.run(_run())
    assert first.unknown_states == {"Atlantis"}
    assert first.stored == observations == 5 + 4 + 3
    assert again.stored == 0
    assert food_again.stored == 0
    assert food_rows == 11
