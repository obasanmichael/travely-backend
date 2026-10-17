"""Monthly climate normals per destination from NASA POWER (no key required).

POWER's climatology grid is 0.5 degrees, so destinations are grouped by grid
cell and each cell is fetched once.
"""
from __future__ import annotations

import calendar
import logging
import time
from dataclasses import dataclass
from typing import Callable

from geoalchemy2.functions import ST_X, ST_Y
from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Destination, DestinationClimate
from pipelines.db import get_or_create_source
from pipelines.http import client

logger = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
GRID_DEGREES = 0.5
MONTH_KEYS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

Cell = tuple[float, float]


@dataclass(frozen=True)
class MonthlyClimate:
    month: int
    temp_mean_c: float
    precip_mm: float


def grid_cell(lon: float, lat: float) -> Cell:
    """Centre of the 0.5-degree cell containing the point."""

    def snap(value: float) -> float:
        return (int(value // GRID_DEGREES) + 0.5) * GRID_DEGREES

    return (snap(lon), snap(lat))


def parse_power(payload: dict) -> list[MonthlyClimate]:
    params = payload["properties"]["parameter"]
    temps, rain = params["T2M"], params["PRECTOTCORR"]
    months = []
    for index, key in enumerate(MONTH_KEYS, start=1):
        if temps.get(key) is None or rain.get(key) is None or temps[key] <= -999 or rain[key] < 0:
            raise ValueError(f"POWER returned no value for {key}")
        # POWER reports mean mm/day; store the monthly total (non-leap year).
        days = calendar.monthrange(2023, index)[1]
        months.append(MonthlyClimate(index, round(temps[key], 1), round(rain[key] * days, 1)))
    return months


def fetch_cell(cell: Cell) -> list[MonthlyClimate]:
    lon, lat = cell
    with client(timeout=90) as http:
        response = http.get(
            POWER_URL,
            params={
                "parameters": "T2M,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "format": "JSON",
            },
        )
        response.raise_for_status()
        return parse_power(response.json())


@dataclass
class ClimateReport:
    destinations: int = 0
    cells_fetched: int = 0
    failed_cells: int = 0


async def fill_climate(
    session: AsyncSession,
    fetcher: Callable[[Cell], list[MonthlyClimate]] = fetch_cell,
    pause_seconds: float = 0.5,
) -> ClimateReport:
    has_climate = exists().where(DestinationClimate.destination_id == Destination.id)
    rows = (
        await session.execute(
            select(
                Destination.id,
                ST_X(func.geometry(Destination.location)),
                ST_Y(func.geometry(Destination.location)),
            ).where(Destination.status != "rejected", ~has_climate)
        )
    ).all()

    by_cell: dict[Cell, list] = {}
    for destination_id, lon, lat in rows:
        by_cell.setdefault(grid_cell(lon, lat), []).append(destination_id)

    report = ClimateReport()
    if not by_cell:
        return report
    source = await get_or_create_source(
        session,
        kind="nasa_power",
        url=POWER_URL,
        title="NASA POWER climatology (T2M, PRECTOTCORR), 0.5 degree grid",
        licence=None,
    )
    for cell, destination_ids in by_cell.items():
        try:
            months = fetcher(cell)
        except Exception as exc:  # one bad cell shouldn't sink the whole run
            logger.warning("climate fetch failed for cell %s: %s", cell, exc)
            report.failed_cells += 1
            continue
        report.cells_fetched += 1
        for destination_id in destination_ids:
            for month in months:
                session.add(
                    DestinationClimate(
                        destination_id=destination_id,
                        month=month.month,
                        temp_mean_c=month.temp_mean_c,
                        precip_mm=month.precip_mm,
                        source_id=source.id,
                    )
                )
            report.destinations += 1
        await session.flush()
        time.sleep(pause_seconds)
    return report
