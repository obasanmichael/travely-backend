from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CATALOG_DF: Optional[pd.DataFrame] = None

REQUIRED_COLUMNS = [
    "state",
    "city",
    "destination",
    "destination_type",
    "climate",
    "best_season",
    "avg_cost_per_day",
    "accommodation_type",
    "nearby_hotel",
    "hotel_price_range",
    "feeding_cost_range",
    "necessities_range",
    "activities",
]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip('"') for col in df.columns]

    if "Main Activities Available" in df.columns:
        df["activities"] = df["Main Activities Available"]
    elif "activities" not in df.columns:
        df["activities"] = ""

    column_mapping = {
        "State": "state",
        "City": "city",
        "Destination Name": "destination",
        "Destination Type": "destination_type",
        "Climate": "climate",
        "Best Season to Visit": "best_season",
        "Least cost per day in Naira": "avg_cost_per_day",
        "Accommodation Type": "accommodation_type",
        "Nearby Hotel": "nearby_hotel",
        "Hotel Price Range (Naira)": "hotel_price_range",
        "Feeding Cost Range (Naira)": "feeding_cost_range",
        "Other Necessities Range (Naira)": "necessities_range",
    }
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=rename_dict)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["avg_cost_per_day"] = pd.to_numeric(df["avg_cost_per_day"], errors="coerce").fillna(20000)

    for col in ["destination_type", "activities"]:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()

    df["destination_type_keywords"] = df["destination_type"]
    df["activities_keywords"] = df["activities"]
    return df


def load_catalog(csv_path: str = "result2.csv") -> pd.DataFrame:
    global _CATALOG_DF

    if not os.path.exists(csv_path):
        logger.error("Catalog file not found: %s", csv_path)
        _CATALOG_DF = pd.DataFrame(columns=REQUIRED_COLUMNS)
        return _CATALOG_DF

    try:
        raw = pd.read_csv(csv_path)
        _CATALOG_DF = _prepare_dataframe(raw)
        logger.info("Loaded %d destinations from catalog", len(_CATALOG_DF))
    except Exception as exc:
        logger.exception("Failed to load catalog: %s", exc)
        _CATALOG_DF = pd.DataFrame(columns=REQUIRED_COLUMNS)

    return _CATALOG_DF


def get_catalog_df() -> pd.DataFrame:
    global _CATALOG_DF
    if _CATALOG_DF is None:
        return load_catalog()
    return _CATALOG_DF.copy()
