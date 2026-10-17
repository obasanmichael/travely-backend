from __future__ import annotations

from geoalchemy2 import alembic_helpers


def include_object(obj, name, type_, reflected, compare_to):
    # PostGIS owns spatial_ref_sys; never diff or drop it.
    if type_ == "table" and name == "spatial_ref_sys":
        return False
    return alembic_helpers.include_object(obj, name, type_, reflected, compare_to)
