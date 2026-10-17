"""state boundaries and tag metadata

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-17 07:24:57.061753
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography

revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_V1 = (
    "osm", "wikidata", "wikipedia", "commons", "nasa_power", "nbs", "liteapi",
    "official_site", "manual", "user_report",
)
_SOURCE_KINDS_V1 = ", ".join(repr(k) for k in _V1)
_SOURCE_KINDS_V2 = ", ".join(repr(k) for k in _V1 + ("geoboundaries",))


def upgrade() -> None:
    op.add_column('destinations', sa.Column('primary_source_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key(op.f('fk_destinations_primary_source_id_sources'), 'destinations', 'sources', ['primary_source_id'], ['id'])
    op.add_column('states', sa.Column('iso_code', sa.Text(), nullable=False))
    op.add_geospatial_column('states', sa.Column('boundary', Geography(geometry_type='MULTIPOLYGON', srid=4326, spatial_index=False, from_text='ST_GeogFromText', name='geography'), nullable=True))
    op.add_column('states', sa.Column('boundary_source_id', sa.BigInteger(), nullable=True))
    op.create_geospatial_index('ix_states_boundary', 'states', ['boundary'], unique=False, postgresql_using='gist', postgresql_ops={})
    op.create_unique_constraint(op.f('uq_states_iso_code'), 'states', ['iso_code'])
    op.create_foreign_key(op.f('fk_states_boundary_source_id_sources'), 'states', 'sources', ['boundary_source_id'], ['id'])
    op.add_column('tags', sa.Column('is_interest', sa.Boolean(), server_default='true', nullable=False))
    # Autogenerate doesn't see CHECK constraint changes.
    op.create_check_constraint(
        'category', 'tags',
        "category IN ('nature', 'activity', 'culture', 'leisure', 'audience')",
    )
    op.drop_constraint(op.f('ck_sources_kind'), 'sources', type_='check')
    op.create_check_constraint('kind', 'sources', f"kind IN ({_SOURCE_KINDS_V2})")


def downgrade() -> None:
    op.drop_constraint(op.f('ck_tags_category'), 'tags', type_='check')
    op.drop_column('tags', 'is_interest')
    op.drop_constraint(op.f('fk_states_boundary_source_id_sources'), 'states', type_='foreignkey')
    op.drop_constraint(op.f('uq_states_iso_code'), 'states', type_='unique')
    op.drop_geospatial_index('ix_states_boundary', table_name='states', postgresql_using='gist', column_name='boundary')
    op.drop_column('states', 'boundary_source_id')
    op.drop_geospatial_column('states', 'boundary')
    op.drop_column('states', 'iso_code')
    op.drop_constraint(op.f('fk_destinations_primary_source_id_sources'), 'destinations', type_='foreignkey')
    op.drop_column('destinations', 'primary_source_id')
    # Runs after boundary_source_id is gone; sources of the removed kind can't stay.
    op.execute("DELETE FROM sources WHERE kind = 'geoboundaries'")
    op.drop_constraint(op.f('ck_sources_kind'), 'sources', type_='check')
    op.create_check_constraint('kind', 'sources', f"kind IN ({_SOURCE_KINDS_V1})")
