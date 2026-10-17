"""baseline schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-17 07:06:18.915401
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.create_table('catalog_versions',
    sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
    sa.Column('published_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('published_by', sa.Text(), nullable=False),
    sa.Column('destination_count', sa.Integer(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_catalog_versions'))
    )
    op.create_table('countries',
    sa.Column('code', sa.String(length=2), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.PrimaryKeyConstraint('code', name=op.f('pk_countries')),
    sa.UniqueConstraint('name', name=op.f('uq_countries_name'))
    )
    op.create_table('sources',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('url', sa.Text(), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('licence', sa.Text(), nullable=True),
    sa.Column('retrieved_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind IN ('osm', 'wikidata', 'wikipedia', 'commons', 'nasa_power', 'nbs', 'liteapi', 'official_site', 'manual', 'user_report')", name=op.f('ck_sources_kind')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_sources'))
    )
    op.create_table('tags',
    sa.Column('slug', sa.Text(), nullable=False),
    sa.Column('label', sa.Text(), nullable=False),
    sa.Column('category', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('slug', name=op.f('pk_tags'))
    )
    op.create_table('cpi_index',
    sa.Column('country_code', sa.String(length=2), nullable=False),
    sa.Column('month', sa.Date(), nullable=False),
    sa.Column('all_items', sa.Numeric(precision=10, scale=3), nullable=False),
    sa.Column('food', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('source_id', sa.BigInteger(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['countries.code'], name=op.f('fk_cpi_index_country_code_countries')),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_cpi_index_source_id_sources')),
    sa.PrimaryKeyConstraint('country_code', 'month', name=op.f('pk_cpi_index'))
    )
    op.create_geospatial_table('states',
    sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
    sa.Column('country_code', sa.String(length=2), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('slug', sa.Text(), nullable=False),
    sa.Column('centroid', Geography(geometry_type='POINT', srid=4326, spatial_index=False, from_text='ST_GeogFromText', name='geography'), nullable=True),
    sa.ForeignKeyConstraint(['country_code'], ['countries.code'], name=op.f('fk_states_country_code_countries')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_states')),
    sa.UniqueConstraint('country_code', 'slug', name='uq_states_country_slug')
    )
    op.create_geospatial_table('cities',
    sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
    sa.Column('state_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('slug', sa.Text(), nullable=False),
    sa.Column('location', Geography(geometry_type='POINT', srid=4326, spatial_index=False, from_text='ST_GeogFromText', name='geography'), nullable=True),
    sa.ForeignKeyConstraint(['state_id'], ['states.id'], name=op.f('fk_cities_state_id_states')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_cities')),
    sa.UniqueConstraint('state_id', 'slug', name='uq_cities_state_slug')
    )
    op.create_index(op.f('ix_cities_state_id'), 'cities', ['state_id'], unique=False)
    op.create_table('users',
    sa.Column('firebase_uid', sa.Text(), nullable=False),
    sa.Column('email', sa.Text(), nullable=True),
    sa.Column('display_name', sa.Text(), nullable=True),
    sa.Column('is_anonymous', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('home_state_id', sa.Integer(), nullable=True),
    sa.Column('analytics_consent', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('marketing_consent', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['home_state_id'], ['states.id'], name=op.f('fk_users_home_state_id_states')),
    sa.PrimaryKeyConstraint('firebase_uid', name=op.f('pk_users'))
    )
    op.create_geospatial_table('destinations',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('slug', sa.Text(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('country_code', sa.String(length=2), nullable=False),
    sa.Column('state_id', sa.Integer(), nullable=False),
    sa.Column('city_id', sa.Integer(), nullable=True),
    sa.Column('location', Geography(geometry_type='POINT', srid=4326, spatial_index=False, from_text='ST_GeogFromText', name='geography', nullable=False), nullable=False),
    sa.Column('status', sa.Text(), server_default='candidate', nullable=False),
    sa.Column('osm_id', sa.Text(), nullable=True),
    sa.Column('wikidata_id', sa.Text(), nullable=True),
    sa.Column('wikipedia_url', sa.Text(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('description_source_id', sa.BigInteger(), nullable=True),
    sa.Column('access_notes', sa.Text(), nullable=True),
    sa.Column('last_verified_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('verified_by', sa.Text(), nullable=True),
    sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status <> 'published' OR (last_verified_at IS NOT NULL AND verified_by IS NOT NULL)", name=op.f('ck_destinations_published_requires_verification')),
    sa.CheckConstraint("status IN ('candidate', 'published', 'rejected')", name=op.f('ck_destinations_status')),
    sa.ForeignKeyConstraint(['city_id'], ['cities.id'], name=op.f('fk_destinations_city_id_cities')),
    sa.ForeignKeyConstraint(['country_code'], ['countries.code'], name=op.f('fk_destinations_country_code_countries')),
    sa.ForeignKeyConstraint(['description_source_id'], ['sources.id'], name=op.f('fk_destinations_description_source_id_sources')),
    sa.ForeignKeyConstraint(['state_id'], ['states.id'], name=op.f('fk_destinations_state_id_states')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_destinations')),
    sa.UniqueConstraint('osm_id', name=op.f('uq_destinations_osm_id')),
    sa.UniqueConstraint('slug', name=op.f('uq_destinations_slug')),
    sa.UniqueConstraint('wikidata_id', name=op.f('uq_destinations_wikidata_id'))
    )
    op.create_index(op.f('ix_destinations_city_id'), 'destinations', ['city_id'], unique=False)
    op.create_geospatial_index('ix_destinations_location', 'destinations', ['location'], unique=False, postgresql_using='gist', postgresql_ops={})
    op.create_index(op.f('ix_destinations_state_id'), 'destinations', ['state_id'], unique=False)
    op.create_index('ix_destinations_status', 'destinations', ['status'], unique=False)
    op.create_table('recommendation_runs',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('uid', sa.Text(), nullable=False),
    sa.Column('idempotency_key', sa.Text(), nullable=False),
    sa.Column('inputs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('outputs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('engine_version', sa.Text(), nullable=False),
    sa.Column('catalog_version_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['catalog_version_id'], ['catalog_versions.id'], name=op.f('fk_recommendation_runs_catalog_version_id_catalog_versions')),
    sa.ForeignKeyConstraint(['uid'], ['users.firebase_uid'], name=op.f('fk_recommendation_runs_uid_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_recommendation_runs')),
    sa.UniqueConstraint('uid', 'idempotency_key', name='uq_recommendation_runs_uid_idem')
    )
    op.create_index('ix_recommendation_runs_uid_created', 'recommendation_runs', ['uid', 'created_at'], unique=False)
    op.create_table('destination_climate',
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('month', sa.SmallInteger(), nullable=False),
    sa.Column('temp_mean_c', sa.Numeric(precision=4, scale=1), nullable=False),
    sa.Column('precip_mm', sa.Numeric(precision=6, scale=1), nullable=False),
    sa.Column('source_id', sa.BigInteger(), nullable=False),
    sa.CheckConstraint('month BETWEEN 1 AND 12', name=op.f('ck_destination_climate_month_range')),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_destination_climate_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_destination_climate_source_id_sources')),
    sa.PrimaryKeyConstraint('destination_id', 'month', name=op.f('pk_destination_climate'))
    )
    op.create_table('destination_media',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('url', sa.Text(), nullable=False),
    sa.Column('file_page_url', sa.Text(), nullable=False),
    sa.Column('author', sa.Text(), nullable=False),
    sa.Column('licence', sa.Text(), nullable=False),
    sa.Column('width', sa.Integer(), nullable=True),
    sa.Column('height', sa.Integer(), nullable=True),
    sa.Column('is_primary', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('source_id', sa.BigInteger(), nullable=False),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_destination_media_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_destination_media_source_id_sources')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_destination_media'))
    )
    op.create_index(op.f('ix_destination_media_destination_id'), 'destination_media', ['destination_id'], unique=False)
    op.create_index('uq_destination_media_primary', 'destination_media', ['destination_id'], unique=True, postgresql_where='is_primary')
    op.create_table('destination_tags',
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('tag_slug', sa.Text(), nullable=False),
    sa.Column('weight', sa.Numeric(precision=3, scale=2), nullable=False),
    sa.CheckConstraint('weight > 0 AND weight <= 1', name=op.f('ck_destination_tags_weight_range')),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_destination_tags_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['tag_slug'], ['tags.slug'], name=op.f('fk_destination_tags_tag_slug_tags')),
    sa.PrimaryKeyConstraint('destination_id', 'tag_slug', name=op.f('pk_destination_tags'))
    )
    op.create_table('price_observations',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('component', sa.Text(), nullable=False),
    sa.Column('tier', sa.Text(), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('state_id', sa.Integer(), nullable=True),
    sa.Column('city_id', sa.Integer(), nullable=True),
    sa.Column('destination_id', sa.UUID(), nullable=True),
    sa.Column('origin_state_id', sa.Integer(), nullable=True),
    sa.Column('mode', sa.Text(), nullable=True),
    sa.Column('amount', sa.Numeric(precision=14, scale=2), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('observed_on', sa.Date(), nullable=False),
    sa.Column('source_id', sa.BigInteger(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("component IN ('accommodation', 'food', 'local_transport', 'intercity_transport', 'entry_fee', 'activity')", name=op.f('ck_price_observations_component')),
    sa.CheckConstraint("tier IS NULL OR tier IN ('budget', 'mid', 'premium')", name=op.f('ck_price_observations_tier')),
    sa.CheckConstraint("unit IN ('per_night', 'per_meal', 'per_day', 'per_trip', 'per_person')", name=op.f('ck_price_observations_unit')),
    sa.CheckConstraint('amount > 0', name=op.f('ck_price_observations_amount_positive')),
    sa.CheckConstraint('state_id IS NOT NULL OR city_id IS NOT NULL OR destination_id IS NOT NULL', name=op.f('ck_price_observations_has_place')),
    sa.ForeignKeyConstraint(['city_id'], ['cities.id'], name=op.f('fk_price_observations_city_id_cities')),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_price_observations_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['origin_state_id'], ['states.id'], name=op.f('fk_price_observations_origin_state_id_states')),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_price_observations_source_id_sources')),
    sa.ForeignKeyConstraint(['state_id'], ['states.id'], name=op.f('fk_price_observations_state_id_states')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_price_observations'))
    )
    op.create_index('ix_price_observations_lookup', 'price_observations', ['component', 'state_id', 'city_id', 'tier'], unique=False)
    op.create_table('price_reports',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('uid', sa.Text(), nullable=True),
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('component', sa.Text(), nullable=False),
    sa.Column('amount', sa.Numeric(precision=14, scale=2), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('travelled_on', sa.Date(), nullable=True),
    sa.Column('status', sa.Text(), server_default='pending', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("component IN ('accommodation', 'food', 'local_transport', 'intercity_transport', 'entry_fee', 'activity')", name=op.f('ck_price_reports_component')),
    sa.CheckConstraint("status IN ('pending', 'accepted', 'rejected')", name=op.f('ck_price_reports_status')),
    sa.CheckConstraint('amount > 0', name=op.f('ck_price_reports_amount_positive')),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_price_reports_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['uid'], ['users.firebase_uid'], name=op.f('fk_price_reports_uid_users'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_price_reports'))
    )
    op.create_index(op.f('ix_price_reports_destination_id'), 'price_reports', ['destination_id'], unique=False)
    op.create_table('review_actions',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('action', sa.Text(), nullable=False),
    sa.Column('reviewer_uid', sa.Text(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('changes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("action IN ('approve', 'fix', 'reject', 'unpublish')", name=op.f('ck_review_actions_action')),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_review_actions_destination_id_destinations'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_review_actions'))
    )
    op.create_index(op.f('ix_review_actions_destination_id'), 'review_actions', ['destination_id'], unique=False)
    op.create_table('run_events',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('event', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("event IN ('impression', 'save', 'dismiss', 'detail_view')", name=op.f('ck_run_events_event')),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_run_events_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['run_id'], ['recommendation_runs.id'], name=op.f('fk_run_events_run_id_recommendation_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_run_events'))
    )
    op.create_index(op.f('ix_run_events_run_id'), 'run_events', ['run_id'], unique=False)
    op.create_table('saved_destinations',
    sa.Column('uid', sa.Text(), nullable=False),
    sa.Column('destination_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['destination_id'], ['destinations.id'], name=op.f('fk_saved_destinations_destination_id_destinations'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['uid'], ['users.firebase_uid'], name=op.f('fk_saved_destinations_uid_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('uid', 'destination_id', name=op.f('pk_saved_destinations'))
    )


def downgrade() -> None:
    op.drop_table('saved_destinations')
    op.drop_index(op.f('ix_run_events_run_id'), table_name='run_events')
    op.drop_table('run_events')
    op.drop_index(op.f('ix_review_actions_destination_id'), table_name='review_actions')
    op.drop_table('review_actions')
    op.drop_index(op.f('ix_price_reports_destination_id'), table_name='price_reports')
    op.drop_table('price_reports')
    op.drop_index('ix_price_observations_lookup', table_name='price_observations')
    op.drop_table('price_observations')
    op.drop_table('destination_tags')
    op.drop_index('uq_destination_media_primary', table_name='destination_media', postgresql_where='is_primary')
    op.drop_index(op.f('ix_destination_media_destination_id'), table_name='destination_media')
    op.drop_table('destination_media')
    op.drop_table('destination_climate')
    op.drop_index('ix_recommendation_runs_uid_created', table_name='recommendation_runs')
    op.drop_table('recommendation_runs')
    op.drop_index('ix_destinations_status', table_name='destinations')
    op.drop_index(op.f('ix_destinations_state_id'), table_name='destinations')
    op.drop_geospatial_index('ix_destinations_location', table_name='destinations', postgresql_using='gist', column_name='location')
    op.drop_index(op.f('ix_destinations_city_id'), table_name='destinations')
    op.drop_geospatial_table('destinations')
    op.drop_table('users')
    op.drop_index(op.f('ix_cities_state_id'), table_name='cities')
    op.drop_geospatial_table('cities')
    op.drop_geospatial_table('states')
    op.drop_table('cpi_index')
    op.drop_table('tags')
    op.drop_table('sources')
    op.drop_table('countries')
    op.drop_table('catalog_versions')
