"""zones food prices and observation key

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-17 07:46:03.175201
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ZONES = "'north_central', 'north_east', 'north_west', 'south_east', 'south_south', 'south_west'"


def upgrade() -> None:
    op.create_table('food_prices',
    sa.Column('zone', sa.Text(), nullable=False),
    sa.Column('item', sa.Text(), nullable=False),
    sa.Column('month', sa.Date(), nullable=False),
    sa.Column('price', sa.Numeric(precision=14, scale=2), nullable=False),
    sa.Column('source_id', sa.BigInteger(), nullable=False),
    sa.CheckConstraint('price > 0', name=op.f('ck_food_prices_price_positive')),
    sa.CheckConstraint(f"zone IN ({_ZONES})", name=op.f('ck_food_prices_zone')),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_food_prices_source_id_sources')),
    sa.PrimaryKeyConstraint('zone', 'item', 'month', name=op.f('pk_food_prices'))
    )
    op.create_unique_constraint('uq_price_observations_natural_key', 'price_observations', ['source_id', 'component', 'mode', 'tier', 'state_id', 'city_id', 'destination_id', 'origin_state_id', 'observed_on'], postgresql_nulls_not_distinct=True)
    op.add_column('states', sa.Column('zone', sa.Text(), nullable=True))
    op.create_check_constraint('zone', 'states', f"zone IS NULL OR zone IN ({_ZONES})")


def downgrade() -> None:
    op.drop_constraint(op.f('ck_states_zone'), 'states', type_='check')
    op.drop_column('states', 'zone')
    op.drop_constraint('uq_price_observations_natural_key', 'price_observations', type_='unique')
    op.drop_table('food_prices')
