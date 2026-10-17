"""price observation subject

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '0004'
down_revision: Union[str, None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_KEY = ['source_id', 'component', 'mode', 'tier', 'state_id', 'city_id', 'destination_id', 'origin_state_id']


def upgrade() -> None:
    op.add_column('price_observations', sa.Column('subject', sa.Text(), nullable=True))
    op.drop_constraint('uq_price_observations_natural_key', 'price_observations', type_='unique')
    op.create_unique_constraint(
        'uq_price_observations_natural_key', 'price_observations',
        _KEY + ['subject', 'observed_on'], postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.drop_constraint('uq_price_observations_natural_key', 'price_observations', type_='unique')
    op.create_unique_constraint(
        'uq_price_observations_natural_key', 'price_observations',
        _KEY + ['observed_on'], postgresql_nulls_not_distinct=True,
    )
    op.drop_column('price_observations', 'subject')
