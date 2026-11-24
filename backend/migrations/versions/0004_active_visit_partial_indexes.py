"""partial indexes on the live (active) slice of visits

Only a few hundred of the visits table's rows are ever WAITING / NEEDS_REASSESSMENT; everything else is
history. The queue, the deterioration monitor and the retention job only ask about that live slice, and
were each doing a full sequential scan of visits. See docs/sql-tuning.md for the traces and plans.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-24 00:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0004'
down_revision: Union[str, None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ACTIVE = sa.text("status IN ('WAITING', 'NEEDS_REASSESSMENT')")  # keep in sync with models.ACTIVE_STATUSES


def upgrade() -> None:
    # On a large live Postgres table, run these as CREATE INDEX CONCURRENTLY outside a transaction instead
    # (see docs/runbook.md); here the partial indexes build in well under a second because they are tiny.
    op.create_index('ix_visits_active_queue', 'visits', ['facility_id', 'final_esi_score', 'arrival_time'],
                    postgresql_where=ACTIVE, sqlite_where=ACTIVE)
    op.create_index('ix_visits_active_patient', 'visits', ['patient_id'],
                    postgresql_where=ACTIVE, sqlite_where=ACTIVE)


def downgrade() -> None:
    op.drop_index('ix_visits_active_patient', table_name='visits')
    op.drop_index('ix_visits_active_queue', table_name='visits')
