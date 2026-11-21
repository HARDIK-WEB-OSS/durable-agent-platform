"""eval results

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "eval_results",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("eval_name", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("dataset", sa.String(length=16), nullable=False),
        sa.Column("split", sa.String(length=8), nullable=False),
        sa.Column("arm", sa.String(length=16), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("paid_usd", sa.Float(), nullable=False),
        sa.Column("shadow_usd", sa.Float(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("llm_calls", sa.Integer(), nullable=False),
        sa.Column("tool_calls", sa.Integer(), nullable=False),
        sa.Column("turns", sa.Integer(), nullable=True),
        sa.Column("llm_s", sa.Float(), nullable=False),
        sa.Column("tool_s", sa.Float(), nullable=False),
        sa.Column("elapsed_s", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("eval_name", "task_id", "arm"),
        sa.CheckConstraint(
            "outcome IN ('correct', 'incorrect', 'error')", name="ck_eval_results_outcome"
        ),
    )


def downgrade() -> None:
    op.drop_table("eval_results")
