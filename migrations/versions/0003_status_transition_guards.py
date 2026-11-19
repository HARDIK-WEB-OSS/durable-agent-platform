"""status transition guards

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-19

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copy of dap.states as of this revision; a test checks the DB still matches it.
RUN_PAIRS = [
    ("pending", "running"),
    ("pending", "cancelled"),
    ("running", "succeeded"),
    ("running", "failed"),
    ("running", "cancelled"),
]
STEP_PAIRS = [
    ("pending", "running"),
    ("pending", "cancelled"),
    ("running", "succeeded"),
    ("running", "pending"),
    ("running", "dead_letter"),
    ("running", "cancelled"),
]


def guard_sql(table: str, pairs: list[tuple[str, str]]) -> list[str]:
    allowed = ", ".join(f"('{src}', '{dst}')" for src, dst in pairs)
    return [
        f"""
        CREATE FUNCTION {table}_guard_transition() RETURNS trigger AS $$
        BEGIN
            IF NEW.status IS DISTINCT FROM OLD.status
               AND (OLD.status::text, NEW.status::text) NOT IN ({allowed}) THEN
                RAISE EXCEPTION 'illegal {table} transition % -> %', OLD.status, NEW.status
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        f"""
        CREATE TRIGGER {table}_guard_transition
        BEFORE UPDATE OF status ON {table}
        FOR EACH ROW EXECUTE FUNCTION {table}_guard_transition()
        """,
    ]


def upgrade() -> None:
    for statement in guard_sql("runs", RUN_PAIRS) + guard_sql("steps", STEP_PAIRS):
        op.execute(statement)


def downgrade() -> None:
    for table in ("steps", "runs"):
        op.execute(f"DROP TRIGGER IF EXISTS {table}_guard_transition ON {table}")
        op.execute(f"DROP FUNCTION IF EXISTS {table}_guard_transition()")
