import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, ClassVar

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from dap.states import AttemptStatus, RunStatus, StepStatus

JsonDict = dict[str, Any]


def status_column(enum_cls: type[StrEnum], name: str) -> Enum:
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda members: [m.value for m in members],
    )


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        JsonDict: JSONB,
        datetime: DateTime(timezone=True),
    }


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task: Mapped[str] = mapped_column(Text)
    input: Mapped[JsonDict] = mapped_column(default=dict)
    status: Mapped[RunStatus] = mapped_column(
        status_column(RunStatus, "run_status"), default=RunStatus.PENDING, index=True
    )
    budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    result: Mapped[JsonDict | None]
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Step(Base):
    __tablename__ = "steps"
    __table_args__ = (
        UniqueConstraint("run_id", "seq"),
        Index(
            "ix_steps_claimable",
            "available_at",
            postgresql_where="status = 'pending'",
        ),
        Index("ix_steps_lease_expiry", "lease_expires_at", postgresql_where="status = 'running'"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(64))
    input: Mapped[JsonDict] = mapped_column(default=dict)
    output: Mapped[JsonDict | None]
    status: Mapped[StepStatus] = mapped_column(
        status_column(StepStatus, "step_status"), default=StepStatus.PENDING
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    available_at: Mapped[datetime] = mapped_column(server_default=func.now())
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None]
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class StepAttempt(Base):
    __tablename__ = "step_attempts"
    __table_args__ = (UniqueConstraint("step_id", "attempt_no"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer)
    worker_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[AttemptStatus] = mapped_column(
        status_column(AttemptStatus, "attempt_status"), default=AttemptStatus.RUNNING
    )
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None]


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), index=True)
    step_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("steps.id"))
    type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[JsonDict] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), index=True)
    result: Mapped[JsonDict]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class DemoSideEffect(Base):
    # No unique constraint on purpose: duplicated side effects must stay detectable.
    __tablename__ = "demo_side_effects"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), index=True)
    step_seq: Mapped[int] = mapped_column(Integer)
    worker_id: Mapped[str] = mapped_column(String(128))
    attempt: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Vector(UserDefinedType[list[float]]):
    """pgvector column type, declared for the schema only.

    Queries pass vectors as text and cast in SQL (CAST(CAST(:v AS text) AS vector)) because
    asyncpg has no codec for the vector type; see agent/tools/search_docs.py.
    """

    cache_ok = True

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({self.dim})"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source", "chunk"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    chunk: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EvalResult(Base):
    """One graded (task, arm) outcome of an evaluation; the router's training data."""

    __tablename__ = "eval_results"
    __table_args__ = (UniqueConstraint("eval_name", "task_id", "arm"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    eval_name: Mapped[str] = mapped_column(String(64))
    task_id: Mapped[str] = mapped_column(String(64))
    dataset: Mapped[str] = mapped_column(String(16))
    split: Mapped[str] = mapped_column(String(8))
    arm: Mapped[str] = mapped_column(String(16))
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"))
    outcome: Mapped[str] = mapped_column(String(16))  # correct | incorrect | error
    answer: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[str] = mapped_column(Text)
    paid_usd: Mapped[float] = mapped_column(Float)
    shadow_usd: Mapped[float] = mapped_column(Float)
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    llm_calls: Mapped[int] = mapped_column(Integer)
    tool_calls: Mapped[int] = mapped_column(Integer)
    turns: Mapped[int | None] = mapped_column(Integer)
    llm_s: Mapped[float] = mapped_column(Float)
    tool_s: Mapped[float] = mapped_column(Float)
    elapsed_s: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class RouterDecision(Base):
    __tablename__ = "router_decisions"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), index=True)
    step_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("steps.id"))
    policy: Mapped[str] = mapped_column(String(64))
    arm: Mapped[str] = mapped_column(String(128))
    propensity: Mapped[float] = mapped_column(Float)
    context: Mapped[JsonDict] = mapped_column(default=dict)
    reward: Mapped[float | None] = mapped_column(Float)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    rewarded_at: Mapped[datetime | None]
