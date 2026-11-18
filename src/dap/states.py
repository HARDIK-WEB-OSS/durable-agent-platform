from collections.abc import Mapping
from enum import StrEnum
from typing import TypeVar


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


class AttemptStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LOST = "lost"


RUN_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

STEP_TRANSITIONS: Mapping[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({StepStatus.RUNNING, StepStatus.CANCELLED}),
    StepStatus.RUNNING: frozenset(
        {
            StepStatus.SUCCEEDED,
            StepStatus.PENDING,
            StepStatus.DEAD_LETTER,
            StepStatus.CANCELLED,
        }
    ),
    StepStatus.SUCCEEDED: frozenset(),
    StepStatus.DEAD_LETTER: frozenset(),
    StepStatus.CANCELLED: frozenset(),
}

S = TypeVar("S", RunStatus, StepStatus)


class IllegalTransitionError(Exception):
    def __init__(self, current: StrEnum, target: StrEnum) -> None:
        super().__init__(f"illegal transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


def is_terminal(transitions: Mapping[S, frozenset[S]], status: S) -> bool:
    return not transitions[status]


def check_transition(transitions: Mapping[S, frozenset[S]], current: S, target: S) -> None:
    if target not in transitions[current]:
        raise IllegalTransitionError(current, target)


def allowed_sources(transitions: Mapping[S, frozenset[S]], target: S) -> frozenset[S]:
    return frozenset(src for src, targets in transitions.items() if target in targets)
