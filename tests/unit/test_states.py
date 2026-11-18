import itertools

import pytest

from dap.states import (
    RUN_TRANSITIONS,
    STEP_TRANSITIONS,
    IllegalTransitionError,
    RunStatus,
    StepStatus,
    allowed_sources,
    check_transition,
    is_terminal,
)

LEGAL_RUN = {
    (RunStatus.PENDING, RunStatus.RUNNING),
    (RunStatus.PENDING, RunStatus.CANCELLED),
    (RunStatus.RUNNING, RunStatus.SUCCEEDED),
    (RunStatus.RUNNING, RunStatus.FAILED),
    (RunStatus.RUNNING, RunStatus.CANCELLED),
}

LEGAL_STEP = {
    (StepStatus.PENDING, StepStatus.RUNNING),
    (StepStatus.PENDING, StepStatus.CANCELLED),
    (StepStatus.RUNNING, StepStatus.SUCCEEDED),
    (StepStatus.RUNNING, StepStatus.PENDING),
    (StepStatus.RUNNING, StepStatus.DEAD_LETTER),
    (StepStatus.RUNNING, StepStatus.CANCELLED),
}

RUN_PAIRS = list(itertools.product(RunStatus, repeat=2))
STEP_PAIRS = list(itertools.product(StepStatus, repeat=2))


@pytest.mark.parametrize(("current", "target"), RUN_PAIRS)
def test_run_transition(current: RunStatus, target: RunStatus) -> None:
    if (current, target) in LEGAL_RUN:
        check_transition(RUN_TRANSITIONS, current, target)
    else:
        with pytest.raises(IllegalTransitionError):
            check_transition(RUN_TRANSITIONS, current, target)


@pytest.mark.parametrize(("current", "target"), STEP_PAIRS)
def test_step_transition(current: StepStatus, target: StepStatus) -> None:
    if (current, target) in LEGAL_STEP:
        check_transition(STEP_TRANSITIONS, current, target)
    else:
        with pytest.raises(IllegalTransitionError):
            check_transition(STEP_TRANSITIONS, current, target)


def test_terminal_states() -> None:
    terminal_runs = {s for s in RunStatus if is_terminal(RUN_TRANSITIONS, s)}
    terminal_steps = {s for s in StepStatus if is_terminal(STEP_TRANSITIONS, s)}
    assert terminal_runs == {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    assert terminal_steps == {
        StepStatus.SUCCEEDED,
        StepStatus.DEAD_LETTER,
        StepStatus.CANCELLED,
    }


def test_allowed_sources_inverts_the_table() -> None:
    assert allowed_sources(RUN_TRANSITIONS, RunStatus.CANCELLED) == {
        RunStatus.PENDING,
        RunStatus.RUNNING,
    }
    assert allowed_sources(STEP_TRANSITIONS, StepStatus.PENDING) == {StepStatus.RUNNING}
