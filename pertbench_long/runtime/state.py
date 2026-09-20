"""Episode state machine. Purchases and analysis happen while RUNNING."""

from __future__ import annotations

from enum import Enum

from pertbench_long.errors import InvalidState


class RunState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUBMITTED = "SUBMITTED"
    TERMINATED = "TERMINATED"
    INFRA_ERROR = "INFRA_ERROR"


ALLOWED = {
    RunState.CREATED: {RunState.RUNNING, RunState.INFRA_ERROR, RunState.TERMINATED},
    RunState.RUNNING: {RunState.SUBMITTED, RunState.TERMINATED, RunState.INFRA_ERROR},
    RunState.SUBMITTED: {RunState.INFRA_ERROR},
    RunState.TERMINATED: set(),
    RunState.INFRA_ERROR: set(),
}

PURCHASE_OK = {RunState.RUNNING}
SUBMIT_OK = {RunState.RUNNING}
MUTATE_OK = {RunState.RUNNING}


class StateMachine:
    def __init__(self, state: RunState = RunState.CREATED) -> None:
        self.state = state

    def transition(self, dest: RunState) -> None:
        if dest not in ALLOWED[self.state]:
            raise InvalidState(f"cannot transition {self.state.value} -> {dest.value}")
        self.state = dest

    def require_running(self, action: str) -> None:
        if self.state != RunState.RUNNING:
            raise InvalidState(f"{action} is not allowed in state {self.state.value}")
