"""Agent protocol. Implementations must not import evaluator private labels."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    name: str

    def run(self, router: Any) -> None:
        ...
