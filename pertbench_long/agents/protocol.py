"""Agent protocol. Implementations must not import evaluator private labels."""

from __future__ import annotations

from typing import Any, Protocol


class Agent(Protocol):
    name: str

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        ...
