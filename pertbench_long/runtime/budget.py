"""Trusted runtime budget checked before each model/tool/submit call."""

from __future__ import annotations

import time
from typing import Any, Optional

from pertbench_long.errors import InvalidState


class RuntimeBudget:
    def __init__(
        self,
        *,
        deadline_monotonic: float | None,
        max_generation_tokens: int | None = None,
        max_tool_calls: int | None = None,
        max_observation_chars: int = 8000,
        strict_token_cap: bool = False,
    ) -> None:
        self.deadline = deadline_monotonic
        self.max_generation_tokens = max_generation_tokens
        self.max_tool_calls = max_tool_calls
        self.max_observation_chars = int(max_observation_chars)
        self.strict_token_cap = bool(strict_token_cap)
        self.tokens_used = 0
        self.unknown_usage_calls = 0
        self.tool_calls = 0
        self.estimated_tokens = 0

    def remaining_s(self) -> float:
        if self.deadline is None:
            return float("inf")
        return max(0.0, float(self.deadline) - time.monotonic())

    def check(self, *, next_model: bool = False, next_tool: bool = False) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("episode_timeout")
        if next_tool and self.max_tool_calls is not None and self.tool_calls >= int(self.max_tool_calls):
            raise InvalidState("external tool call budget exceeded")
        if next_model and self.max_generation_tokens is not None:
            if self.tokens_used + self.estimated_tokens >= int(self.max_generation_tokens):
                raise TimeoutError("token_budget")
            if self.strict_token_cap and self.unknown_usage_calls:
                raise InvalidState("strict token cap cannot be enforced because the provider omitted usage")

    def http_timeout_s(self, configured: int) -> int:
        remaining = self.remaining_s()
        if remaining <= 0:
            raise TimeoutError("episode_timeout")
        configured_i = max(1, int(configured))
        if remaining == float("inf"):
            return configured_i
        return max(1, min(configured_i, int(remaining)))

    def tool_timeout_s(self, configured: int) -> int:
        return self.http_timeout_s(configured)

    def count_tool(self) -> None:
        self.check(next_tool=True)
        self.tool_calls += 1

    def consume_usage(self, usage: dict[str, Any] | None, *, fallback_completion_tokens: int | None = None) -> None:
        usage = usage or {}
        completion = usage.get("completion_tokens")
        prompt = usage.get("prompt_tokens")
        total = usage.get("total_tokens")
        estimated = bool(usage.get("estimated"))
        if estimated or (completion is None and total is None):
            self.unknown_usage_calls += 1
            if fallback_completion_tokens is not None:
                self.estimated_tokens += int(fallback_completion_tokens)
            return
        added = int(total) if isinstance(total, int) else (int(prompt or 0) + int(completion or 0))
        self.tokens_used += max(0, added)

    def clip_observation(self, text: str) -> str:
        if text is None:
            return ""
        raw = str(text)
        if len(raw) <= self.max_observation_chars:
            return raw
        return raw[: self.max_observation_chars] + "\n…[truncated]"

    def snapshot(self) -> dict[str, Any]:
        return {
            "remaining_s": None if self.deadline is None else self.remaining_s(),
            "tokens_used": self.tokens_used,
            "estimated_tokens": self.estimated_tokens,
            "unknown_usage_calls": self.unknown_usage_calls,
            "tool_calls": self.tool_calls,
            "max_generation_tokens": self.max_generation_tokens,
            "max_tool_calls": self.max_tool_calls,
            "strict_token_cap": self.strict_token_cap,
        }
