"""Trusted runtime budget checked before each model/tool/submit call.

Generation budget counts completion tokens only. Prompt/total tokens are
recorded separately. max_total_tokens is an independent optional cap.
"""

from __future__ import annotations

import time
from typing import Any

from pertbench_long.errors import InvalidState


class RuntimeBudget:
    def __init__(
        self,
        *,
        deadline_monotonic: float | None,
        max_generation_tokens: int | None = None,
        max_total_tokens: int | None = None,
        max_tool_calls: int | None = None,
        max_observation_chars: int = 8000,
        strict_token_cap: bool = False,
    ) -> None:
        self.deadline = deadline_monotonic
        self.max_generation_tokens = max_generation_tokens
        self.max_total_tokens = max_total_tokens
        self.max_tool_calls = max_tool_calls
        self.max_observation_chars = int(max_observation_chars)
        self.strict_token_cap = bool(strict_token_cap)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.unknown_usage_calls = 0
        self.tool_calls = 0
        self.estimated_completion_tokens = 0
        self.stop_reason: str | None = None
        # backward-compatible alias used by older tests/docs
        self.tokens_used = 0

    def remaining_s(self) -> float:
        if self.deadline is None:
            return float("inf")
        return max(0.0, float(self.deadline) - time.monotonic())

    def remaining_generation(self) -> int | None:
        if self.max_generation_tokens is None:
            return None
        used = int(self.completion_tokens) + int(self.estimated_completion_tokens)
        return max(0, int(self.max_generation_tokens) - used)

    def output_cap_for_call(self, per_call: int) -> int:
        remaining = self.remaining_generation()
        per_call_i = max(1, int(per_call))
        if remaining is None:
            return per_call_i
        if remaining <= 0:
            self.stop_reason = "token_budget"
            raise TimeoutError("token_budget")
        return max(1, min(per_call_i, remaining))

    def generation_over_cap(self) -> bool:
        if self.max_generation_tokens is None:
            return False
        return int(self.completion_tokens) + int(self.estimated_completion_tokens) > int(self.max_generation_tokens)

    def check(self, *, next_model: bool = False, next_tool: bool = False) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop_reason = "episode_timeout"
            raise TimeoutError("episode_timeout")
        if next_tool and self.max_tool_calls is not None and self.tool_calls >= int(self.max_tool_calls):
            raise InvalidState("external tool call budget exceeded")
        if next_model and self.max_generation_tokens is not None:
            remaining = self.remaining_generation()
            if remaining is not None and remaining <= 0:
                self.stop_reason = "token_budget"
                raise TimeoutError("token_budget")
            if self.strict_token_cap and self.unknown_usage_calls:
                raise InvalidState("strict token cap cannot be enforced because the provider omitted usage")

    def check_after_usage(self) -> None:
        """Call after consume_usage, before dispatching tools. Over-cap cannot submit."""
        if self.generation_over_cap():
            self.stop_reason = "token_budget"
            raise TimeoutError("token_budget")
        if self.max_total_tokens is not None and int(self.total_tokens) > int(self.max_total_tokens):
            self.stop_reason = "token_budget"
            raise TimeoutError("token_budget")

    def http_timeout_s(self, configured: int) -> int:
        remaining = self.remaining_s()
        if remaining <= 0:
            self.stop_reason = "episode_timeout"
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
        if isinstance(prompt, int):
            self.prompt_tokens += prompt
        if estimated or completion is None:
            self.unknown_usage_calls += 1
            reserved = int(fallback_completion_tokens or 0)
            self.estimated_completion_tokens += reserved
            if isinstance(total, int):
                self.total_tokens += total
            else:
                self.total_tokens += int(prompt or 0) + reserved
            self.tokens_used = self.completion_tokens
            if self.strict_token_cap:
                raise InvalidState("strict token cap cannot be enforced because the provider omitted usage")
            return
        self.completion_tokens += int(completion)
        self.tokens_used = self.completion_tokens
        if isinstance(total, int):
            self.total_tokens += total
        else:
            self.total_tokens += int(prompt or 0) + int(completion)

    def restore(self, payload: dict[str, Any] | None, *, remaining_s: float | None = None) -> None:
        if not payload:
            return
        self.completion_tokens = int(payload.get("completion_tokens") or payload.get("tokens_used") or 0)
        self.tokens_used = self.completion_tokens
        self.prompt_tokens = int(payload.get("prompt_tokens") or 0)
        self.total_tokens = int(payload.get("total_tokens") or 0)
        self.estimated_completion_tokens = int(payload.get("estimated_completion_tokens") or payload.get("estimated_tokens") or 0)
        self.unknown_usage_calls = int(payload.get("unknown_usage_calls") or 0)
        self.tool_calls = int(payload.get("tool_calls") or 0)
        if remaining_s is not None:
            self.deadline = time.monotonic() + max(0.0, float(remaining_s))

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
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "tokens_used": self.completion_tokens,
            "estimated_completion_tokens": self.estimated_completion_tokens,
            "estimated_tokens": self.estimated_completion_tokens,
            "unknown_usage_calls": self.unknown_usage_calls,
            "tool_calls": self.tool_calls,
            "max_generation_tokens": self.max_generation_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_tool_calls": self.max_tool_calls,
            "strict_token_cap": self.strict_token_cap,
            "stop_reason": self.stop_reason,
            "unit": "completion_tokens",
        }
