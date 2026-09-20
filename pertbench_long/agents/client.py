"""OpenAI-compatible chat transport. Keys are never logged. No silent parameter stripping."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urljoin

from pertbench_long.errors import ConfigError, TransportError
from pertbench_long.hashes import sha256_json
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.config import first_defined


def chat_completions_url(base_url: str, endpoint: str | None = None) -> str:
    if endpoint:
        text = endpoint.rstrip("/")
        if text.endswith("/chat/completions"):
            return text
        return text
    base = (base_url or "").rstrip("/")
    if not base:
        raise ConfigError("model.base_url is required")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


@dataclass
class ModelCapabilities:
    token_parameter: str = "max_tokens"
    supports_temperature: bool = True
    supports_tools: bool = True
    supports_tool_choice: bool = True
    supports_parallel_calls: bool = False
    supports_seed: bool = True
    action_format: str = "native_tools"  # native_tools | json_action


@dataclass
class NormalizedResponse:
    assistant_message: dict[str, Any]
    actions: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, Any]
    provider_request_id: str | None
    raw: dict[str, Any]


class ModelClient:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        raw_model = cfg.get("model")
        if isinstance(raw_model, dict):
            model_cfg = dict(raw_model)
        else:
            model_cfg = dict(cfg)
        self.model = model_cfg.get("model") if isinstance(model_cfg.get("model"), str) else (raw_model if isinstance(raw_model, str) else model_cfg.get("name"))
        self.backend = model_cfg.get("backend", "openai_compatible_chat")
        if self.backend != "openai_compatible_chat":
            raise ConfigError(f"unsupported backend {self.backend!r}; only openai_compatible_chat is implemented")
        self.base_url = model_cfg.get("base_url")
        self.endpoint = model_cfg.get("endpoint")
        self.url = chat_completions_url(self.base_url or "", self.endpoint)
        self.auth_mode = model_cfg.get("auth_mode") or ("bearer_env" if model_cfg.get("api_key_env") else "none")
        if self.auth_mode not in {"none", "bearer_env"}:
            raise ConfigError("auth_mode must be none or bearer_env")
        self.api_key_env = model_cfg.get("api_key_env")
        self.temperature = first_defined(model_cfg.get("temperature"), cfg.get("temperature"), default=0.0)
        self.max_output_tokens = first_defined(
            model_cfg.get("max_output_tokens_per_call"),
            model_cfg.get("max_tokens"),
            cfg.get("max_output_tokens_per_call"),
            default=2048,
        )
        self.request_timeout_s = int(first_defined(model_cfg.get("request_timeout_s"), cfg.get("request_timeout_s"), default=120))
        self.max_retries = int(first_defined(model_cfg.get("max_retries"), cfg.get("max_retries"), default=2))
        cap = model_cfg.get("capabilities") or {}
        self.capabilities = ModelCapabilities(
            token_parameter=str(model_cfg.get("token_parameter") or cap.get("token_parameter") or "max_tokens"),
            supports_temperature=bool(cap.get("supports_temperature", model_cfg.get("temperature", 0) is not None)),
            supports_tools=bool(cap.get("supports_tools", True)),
            supports_tool_choice=bool(cap.get("supports_tool_choice", True)),
            supports_parallel_calls=bool(cap.get("supports_parallel_calls", False)),
            supports_seed=bool(cap.get("supports_seed", True)),
            action_format=str(model_cfg.get("action_format") or cap.get("action_format") or "native_tools"),
        )
        self.seed = first_defined(model_cfg.get("seed"), cfg.get("seed"))
        seed_status = "sent" if (self.capabilities.supports_seed and self.seed is not None) else (
            "unsupported" if (self.seed is not None and not self.capabilities.supports_seed) else "unset"
        )
        self.resolved_profile = {
            "backend": self.backend,
            "url": self.url,
            "model": self.model,
            "auth_mode": self.auth_mode,
            "api_key_env": self.api_key_env if self.auth_mode == "bearer_env" else None,
            "action_format": self.capabilities.action_format,
            "token_parameter": self.capabilities.token_parameter,
            "temperature": self.temperature,
            "max_output_tokens_per_call": self.max_output_tokens,
            "seed": self.seed if seed_status == "sent" else None,
            "seed_status": seed_status,
        }
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "unknown_usage_calls": 0}
        self.last_requested_output_tokens = None
        if not self.model:
            raise ConfigError("model name is required")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_mode == "none":
            return headers
        if not self.api_key_env:
            raise ConfigError("bearer_env requires api_key_env; OPENAI_API_KEY is not used as a fallback")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(f"environment variable {self.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None,
        *,
        output_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        cap = self.capabilities
        if cap.supports_temperature and self.temperature is not None:
            payload["temperature"] = self.temperature
        token_limit = first_defined(output_tokens, self.max_output_tokens)
        if cap.token_parameter and token_limit is not None:
            payload[cap.token_parameter] = int(token_limit)
        if cap.supports_seed and self.seed is not None:
            payload["seed"] = int(self.seed)
        if cap.action_format == "native_tools" and cap.supports_tools and tool_schemas:
            payload["tools"] = tool_schemas
            if cap.supports_tool_choice:
                payload["tool_choice"] = "auto"
        return payload

    def generate(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]] | None = None, *, budget: RuntimeBudget | None = None) -> NormalizedResponse:
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                output_tokens = self.max_output_tokens
                timeout_s = self.request_timeout_s
                if budget is not None:
                    budget.check(next_model=True)
                    output_tokens = budget.output_cap_for_call(self.max_output_tokens)
                    timeout_s = budget.http_timeout_s(self.request_timeout_s)
                self.last_requested_output_tokens = output_tokens
                payload = self._payload(messages, tool_schemas, output_tokens=output_tokens)
                if self.capabilities.action_format == "json_action":
                    payload.setdefault("messages", messages)
                body = json.dumps(payload).encode("utf-8")
                raw = self._post(body, timeout_s=timeout_s)
                return self._normalize(raw)
            except TransportError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_retries:
                    if isinstance(exc, TransportError):
                        exc.details = {**exc.details, "retries": attempts - 1, "retryable": exc.retryable}
                    raise
                delay = min(2 ** attempt, 8)
                if budget is not None:
                    delay = min(delay, max(0.0, budget.remaining_s()))
                    if delay <= 0:
                        raise TimeoutError("episode_timeout") from exc
                time.sleep(delay)
        raise last_error or TransportError("request failed")

    def _post(self, body: bytes, *, timeout_s: int | None = None) -> dict[str, Any]:
        req = urllib.request.Request(self.url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=int(first_defined(timeout_s, self.request_timeout_s))) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                if not isinstance(raw, dict):
                    raise TransportError("provider returned a non-object body")
                return raw
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            retryable = status == 429 or 500 <= status < 600
            if status in {400, 401, 403, 404}:
                retryable = False
            raise TransportError(f"HTTP {status}", retryable=retryable, details={"status": status}) from exc
        except TimeoutError as exc:
            raise TransportError("timeout", retryable=True, details={"status": None}) from exc
        except urllib.error.URLError as exc:
            raise TransportError(f"connection error: {exc.reason}", retryable=True, details={"status": None}) from exc

    def _normalize(self, raw: dict[str, Any]) -> NormalizedResponse:
        if "error" in raw and "choices" not in raw:
            raise TransportError(str(raw.get("error")), retryable=False)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TransportError("missing choices", retryable=False)
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise TransportError("assistant message is not an object", retryable=False)
        finish = choices[0].get("finish_reason")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        if not usage:
            self.usage_totals["unknown_usage_calls"] += 1
            usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "estimated": True}
        else:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    self.usage_totals[key] = int(self.usage_totals.get(key) or 0) + value
        actions = []
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                fn = (call or {}).get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    parsed_args = {"_parse_error": True, "raw": args_raw}
                if not isinstance(parsed_args, dict):
                    parsed_args = {"_parse_error": True, "raw": parsed_args}
                actions.append({"id": call.get("id"), "name": fn.get("name"), "arguments": parsed_args})
        if self.capabilities.action_format == "json_action" and not actions:
            content = message.get("content") or ""
            parsed = parse_json_action(content if isinstance(content, str) else "")
            if parsed:
                actions.append({"id": "json_action", "name": parsed["action"], "arguments": parsed.get("arguments") or {}})
        assistant = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }
        return NormalizedResponse(
            assistant_message=assistant,
            actions=actions,
            finish_reason=finish,
            usage=usage,
            provider_request_id=raw.get("id"),
            raw=raw,
        )


def parse_json_action(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    blob = text.strip()
    if "```" in blob:
        parts = blob.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                blob = part
                break
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        obj = json.loads(blob[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "action" not in obj:
        return None
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {"_parse_error": True, "raw": arguments}
    return {"action": obj["action"], "arguments": arguments}


def json_action_instruction(tool_schemas: list[dict[str, Any]] | list[str]) -> str:
    if tool_schemas and isinstance(tool_schemas[0], str):
        names = list(tool_schemas)
        return (
            "Return a single JSON object {\"action\": <name>, \"arguments\": {...}} with no extra keys. "
            f"Allowed actions: {', '.join(names)}. Do not use eval. Text-only completion is not a submit."
        )
    compact = []
    for item in tool_schemas:
        fn = (item or {}).get("function") or item
        compact.append({"name": fn.get("name"), "description": fn.get("description"), "parameters": fn.get("parameters")})
    return (
        "Return a single JSON object {\"action\": <name>, \"arguments\": {...}} with no extra keys. "
        "arguments must be a JSON object matching the tool parameters schema. "
        f"Tools: {json.dumps(compact)}. Do not use eval. Text-only completion is not a submit."
    )
