"""Provider-agnostic LLM adapter. Keys come from the host environment and are never logged.

Live API calls are unverified unless the user supplies credentials and explicitly runs them.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from pertbench_long.errors import IsolationUnavailable


class LLMAdapter:
    name = "llm_adapter"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.model = cfg.get("model", "unspecified")
        self.endpoint = cfg.get("endpoint") or os.environ.get("PERTBENCH_LONG_LLM_ENDPOINT")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        self.api_key_env = cfg.get("api_key_env", "PERTBENCH_LONG_API_KEY")

    def credentials_present(self) -> bool:
        return bool(os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY"))

    def complete(self, messages: list[dict[str, str]], tools: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        if not self.credentials_present() or not self.endpoint:
            return {
                "status": "unverified",
                "message": "adapter implemented, live run unverified",
                "model": self.model,
            }
        try:
            import urllib.request

            key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
            payload = json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return {"status": "ok", "response": body}
        except Exception as exc:
            return {"status": "error", "message": type(exc).__name__, "retryable": True}

    def run(self, broker) -> None:
        raise IsolationUnavailable("LLM live run is not executed unless credentials and endpoint are configured")
