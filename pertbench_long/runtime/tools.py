"""Versioned JSON tool schemas and a trusted dispatcher. The model never receives a Broker object."""

from __future__ import annotations

import json
from typing import Any, Callable

from pertbench_long.errors import InvalidState, ToolObservationError
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.executor import PythonExecutor

TOOL_SCHEMA_VERSION = "tools_v1"


def tool_schemas(*, allow_purchase: bool, allow_shell: bool = False) -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_evidence",
                "description": "List currently visible evidence artifacts. Use artifact_id with inspect_artifact.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_artifact",
                "description": "Inspect a visible artifact by artifact_id or logical_path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "logical_path": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Run Python in the isolated analysis worker. Code cannot import the oracle or private labels.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_experiments",
                "description": "List purchasable queryable experiments (IDs, cell type, cost).",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_budget",
                "description": "Return remaining experimental credits.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_prediction_snapshot",
                "description": "Freeze a prediction parquet (and optional claims) for the current evidence version.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prediction_path": {"type": "string"},
                        "claims_path": {"type": "string"},
                    },
                    "required": ["prediction_path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submit the final prediction. Only a successful submit receipt completes the episode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prediction_path": {"type": "string"},
                        "claims_path": {"type": "string"},
                        "stop_reason": {"type": "string"},
                    },
                    "required": ["prediction_path", "claims_path"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if allow_purchase:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "request_experiment",
                    "description": "Purchase one queryable experiment. Requires a frozen snapshot for the current evidence version. Serial only.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "experiment_id": {"type": "string"},
                            "request_id": {"type": "string"},
                        },
                        "required": ["experiment_id", "request_id"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if allow_shell:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Optional whitelist shell. Disabled unless explicitly enabled.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


READ_ONLY = frozenset({"list_evidence", "inspect_artifact", "list_experiments", "get_budget"})
MUTATING = frozenset({"run_python", "run_shell", "save_prediction_snapshot", "submit", "request_experiment"})


class ToolRouter:
    def __init__(
        self,
        broker: Broker,
        executor: PythonExecutor,
        *,
        allow_purchase: bool = True,
        allow_shell: bool = False,
        serial_purchase: bool = True,
    ) -> None:
        self.broker = broker
        self.executor = executor
        self.allow_purchase = allow_purchase
        self.allow_shell = allow_shell
        self.serial_purchase = serial_purchase
        self.schemas = tool_schemas(allow_purchase=allow_purchase, allow_shell=allow_shell)
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "list_evidence": lambda a: broker.list_evidence(),
            "inspect_artifact": lambda a: broker.inspect_artifact(a.get("artifact_id") or a.get("logical_path") or ""),
            "run_python": lambda a: broker.run_python(str(a.get("code") or ""), self.executor),
            "list_experiments": lambda a: broker.list_experiments(),
            "get_budget": lambda a: broker.get_budget(),
            "save_prediction_snapshot": lambda a: broker.save_snapshot(str(a["prediction_path"]), a.get("claims_path")),
            "submit": lambda a: broker.submit(str(a["prediction_path"]), str(a["claims_path"]), stop_reason=str(a.get("stop_reason") or "submitted")),
            "request_experiment": lambda a: broker.request_experiment(str(a["experiment_id"]), str(a["request_id"])),
            "run_shell": lambda a: broker.run_shell(str(a.get("command") or ""), self.executor),
        }

    def allowed_names(self) -> set[str]:
        names = {t["function"]["name"] for t in self.schemas}
        return names

    def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        if name not in self.allowed_names():
            return {"status": "error", "error_code": "UNKNOWN_TOOL", "message": f"tool {name!r} is not available", "retryable": False}
        if name == "request_experiment" and not self.allow_purchase:
            return {"status": "error", "error_code": "PURCHASE_DISABLED", "message": "purchase tools are disabled for this policy", "retryable": False}
        try:
            result = self._handlers[name](args)
            if isinstance(result, dict) and result.get("status") == "error":
                return result
            return {"status": "ok", "tool": name, "observation": result}
        except ToolObservationError as exc:
            return {"status": "error", "error_code": exc.error_code, "message": exc.message, "retryable": False}
        except InvalidState as exc:
            return {"status": "error", "error_code": exc.error_code, "message": exc.message, "retryable": False}
        except Exception as exc:
            from pertbench_long.errors import PertBenchLongError

            if isinstance(exc, PertBenchLongError):
                return {"status": "error", "error_code": exc.error_code, "message": exc.message, "retryable": False}
            return {"status": "error", "error_code": "TOOL_OBSERVATION_ERROR", "message": type(exc).__name__, "retryable": False}

    def dispatch_many(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Read-only tools may run together; purchases are always serial and exclusive."""
        names = [c.get("name") for c in calls]
        if "request_experiment" in names and len(calls) > 1:
            return [
                {
                    "status": "error",
                    "error_code": "SERIAL_PURCHASE_REQUIRED",
                    "message": "request_experiment cannot share a turn with other tools",
                    "retryable": False,
                }
                for _ in calls
            ]
        return [self.dispatch(str(c.get("name")), c.get("arguments") or {}) for c in calls]
