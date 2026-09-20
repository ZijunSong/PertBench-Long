"""Trusted AgentLoop: model → tool → observation until submit. Private labels never enter messages."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pertbench_long.agents.client import ModelClient, json_action_instruction, parse_json_action
from pertbench_long.errors import TransportError
from pertbench_long.evaluation.contract import CONTRACT_FILENAME, contract_prompt_block, public_submission_contract
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.tools import ToolRouter, TOOL_SCHEMA_VERSION


def public_system_prompt(
    public: dict[str, Any],
    *,
    allow_purchase: bool,
    workspace: str,
    gene_count: int,
    contract: dict[str, Any] | None = None,
) -> str:
    tools = "list_evidence, inspect_artifact, run_python, list_experiments, get_budget, save_prediction_snapshot, submit"
    if allow_purchase:
        tools += ", request_experiment"
    contract = contract or public_submission_contract(effect_unit=str((public.get("canonical_units") or {}).get("effect") or "log1p_mean_diff"))
    return (
        "You are a scientific agent in PertBench-Long. Predict held-out perturbation effects.\n"
        f"Objective: {public.get('objective')}\n"
        f"Episode: {public.get('episode_id')} protocol={public.get('protocol')} budget={public.get('experimental_budget')} credits.\n"
        f"Targets: {public.get('targets')}\n"
        f"Gene universe size G={gene_count}. Write a full target×gene parquet via run_python; do not paste matrices into chat.\n"
        f"Visible workspace: {workspace}. Prefer relative paths such as outputs/predictions.parquet; "
        f"/workspace/... is the same tree. Host absolute paths are rejected. Tools: {tools}. Schema {TOOL_SCHEMA_VERSION}.\n"
        f"{contract_prompt_block(contract)}\n"
        "A sentence like 'I am done' is not a submission. Only the submit tool issues a receipt.\n"
        "Before purchasing, freeze a snapshot for the current evidence version. Purchases are serial.\n"
        "Do not invent private labels or target outcomes."
    )


class AgentLoop:
    def __init__(
        self,
        client: ModelClient,
        router: ToolRouter,
        *,
        public_spec,
        workspace: Path,
        allow_purchase: bool,
        max_steps: int = 60,
        deadline_monotonic: float | None = None,
        max_json_retries: int = 2,
        history_path: Path | None = None,
        runtime_budget: RuntimeBudget | None = None,
        seed_status: str | None = None,
    ) -> None:
        self.client = client
        self.router = router
        self.public_spec = public_spec
        self.workspace = Path(workspace)
        self.allow_purchase = allow_purchase
        self.max_steps = int(max_steps)
        self.deadline = deadline_monotonic
        self.max_json_retries = max_json_retries
        self.history_path = Path(history_path) if history_path else self.workspace.parent / "messages.jsonl"
        self.submitted = False
        self.stop_reason = None
        self.runtime_budget = runtime_budget
        self.seed_status = seed_status

    def _check_deadline(self) -> None:
        if self.runtime_budget is not None:
            self.runtime_budget.check()
            return
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError("episode_timeout")

    def _checkpoint_path(self) -> Path:
        return self.workspace.parent / "agent_messages.json"

    def _save_messages(self, messages: list[dict[str, Any]]) -> None:
        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"schema": "agent_messages_v1", "messages": messages}, default=str), encoding="utf-8")
        tmp.replace(path)

    def _load_or_init_messages(self, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
        path = self._checkpoint_path()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            saved = payload.get("messages") if isinstance(payload, dict) else None
            if payload.get("schema") == "agent_messages_v1" and isinstance(saved, list) and saved:
                return saved
        self._save_messages(default)
        return default

    def _persist_budget(self) -> None:
        broker = getattr(self.router, "broker", None)
        if broker is not None and hasattr(broker, "persist_progress"):
            broker.persist_progress()

    def _append_history(self, item: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, default=str) + "\n")

    def _clip(self, payload: Any) -> Any:
        text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
        if self.runtime_budget is not None:
            return self.runtime_budget.clip_observation(text)
        if len(text) > 8000:
            return text[:8000] + "\n…[truncated]"
        return text

    def _tool_result_message(self, action: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        content = self._clip(obs)
        if self.client.capabilities.action_format == "native_tools":
            return {"role": "tool", "tool_call_id": action.get("id") or action.get("name") or "tool", "content": content if isinstance(content, str) else json.dumps(content)}
        return {"role": "user", "content": json.dumps({"tool_result": obs if isinstance(obs, dict) else json.loads(content)})}

    def _malformed_obs(self, message: str) -> dict[str, Any]:
        return {"status": "error", "error_code": "MALFORMED_ARGUMENTS", "message": message, "retryable": True}

    def run(self) -> dict[str, Any]:
        public = self.public_spec.to_dict() if hasattr(self.public_spec, "to_dict") else dict(self.public_spec)
        genes_path = self.workspace / public.get("gene_universe_artifact", "genes_v1.tsv")
        gene_count = max(0, len(genes_path.read_text(encoding="utf-8").splitlines()) - 1) if genes_path.exists() else 0
        contract_path = self.workspace / CONTRACT_FILENAME
        if contract_path.exists():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        else:
            contract = public_submission_contract(effect_unit=str((public.get("canonical_units") or {}).get("effect") or "log1p_mean_diff"))
        workspace_label = "/workspace" if getattr(self.router.executor, "backend", "") == "docker" else str(self.workspace)
        system = public_system_prompt(
            public,
            allow_purchase=self.allow_purchase,
            workspace=workspace_label,
            gene_count=gene_count,
            contract=contract,
        )
        if self.client.capabilities.action_format == "json_action":
            system += "\n" + json_action_instruction(self.router.schemas)
        initial_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "initial_evidence": public.get("initial_evidence"),
                        "reference_evidence": public.get("reference_evidence"),
                        "candidate_experiments": public.get("candidate_experiments"),
                        "budget": public.get("experimental_budget"),
                        "submission_contract": CONTRACT_FILENAME,
                        "output": "outputs/predictions.parquet plus outputs/claims.json then submit",
                    }
                ),
            },
        ]
        messages = self._load_or_init_messages(initial_messages)
        json_retries = 0
        tools = self.router.schemas if self.client.capabilities.action_format == "native_tools" else None
        steps = max(0, self.max_steps)
        for step in range(steps):
            self._check_deadline()
            try:
                if self.runtime_budget is not None:
                    self.runtime_budget.check(next_model=True)
                response = self.client.generate(messages, tools, budget=self.runtime_budget)
            except TransportError:
                raise
            if self.runtime_budget is not None:
                reserved = None
                if self.client.last_requested_output_tokens is not None:
                    reserved = self.client.last_requested_output_tokens
                else:
                    reserved = self.client.max_output_tokens
                self.runtime_budget.consume_usage(response.usage, fallback_completion_tokens=reserved)
                self._persist_budget()
                try:
                    self.runtime_budget.check_after_usage()
                except TimeoutError:
                    self.stop_reason = "token_budget"
                    assistant = {k: v for k, v in response.assistant_message.items() if v is not None}
                    messages.append(assistant if assistant.get("role") else {"role": "assistant", **assistant})
                    if self.client.capabilities.action_format == "native_tools":
                        for action in response.actions:
                            messages.append(
                                self._tool_result_message(
                                    action,
                                    {"status": "error", "error_code": "TOKEN_BUDGET", "message": "generation budget exceeded before tool execution", "retryable": False},
                                )
                            )
                    self._save_messages(messages)
                    raise
            self._append_history({"step": step, "response": response.assistant_message, "finish_reason": response.finish_reason, "usage": response.usage})
            assistant = {k: v for k, v in response.assistant_message.items() if v is not None}
            messages.append(assistant if assistant.get("role") else {"role": "assistant", **assistant})
            self._save_messages(messages)
            native = self.client.capabilities.action_format == "native_tools"
            if response.finish_reason == "length":
                if native and assistant.get("tool_calls"):
                    for call in assistant.get("tool_calls") or []:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": (call or {}).get("id") or "truncated",
                                "content": json.dumps(self._malformed_obs("tool call was truncated; resend complete arguments")),
                            }
                        )
                messages.append({"role": "user", "content": "Your previous output was truncated. Continue with a valid tool call. Text is not a submit."})
                self._save_messages(messages)
                continue
            actions = [a for a in response.actions if a.get("name") or a.get("id")]
            if not actions:
                json_retries += 1
                if json_retries > self.max_json_retries:
                    self.stop_reason = "empty_or_non_action"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "No tool action was parsed. Call a tool. Saying you finished does not submit.",
                    }
                )
                self._save_messages(messages)
                continue
            json_retries = 0
            names = [a.get("name") for a in actions]
            if "request_experiment" in names and len(actions) > 1:
                blocked = {
                    "status": "error",
                    "error_code": "SERIAL_PURCHASE_REQUIRED",
                    "message": "request_experiment cannot share a turn with other tools",
                    "retryable": False,
                }
                for action in actions:
                    messages.append(self._tool_result_message(action, blocked))
                self._save_messages(messages)
                continue
            for action in actions:
                args = action.get("arguments")
                if not isinstance(args, dict):
                    action["arguments"] = {"_parse_error": True, "raw": args}
                obs = None
                if action.get("arguments", {}).get("_parse_error"):
                    obs = self._malformed_obs("Tool arguments were not a JSON object. Resend one well-formed call.")
                elif not action.get("name"):
                    obs = {"status": "error", "error_code": "UNKNOWN_TOOL", "message": "tool name missing", "retryable": True}
                else:
                    obs = self.router.dispatch(action["name"], action.get("arguments") or {})
                messages.append(self._tool_result_message(action, obs))
                self._save_messages(messages)
                self._append_history({"step": step, "tool": action.get("name"), "observation": obs})
                if action.get("name") == "submit" and obs.get("status") == "ok" and (obs.get("observation") or {}).get("receipt"):
                    self.submitted = True
                    return {"submitted": True, "steps": step + 1}
            continue
        return {"submitted": self.submitted, "steps": self.max_steps, "stop_reason": self.stop_reason or "max_steps"}
