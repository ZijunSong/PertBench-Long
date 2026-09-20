"""Trusted AgentLoop: model → tool → observation until submit. Private labels never enter messages."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pertbench_long.agents.client import ModelClient, json_action_instruction, parse_json_action
from pertbench_long.errors import ConfigError, InvalidState, IsolationUnavailable, TransportError
from pertbench_long.hashes import sha256_json
from pertbench_long.runtime.tools import ToolRouter, TOOL_SCHEMA_VERSION


def public_system_prompt(public: dict[str, Any], *, allow_purchase: bool, workspace: str, gene_count: int) -> str:
    tools = "list_evidence, inspect_artifact, run_python, list_experiments, get_budget, save_prediction_snapshot, submit"
    if allow_purchase:
        tools += ", request_experiment"
    return (
        "You are a scientific agent in PertBench-Long. Predict held-out perturbation effects.\n"
        f"Objective: {public.get('objective')}\n"
        f"Episode: {public.get('episode_id')} protocol={public.get('protocol')} budget={public.get('experimental_budget')} credits.\n"
        f"Targets: {public.get('targets')}\n"
        f"Gene universe size G={gene_count}. Write a full target×gene parquet via run_python; do not paste matrices into chat.\n"
        f"Visible workspace: {workspace}. Tools: {tools}. Schema {TOOL_SCHEMA_VERSION}.\n"
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
    ) -> None:
        self.client = client
        self.router = router
        self.public_spec = public_spec
        self.workspace = Path(workspace)
        self.allow_purchase = allow_purchase
        self.max_steps = max_steps
        self.deadline = deadline_monotonic
        self.max_json_retries = max_json_retries
        self.history_path = Path(history_path) if history_path else self.workspace.parent / "messages.jsonl"
        self.submitted = False
        self.stop_reason = None

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError("episode_timeout")

    def _append_history(self, item: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, default=str) + "\n")

    def run(self) -> dict[str, Any]:
        public = self.public_spec.to_dict() if hasattr(self.public_spec, "to_dict") else dict(self.public_spec)
        genes_path = self.workspace / public.get("gene_universe_artifact", "genes_v1.tsv")
        gene_count = max(0, len(genes_path.read_text(encoding="utf-8").splitlines()) - 1) if genes_path.exists() else 0
        system = public_system_prompt(public, allow_purchase=self.allow_purchase, workspace="/workspace", gene_count=gene_count)
        if self.client.capabilities.action_format == "json_action":
            system += "\n" + json_action_instruction([t["function"]["name"] for t in self.router.schemas])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "initial_evidence": public.get("initial_evidence"),
                        "reference_evidence": public.get("reference_evidence"),
                        "candidate_experiments": public.get("candidate_experiments"),
                        "budget": public.get("experimental_budget"),
                        "output": "outputs/predictions.parquet plus outputs/claims.json then submit",
                    }
                ),
            },
        ]
        json_retries = 0
        tools = self.router.schemas if self.client.capabilities.action_format == "native_tools" else None
        for step in range(self.max_steps):
            self._check_deadline()
            try:
                response = self.client.generate(messages, tools)
            except TransportError:
                raise
            self._append_history({"step": step, "response": response.assistant_message, "finish_reason": response.finish_reason, "usage": response.usage})
            assistant = {k: v for k, v in response.assistant_message.items() if v is not None}
            messages.append(assistant if assistant.get("role") else {"role": "assistant", **assistant})
            if response.finish_reason == "length":
                messages.append({"role": "user", "content": "Your previous output was truncated. Continue with a valid tool call. Text is not a submit."})
                continue
            actions = [a for a in response.actions if a.get("name")]
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
                continue
            json_retries = 0
            if any(a.get("arguments", {}).get("_parse_error") for a in actions):
                messages.append({"role": "user", "content": "Tool arguments were not valid JSON. Resend one well-formed call."})
                continue
            if self.client.capabilities.action_format == "native_tools" and len(actions) > 1:
                results = self.router.dispatch_many([{"name": a["name"], "arguments": a.get("arguments") or {}} for a in actions])
                for action, obs in zip(actions, results):
                    messages.append({"role": "tool", "tool_call_id": action.get("id") or action["name"], "content": json.dumps(obs)})
                    if action["name"] == "submit" and obs.get("status") == "ok" and (obs.get("observation") or {}).get("receipt"):
                        self.submitted = True
                        return {"submitted": True, "steps": step + 1}
                continue
            action = actions[0]
            obs = self.router.dispatch(action["name"], action.get("arguments") or {})
            self._append_history({"step": step, "tool": action["name"], "observation": obs})
            if self.client.capabilities.action_format == "native_tools":
                messages.append({"role": "tool", "tool_call_id": action.get("id") or action["name"], "content": json.dumps(obs)})
            else:
                messages.append({"role": "user", "content": json.dumps({"tool_result": obs})})
            if action["name"] == "submit" and obs.get("status") == "ok" and (obs.get("observation") or {}).get("receipt"):
                self.submitted = True
                return {"submitted": True, "steps": step + 1}
        return {"submitted": self.submitted, "steps": self.max_steps, "stop_reason": self.stop_reason or "max_steps"}
