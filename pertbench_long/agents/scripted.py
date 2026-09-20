"""Deterministic mock that uses the same ToolRouter as LLM agents."""

from __future__ import annotations

from typing import Any

from pertbench_long.baselines.engine import _write_pred, estimate
from pertbench_long.runtime.tools import ToolRouter


class ScriptedMockAgent:
    name = "scripted_mock"

    def __init__(self, policy: str = "two_query") -> None:
        self.policy = policy

    def run(self, router: ToolRouter | Any) -> None:
        if not isinstance(router, ToolRouter):
            from pertbench_long.runtime.executor import DebugPythonExecutor
            from pertbench_long.runtime.tools import ToolRouter as Router

            router = Router(router, DebugPythonExecutor(), allow_purchase=True)
        broker = router.broker
        effects, probs = estimate(broker)
        pred, claims = _write_pred(broker, effects, probs)
        snap = router.dispatch("save_prediction_snapshot", {"prediction_path": pred, "claims_path": claims})
        if snap.get("status") != "ok":
            raise RuntimeError(snap)
        catalog = [c["experiment_id"] for c in router.dispatch("list_experiments", {}).get("observation") or []]
        n_buy = 0 if self.policy == "zero_query" else 2 if self.policy == "two_query" else 1
        if self.policy == "early_stop":
            n_buy = 1
        if not router.allow_purchase:
            n_buy = 0
        for i, exp_id in enumerate(catalog[:n_buy], start=1):
            bought = router.dispatch("request_experiment", {"experiment_id": exp_id, "request_id": f"req_{i:04d}"})
            if bought.get("status") != "ok":
                break
            effects, probs = estimate(broker)
            pred, claims = _write_pred(broker, effects, probs)
            router.dispatch("save_prediction_snapshot", {"prediction_path": pred, "claims_path": claims})
            if self.policy == "early_stop":
                break
        stop = "no_query" if n_buy == 0 else "early_stop" if self.policy == "early_stop" else "budget_exhausted"
        result = router.dispatch("submit", {"prediction_path": pred, "claims_path": claims, "stop_reason": stop})
        if result.get("status") != "ok":
            raise RuntimeError(result)
