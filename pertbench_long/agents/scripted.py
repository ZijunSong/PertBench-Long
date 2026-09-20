"""Deterministic mock that uses the oracle/tool interface and never reads private labels."""

from __future__ import annotations

from typing import Any

from pertbench_long.baselines.engine import estimate, _write_pred
from pertbench_long.runtime.broker import Broker


class ScriptedMockAgent:
    name = "scripted_mock"

    def __init__(self, policy: str = "two_query") -> None:
        self.policy = policy

    def run(self, broker: Broker) -> None:
        effects, probs = estimate(broker)
        pred, claims = _write_pred(broker, effects, probs)
        broker.save_snapshot(pred, claims)
        catalog = [c["experiment_id"] for c in broker.list_experiments()]
        n_buy = 0 if self.policy == "zero_query" else 2 if self.policy == "two_query" else 1
        if self.policy == "early_stop":
            n_buy = 1
        for i, exp_id in enumerate(catalog[:n_buy], start=1):
            broker.request_experiment(exp_id, request_id=f"req_{i:04d}")
            effects, probs = estimate(broker)
            pred, claims = _write_pred(broker, effects, probs)
            broker.save_snapshot(pred, claims)
            if self.policy == "early_stop":
                break
        stop = "no_query" if n_buy == 0 else "early_stop" if self.policy == "early_stop" else "budget_exhausted"
        broker.submit(pred, claims, stop_reason=stop)
