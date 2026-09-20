"""LLM agent entry points. Live calls require a configured ModelClient; missing config is not a passing eval."""

from __future__ import annotations

from typing import Any

from pertbench_long.agents.client import ModelClient
from pertbench_long.agents.loop import AgentLoop
from pertbench_long.errors import AgentIncomplete
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.config import first_defined
from pertbench_long.runtime.tools import ToolRouter


class LLMAdapter:
    """Back-compat name used by list/docs. The loop is AgentLoop + ModelClient + ToolRouter."""

    name = "llm_adaptive_query"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.allow_purchase = bool(self.config.get("allow_purchase", True))
        if "llm_no_query" in str(self.config.get("policy") or self.name):
            self.allow_purchase = False

    def run(self, router: ToolRouter) -> None:
        client = ModelClient(self.config)
        runtime = dict(self.config.get("runtime") or {})
        max_steps = first_defined(self.config.get("max_agent_steps"), runtime.get("max_agent_steps"), default=60)
        deadline = first_defined(self.config.get("deadline_monotonic"), None)
        budget = self.config.get("runtime_budget")
        if budget is not None and not isinstance(budget, RuntimeBudget):
            budget = None
        loop = AgentLoop(
            client,
            router,
            public_spec=router.broker.public_spec,
            workspace=router.broker.workspace,
            allow_purchase=self.allow_purchase and router.allow_purchase,
            max_steps=int(max_steps),
            deadline_monotonic=deadline,
            runtime_budget=budget if isinstance(budget, RuntimeBudget) else router.broker.runtime_budget,
            seed_status=client.resolved_profile.get("seed_status"),
        )
        loop.run()
        if not loop.submitted:
            raise AgentIncomplete("model loop ended without a submit receipt")


def policy_allows_purchase(agent_name: str) -> bool:
    if agent_name in {"llm_no_query", "mean_delta_no_query", "no_change", "no_change_neutral_onehot", "no_change_dev_prior"}:
        return False
    return True
