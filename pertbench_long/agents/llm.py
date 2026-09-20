"""LLM agent entry points. Live calls require a configured ModelClient; missing config is not a passing eval."""

from __future__ import annotations

from typing import Any

from pertbench_long.agents.client import ModelClient
from pertbench_long.agents.loop import AgentLoop
from pertbench_long.errors import AgentIncomplete, ConfigError
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
        loop = AgentLoop(
            client,
            router,
            public_spec=router.broker.public_spec,
            workspace=router.broker.workspace,
            allow_purchase=self.allow_purchase and router.allow_purchase,
            max_steps=int(self.config.get("max_agent_steps") or (self.config.get("runtime") or {}).get("max_agent_steps") or 60),
        )
        loop.run()
        if not loop.submitted:
            raise AgentIncomplete("model loop ended without a submit receipt")


def policy_allows_purchase(agent_name: str) -> bool:
    if agent_name in {"llm_no_query", "mean_delta_no_query", "no_change", "no_change_neutral_onehot", "no_change_dev_prior"}:
        return False
    return True
