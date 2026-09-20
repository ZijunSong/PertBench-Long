"""Non-LLM baselines."""

from pertbench_long.baselines.engine import BASELINE_REGISTRY, run_baseline

__all__ = ["BASELINE_REGISTRY", "run_baseline"]
