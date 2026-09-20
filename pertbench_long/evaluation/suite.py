"""Grouped model evaluation suite. Each run has its own workspace and ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pertbench_long.cli import _load_yaml
from pertbench_long.errors import ConfigError
from pertbench_long.evaluation.outcomes import json_safe, summarize_runs
from pertbench_long.runtime.runner import EpisodeRunner, score_run


def run_suite(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(Path(config_path))
    episodes = cfg.get("episodes")
    models = cfg.get("models")
    policies = cfg.get("policies") or ["llm_no_query", "llm_adaptive_query"]
    seeds = cfg.get("seeds") or [0]
    output = Path(cfg.get("output") or "runs/suite")
    if not episodes or not models:
        raise ConfigError("suite config requires episodes and models")
    rows = []
    failed = False
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            for model in models:
                for policy in policies:
                    for seed in seeds:
                        run_out = output / str(model.get("name") or model.get("model")) / str(policy) / f"seed{seed}"
                        runner = EpisodeRunner(
                            public_dir=Path(episode["public_dir"]),
                            private_manifest=Path(episode["private_manifest"]),
                            run_root=run_out,
                            mode=str(cfg.get("mode") or "local_trusted_debug"),
                            agent=str(policy),
                            agent_kwargs={"model": model, "seed": seed, "allow_purchase": policy != "llm_no_query"},
                            resource_limits=dict(cfg.get("runtime") or {}),
                            resolved_config={"mode": cfg.get("mode"), "model": model, "runtime": cfg.get("runtime") or {}, "evaluation": cfg.get("evaluation") or {}},
                        )
                        result = runner.run()
                        scores = None
                        if (Path(result["run_dir"]) / "termination.json").exists() and result.get("run_id"):
                            try:
                                scores = score_run(Path(result["run_dir"]), Path(episode["private_manifest"]))
                            except Exception as exc:
                                scores = {"outcome": "infra_error", "reason": type(exc).__name__}
                        row = {
                            "run_dir": result.get("run_dir"),
                            "outcome": (scores or result).get("outcome"),
                            "direction_score": None if not scores else scores.get("direction_score"),
                            "model": model.get("model"),
                            "policy": policy,
                            "seed": seed,
                            "isolation_qualified": result.get("isolation_qualified"),
                            "reason": result.get("reason"),
                        }
                        if row["outcome"] != "completed":
                            failed = True
                        handle.write(json.dumps(row, default=str) + "\n")
                        rows.append(row)
    summary = summarize_runs(rows, group_by=["model", "policy", "seed"])
    summary["failed"] = failed
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    return summary
