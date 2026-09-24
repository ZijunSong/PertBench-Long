"""Grouped model evaluation suite. Each run has its own workspace and ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pertbench_long.cli import _load_yaml
from pertbench_long.errors import ConfigError
from pertbench_long.evaluation.outcomes import json_safe, summarize_runs
from pertbench_long.runtime.config import agent_kwargs_from_resolved, resolve_run_config
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
                        resolved = resolve_run_config(
                            cli={"seed": seed, "agent": policy, "mode": cfg.get("mode"), "output": str(run_out)},
                            config_path=Path(config_path),
                            yaml_cfg={
                                "mode": cfg.get("mode") or "local_trusted_debug",
                                "public_dir": episode["public_dir"],
                                "private_manifest": episode["private_manifest"],
                                "model": model,
                                "runtime": cfg.get("runtime") or {},
                                "evaluation": cfg.get("evaluation") or {},
                                "agent": policy,
                                "output": str(run_out),
                            },
                            documented={"agent": policy, "output": str(run_out)},
                        )
                        runner = EpisodeRunner(
                            public_dir=Path(resolved["public_dir"]),
                            private_manifest=Path(resolved["private_manifest"]),
                            run_root=run_out,
                            mode=str(resolved["mode"]),
                            agent=str(policy),
                            agent_kwargs=agent_kwargs_from_resolved(resolved, extra={"allow_purchase": policy != "llm_no_query"}),
                            resource_limits=dict(resolved.get("runtime") or {}),
                            resolved_config=resolved,
                        )
                        result = runner.run()
                        scores = None
                        score_error = None
                        if (Path(result["run_dir"]) / "termination.json").exists() and result.get("run_id"):
                            try:
                                scores = score_run(Path(result["run_dir"]), Path(episode["private_manifest"]))
                            except Exception as exc:
                                scores = {"outcome": "infra_error", "reason": type(exc).__name__, "score_status": "not_scored"}
                                score_error = type(exc).__name__
                        manifest = {}
                        man_path = Path(result["run_dir"]) / "run_manifest.json" if result.get("run_dir") else None
                        if man_path and man_path.exists():
                            manifest = json.loads(man_path.read_text(encoding="utf-8"))
                        scored_outcome = None if not scores else scores.get("outcome")
                        exec_outcome = result.get("outcome")
                        row = {
                            "run_id": result.get("run_id"),
                            "run_dir": result.get("run_dir"),
                            "episode_id": manifest.get("episode_id") or episode.get("episode_id"),
                            "release": (manifest.get("binding") or {}).get("public_spec_hash"),
                            "outcome": scored_outcome or exec_outcome,
                            "execution_outcome": exec_outcome,
                            "scored_outcome": scored_outcome,
                            "direction_score": None if not scores else scores.get("direction_score"),
                            "AUBC": None if not scores else scores.get("AUBC"),
                            "score_status": None if not scores else scores.get("score_status"),
                            "model": model.get("model") or model.get("name"),
                            "policy": policy,
                            "seed": seed,
                            "budget": manifest.get("experimental_budget"),
                            "isolation_qualified": result.get("isolation_qualified") or manifest.get("isolation_qualified"),
                            "scoring_track": (manifest.get("binding") or {}).get("scoring_track") or episode.get("scoring_track"),
                            "synthetic": manifest.get("synthetic") if "synthetic" in manifest else episode.get("synthetic"),
                            "official_eligible": bool(manifest.get("official_eligible")),
                            "reason": (scores or {}).get("reason") or score_error or result.get("reason"),
                        }
                        if row["outcome"] != "completed":
                            failed = True
                        handle.write(json.dumps(row, default=str) + "\n")
                        rows.append(row)
    summary = summarize_runs(
        rows,
        group_by=["model", "policy", "seed", "scoring_track", "synthetic", "budget", "isolation_qualified", "official_eligible", "episode_id"],
    )
    summary["failed"] = failed
    summary["incompatible_tracks_not_pooled"] = True
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    return summary
