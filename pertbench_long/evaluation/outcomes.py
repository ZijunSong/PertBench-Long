"""Run outcomes and aggregation. Failures stay in the denominator."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

OUTCOMES = (
    "completed",
    "invalid_submission",
    "agent_timeout",
    "agent_tool_failure",
    "infra_error",
    "invalid_episode",
)

SCIENCE_FAILURES = frozenset({"invalid_submission", "agent_timeout", "agent_tool_failure"})
INFRA = frozenset({"infra_error"})


def official_direction_score(outcome: str, score: Optional[float]) -> float:
    if outcome == "completed":
        return float(score or 0.0)
    if outcome in SCIENCE_FAILURES:
        return 0.0
    if outcome == "invalid_episode":
        return float("nan")
    if outcome == "infra_error":
        return float("nan")
    raise ValueError(f"unknown outcome {outcome}")


def official_aubc(outcome: str, aubc_value: Optional[float]) -> float:
    if outcome == "completed":
        return float(aubc_value or 0.0)
    if outcome in SCIENCE_FAILURES:
        return 0.0
    return float("nan")


def summarize_runs(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(runs)
    by_outcome: dict[str, int] = {k: 0 for k in OUTCOMES}
    science_scores = []
    infra = []
    invalid_episodes = []
    skipped_nanmean = 0
    for run in items:
        outcome = str(run["outcome"])
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        if outcome in SCIENCE_FAILURES or outcome == "completed":
            science_scores.append(official_direction_score(outcome, run.get("direction_score")))
        elif outcome == "infra_error":
            infra.append({"run_id": run.get("run_id"), "reason": run.get("reason")})
        elif outcome == "invalid_episode":
            invalid_episodes.append({"run_id": run.get("run_id"), "reason": run.get("reason")})
    n_science = len(science_scores)
    mean_score = float(sum(science_scores) / n_science) if n_science else float("nan")
    return {
        "n_runs": len(items),
        "by_outcome": by_outcome,
        "science_denominator": n_science,
        "mean_direction_score_including_failures": mean_score,
        "infra_errors": infra,
        "invalid_episodes": invalid_episodes,
        "nanmean_not_used": True,
        "n_silently_dropped": skipped_nanmean,
        "note": "Seeds and prompt variants are repeated measures, not independent biological tasks.",
    }
