"""Run outcomes and aggregation. Failures stay in the denominator. Unscored is not a zero."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

OUTCOMES = (
    "completed",
    "invalid_submission",
    "agent_timeout",
    "agent_tool_failure",
    "infra_error",
    "invalid_episode",
    "integrity_failure",
    "not_scored",
)

SCIENCE_FAILURES = frozenset({"invalid_submission", "agent_timeout", "agent_tool_failure"})
INFRA = frozenset({"infra_error"})
JSON_SAFE_NAN = None


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def official_direction_score(outcome: str, score: Optional[float], *, score_status: str = "scored") -> Optional[float]:
    if score_status == "not_scored" or outcome == "not_scored":
        return None
    if outcome == "integrity_failure":
        return None
    if outcome == "completed":
        if score is None:
            return None
        return float(score)
    if outcome in SCIENCE_FAILURES:
        return 0.0
    if outcome == "invalid_episode":
        return None
    if outcome == "infra_error":
        return None
    raise ValueError(f"unknown outcome {outcome}")


def official_aubc(outcome: str, aubc_value: Optional[float], *, score_status: str = "scored") -> Optional[float]:
    if score_status == "not_scored" or outcome in {"not_scored", "integrity_failure", "infra_error", "invalid_episode"}:
        return None
    if outcome == "completed":
        if aubc_value is None:
            return None
        return float(aubc_value)
    if outcome in SCIENCE_FAILURES:
        return 0.0
    return None


def _group_key(run: Mapping[str, Any], fields: Sequence[str]) -> tuple:
    from typing import Sequence

    return tuple(str(run.get(f, "unknown")) for f in fields)


def summarize_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    group_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    items = list(runs)
    fields = list(group_by or ("agent", "model", "policy", "action_format", "budget", "isolation_qualified", "scoring_track", "synthetic", "official_eligible"))
    by_outcome: dict[str, int] = {k: 0 for k in OUTCOMES}
    science_scores: list[float] = []
    infra = []
    invalid_episodes = []
    integrity = []
    not_scored = []
    skipped_nanmean = 0
    groups: dict[str, dict[str, Any]] = {}
    incomplete_comparison = False

    for run in items:
        scored = None
        outcome = str(run.get("outcome") or "not_scored")
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        if outcome == "infra_error":
            infra.append({"run_id": run.get("run_id"), "reason": run.get("reason")})
        elif outcome == "invalid_episode":
            invalid_episodes.append({"run_id": run.get("run_id"), "reason": run.get("reason")})
        elif outcome == "integrity_failure":
            integrity.append({"run_id": run.get("run_id"), "reason": run.get("reason")})
            incomplete_comparison = True
        elif outcome == "not_scored":
            not_scored.append({"run_id": run.get("run_id"), "reason": run.get("reason") or "not_scored"})
            incomplete_comparison = True
        elif outcome == "completed" and run.get("direction_score") is None and run.get("score_status") == "not_scored":
            not_scored.append({"run_id": run.get("run_id"), "reason": "completed_without_score"})
            incomplete_comparison = True
        elif outcome in SCIENCE_FAILURES or outcome == "completed":
            scored = official_direction_score(outcome, run.get("direction_score"), score_status="scored")
            if scored is None:
                not_scored.append({"run_id": run.get("run_id"), "reason": "completed_without_score"})
                incomplete_comparison = True
            else:
                science_scores.append(float(scored))
        key = "|".join(str(run.get(f, "unknown")) for f in fields)
        bucket = groups.setdefault(
            key,
            {
                "key": {f: run.get(f) for f in fields},
                "n": 0,
                "science_scores": [],
                "by_outcome": {},
            },
        )
        bucket["n"] += 1
        bucket["by_outcome"][outcome] = bucket["by_outcome"].get(outcome, 0) + 1
        if scored is not None and (outcome in SCIENCE_FAILURES or outcome == "completed"):
            bucket["science_scores"].append(float(scored))

    group_rows = []
    for bucket in groups.values():
        scores = bucket["science_scores"]
        mean = float(sum(scores) / len(scores)) if scores else None
        group_rows.append(
            {
                "key": bucket["key"],
                "n": bucket["n"],
                "science_denominator": len(scores),
                "mean_direction_score_including_failures": mean,
                "by_outcome": bucket["by_outcome"],
            }
        )
    n_science = len(science_scores)
    mean_score = float(sum(science_scores) / n_science) if n_science else None
    return json_safe(
        {
            "n_runs": len(items),
            "by_outcome": by_outcome,
            "science_denominator": n_science,
            "mean_direction_score_including_failures": mean_score,
            "infra_errors": infra,
            "invalid_episodes": invalid_episodes,
            "integrity_failures": integrity,
            "not_scored": not_scored,
            "incomplete_comparison": incomplete_comparison,
            "nanmean_not_used": True,
            "n_silently_dropped": skipped_nanmean,
            "groups": group_rows,
            "group_by": fields,
            "note": "Seeds and prompt variants are repeated measures, not independent biological tasks. Unscored and integrity failures are not treated as model score 0.",
        }
    )
