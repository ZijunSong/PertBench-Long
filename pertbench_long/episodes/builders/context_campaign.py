"""T3: shared-budget allocation across cell contexts on sci-Plex3."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.data.audit_conditions import audit_conditions, match_controls
from pertbench_long.episodes.builders.common import build_episode_from_condition_ids
from pertbench_long.errors import InsufficientEligible
from pertbench_long.schemas.conditions import PERTURBATION_CONTROL
from pertbench_long.schemas.types import PROTOCOL_CONTEXT_CAMPAIGN

CONTEXT_OBJECTIVE = (
    "You are responsible for response prediction in multiple target cell contexts with a shared credit budget. "
    "A source context has richer evidence; target contexts have limited calibration data. "
    "Choose additional experiments and predict every target. Scores average equally across contexts, "
    "so solving only one context cannot cover failure on another."
)


def assign_context_roles(
    store: CanonicalStore,
    *,
    source_context: str,
    target_contexts: Sequence[str],
    min_candidates: int,
    min_targets: int,
    min_cells: int,
    seed: int = 1701,
) -> dict[str, Any]:
    audit = audit_conditions(store, min_cells=min_cells, min_candidates=min_candidates, min_targets=min_targets, protocol=PROTOCOL_CONTEXT_CAMPAIGN)
    controls = match_controls(store)
    by_ctx: dict[str, list[str]] = defaultdict(list)
    for cid in audit["eligible_condition_ids"]:
        rec = store.records[store.condition_to_rows[cid][0]]
        if rec.perturbation_kind == PERTURBATION_CONTROL or rec.perturbation_id.lower() in {"control", "vehicle", "dmso"}:
            continue
        by_ctx[rec.resolved_context_id()].append(cid)
    if source_context not in by_ctx:
        raise InsufficientEligible(f"insufficient_eligible_conditions: missing source context {source_context}")
    source_items = sorted(by_ctx[source_context])
    o_ids = list(source_items[: max(4, len(source_items) // 2)])
    q_ids = [cid for cid in source_items if cid not in o_ids]
    t_ids: list[str] = []
    for ctx in target_contexts:
        items = sorted(by_ctx.get(ctx) or [])
        if len(items) < 4:
            raise InsufficientEligible(f"insufficient_eligible_conditions: context {ctx} has {len(items)} eligible conditions")
        o_ids.append(items[0])
        t_ids.extend(items[1:7])
        q_ids.extend(items[7:])
    if len(q_ids) < min_candidates or len(t_ids) < min_targets:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: Q={len(q_ids)} T={len(t_ids)}",
            details={"reason": "insufficient_eligible_conditions"},
        )
    used = o_ids + q_ids + t_ids
    c_ids = sorted({controls[cid] for cid in used if controls.get(cid)})
    mapping = {cid: controls[cid] for cid in used if controls.get(cid)}
    return {"O": o_ids, "Q": q_ids, "T": t_ids, "C": c_ids, "control_mapping": mapping, "audit": audit}


def build_context_campaign_episode(
    store: CanonicalStore,
    *,
    dest: Path,
    episode_id: str,
    source_context: str = "A549",
    target_contexts: Sequence[str] | None = None,
    split_variant: str = "few_shot_target_context",
    experimental_budget: int = 16,
    min_candidates: int = 48,
    min_targets: int = 12,
    n_panel: int | None = 256,
    synthetic: bool = False,
    study_name: str = "sciplex3",
    data_release_id: str = "local_unreleased",
    min_cells: int = 5,
    a_family: float = 0.5,
    seed: int = 1701,
    partition: str | None = None,
    resource_profile: str = "long_cpu_v1",
    requested_track: str = "pilot",
    label_estimand: str = "auto",
    provenance_verified: bool = False,
    source_verified: bool = False,
    scoring_scale_hash: str | None = None,
    evidence: dict | None = None,
    reference_policy: str = "matched_vehicle_v1",
    development_episodes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    targets = list(target_contexts or ["K562", "MCF7"])
    roles = assign_context_roles(
        store,
        source_context=source_context,
        target_contexts=targets,
        min_candidates=min_candidates,
        min_targets=min_targets,
        min_cells=min_cells,
        seed=seed,
    )
    return build_episode_from_condition_ids(
        store,
        dest=dest,
        episode_id=episode_id,
        protocol=PROTOCOL_CONTEXT_CAMPAIGN,
        observed_condition_ids=roles["O"],
        queryable_condition_ids=roles["Q"],
        target_condition_ids=roles["T"],
        control_condition_ids=roles["C"],
        control_mapping=roles["control_mapping"],
        split_variant=split_variant,
        objective=CONTEXT_OBJECTIVE,
        synthetic=synthetic,
        study_name=study_name,
        data_release_id=data_release_id,
        experimental_budget=experimental_budget,
        min_candidates=min_candidates,
        min_targets=min_targets,
        n_panel=n_panel,
        a_family=a_family,
        notes=["few-shot target-context adaptation; not zero-shot", f"actual_Q={len(roles['Q'])}", f"actual_T={len(roles['T'])}"],
        partition=partition or ("synthetic" if synthetic else None),
        requested_track=requested_track,
        label_estimand=label_estimand,
        provenance_verified=provenance_verified,
        source_verified=source_verified,
        scoring_scale_hash=scoring_scale_hash,
        evidence=evidence,
        reference_policy=reference_policy,
        development_episodes=development_episodes,
        seed=seed,
        resource_profile=resource_profile,
    )
