"""T1: chemical dose acquisition. Roles are assigned from a complete condition table."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.data.audit_conditions import audit_conditions, match_controls, require_scale
from pertbench_long.episodes.builders.common import build_episode_from_condition_ids
from pertbench_long.errors import InsufficientEligible
from pertbench_long.schemas.conditions import PERTURBATION_CHEMICAL, PERTURBATION_CONTROL, canonical_dose_nm, canonical_time_s
from pertbench_long.schemas.types import PROTOCOL_CHEMICAL_DOSE


DOSE_OBJECTIVE = (
    "You have control data and some drug-dose expression evidence for one cell context. "
    "The candidate catalog lists additional experiments; each costs 1 credit. "
    "Select experiments, analyze returned evidence, and predict expression change for every target condition "
    "on the fixed gene panel. Targets cannot be purchased. Keep the evidence IDs that support each update."
)


def _dose_group_key(rec) -> tuple[str, str, str, str]:
    try:
        time_key = str(canonical_time_s(rec.time, rec.time_unit)) if rec.time is not None else "none"
    except Exception:
        time_key = f"{rec.time}|{rec.time_unit}"
    return (rec.resolved_context_id(), rec.perturbation_id, time_key, rec.assay or "unknown")


def _dose_groups(store: CanonicalStore, eligible: Sequence[str], controls: dict[str, str]) -> dict[tuple[str, str, str, str], list[str]]:
    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for cid in eligible:
        rec = store.records[store.condition_to_rows[cid][0]]
        if rec.perturbation_kind not in {PERTURBATION_CHEMICAL, "unknown"}:
            continue
        if rec.perturbation_kind == PERTURBATION_CONTROL or rec.perturbation_id.lower() in {"control", "vehicle", "dmso"}:
            continue
        if rec.dose is None:
            continue
        groups[_dose_group_key(rec)].append(cid)
    ordered = {}
    for key, ids in groups.items():
        ordered[key] = sorted(ids, key=lambda cid: canonical_dose_nm(store.records[store.condition_to_rows[cid][0]].dose, store.records[store.condition_to_rows[cid][0]].dose_unit))
    return ordered


def assign_dose_roles(
    store: CanonicalStore,
    *,
    split_variant: str,
    context_ids: Sequence[str] | None,
    min_candidates: int,
    min_targets: int,
    min_cells: int,
) -> dict[str, Any]:
    audit = audit_conditions(store, min_cells=min_cells, min_candidates=min_candidates, min_targets=min_targets, protocol=PROTOCOL_CHEMICAL_DOSE)
    controls = match_controls(store)
    eligible = [cid for cid in audit["eligible_condition_ids"] if not context_ids or store.records[store.condition_to_rows[cid][0]].resolved_context_id() in set(context_ids)]
    groups = _dose_groups(store, eligible, controls)
    usable = {k: v for k, v in groups.items() if len(v) >= 3}
    if len(usable) < min_targets:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: only {len(usable)} compounds have >=3 doses",
            details={"reason": "insufficient_eligible_conditions", "n_compounds": len(usable)},
        )
    o_ids, q_ids, t_ids = [], [], []
    for key, ids in sorted(usable.items()):
        doses = [canonical_dose_nm(store.records[store.condition_to_rows[cid][0]].dose, store.records[store.condition_to_rows[cid][0]].dose_unit) for cid in ids]
        if split_variant == "dose_interpolation":
            t_ids.append(ids[-2] if len(ids) >= 4 else ids[len(ids) // 2])
            remaining = [cid for cid in ids if cid not in t_ids]
            o_ids.append(remaining[0])
            q_ids.extend(remaining[1:])
        elif split_variant == "dose_extrapolation":
            t_ids.append(ids[-1])
            o_ids.append(ids[0])
            q_ids.extend(ids[1:-1])
        else:
            raise InsufficientEligible(f"unknown split_variant {split_variant}")
        if len(t_ids) >= min_targets and len(q_ids) >= min_candidates:
            # keep assigning remaining compounds into Q/O only after T is filled
            continue
    if len(q_ids) < min_candidates or len(t_ids) < min_targets:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: Q={len(q_ids)} T={len(t_ids)}",
            details={"reason": "insufficient_eligible_conditions", "Q": len(q_ids), "T": len(t_ids)},
        )
    t_set = set(t_ids[:])
    o_ids = [cid for cid in o_ids if cid not in t_set and cid not in q_ids]
    q_ids = [cid for cid in q_ids if cid not in t_set]
    c_ids = sorted({controls[cid] for cid in o_ids + q_ids + t_ids if controls.get(cid)})
    mapping = {cid: controls[cid] for cid in o_ids + q_ids + t_ids if controls.get(cid)}
    require_scale({"n_eligible": len(eligible), "reasons": audit["reasons"]}, min_candidates=min_candidates, min_targets=min_targets)
    return {
        "O": o_ids,
        "Q": q_ids,
        "T": t_ids,
        "C": c_ids,
        "control_mapping": mapping,
        "audit": audit,
    }


def build_chemical_dose_episode(
    store: CanonicalStore,
    *,
    dest: Path,
    episode_id: str,
    split_variant: str = "dose_interpolation",
    context_ids: Sequence[str] | None = None,
    experimental_budget: int = 8,
    min_candidates: int = 24,
    min_targets: int = 8,
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
    roles = assign_dose_roles(
        store,
        split_variant=split_variant,
        context_ids=context_ids,
        min_candidates=min_candidates,
        min_targets=min_targets,
        min_cells=min_cells,
    )
    return build_episode_from_condition_ids(
        store,
        dest=dest,
        episode_id=episode_id,
        protocol=PROTOCOL_CHEMICAL_DOSE,
        observed_condition_ids=roles["O"],
        queryable_condition_ids=roles["Q"],
        target_condition_ids=roles["T"],
        control_condition_ids=roles["C"],
        control_mapping=roles["control_mapping"],
        split_variant=split_variant,
        objective=DOSE_OBJECTIVE,
        synthetic=synthetic,
        study_name=study_name,
        data_release_id=data_release_id,
        experimental_budget=experimental_budget,
        min_candidates=min_candidates,
        min_targets=min_targets,
        n_panel=n_panel,
        a_family=a_family,
        notes=[f"split_variant={split_variant}", f"requested={min_candidates}/{min_targets}", f"actual_Q={len(roles['Q'])}", f"actual_T={len(roles['T'])}"],
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
