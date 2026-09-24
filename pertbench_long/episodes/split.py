"""Two-layer split and leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pertbench_long.errors import SplitLeakError
from pertbench_long.hashes import sha256_json
from pertbench_long.schemas.types import ExperimentRecord
from pertbench_long.schemas.validate import validate_disjoint_roles

PROTOCOL_NAMES = {
    "within_study_celltype_ood_v1": "within-study cell-type OOD; not donor-held-out; not study-held-out",
    "donor_heldout_v1": "donor-held-out; shared controls follow donor rules",
    "study_heldout_v1": "study-held-out",
    "chemical_dose_acquisition_v1": "chemical dose acquisition; interpolation or extrapolation",
    "genetic_pair_acquisition_v1": "genetic pair acquisition; pair-holdout with seen genes",
    "context_campaign_acquisition_v1": "cross-context budget allocation on shared compounds",
}


def condition_sets(records: Sequence[ExperimentRecord], ids: Iterable[str]) -> set[str]:
    wanted = set(ids)
    return {r.condition_key() for r in records if r.observation_id in wanted}


def audit_split(
    records: Sequence[ExperimentRecord],
    *,
    observed_obs: Sequence[str],
    queryable_obs: Sequence[str],
    target_obs: Sequence[str],
    control_obs: Sequence[str],
    observed_conditions: Sequence[str],
    queryable_conditions: Sequence[str],
    target_conditions: Sequence[str],
    development_target_conditions: Sequence[str] | None = None,
    protocol: str,
    public_data: bool = True,
    split_variant: str | None = None,
    control_mapping: Mapping[str, str] | None = None,
    control_conditions: Sequence[str] | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    validate_disjoint_roles(observed_obs, queryable_obs, target_obs, layer="observation")
    validate_disjoint_roles(observed_conditions, queryable_conditions, target_conditions, layer="condition")
    rec_by_id = {r.observation_id: r for r in records}
    if control_conditions is None:
        control_conditions = []
        for oid in control_obs:
            rec = rec_by_id.get(oid)
            if rec:
                control_conditions.append(rec.condition_id or rec.condition_key())
        control_conditions = sorted(set(control_conditions))
    leaks = []
    treated_qt = set(queryable_conditions) | set(target_conditions)
    leaked_c = set(control_conditions) & treated_qt
    for cid in sorted(leaked_c):
        leaks.append({"type": "control_contains_treated_condition", "condition": cid})
    mapping = dict(control_mapping or {})
    for treated, ctrl in mapping.items():
        if treated == ctrl:
            leaks.append({"type": "self_control", "condition": treated})
        if ctrl not in set(control_conditions):
            leaks.append({"type": "mapped_control_not_in_C", "condition": treated, "control": ctrl})
    for oid in list(observed_obs) + list(queryable_obs):
        rec = rec_by_id.get(oid)
        if rec and rec.condition_key() in set(target_conditions):
            leaks.append({"type": "condition_exposure", "observation_id": oid, "condition": rec.condition_key()})
    c_source = {
        rec_by_id[oid].source_identity()
        for oid in control_obs
        if oid in rec_by_id and rec_by_id[oid].source_identity()
    }
    for oid in list(queryable_obs) + list(target_obs):
        rec = rec_by_id.get(oid)
        if rec and rec.source_identity() and rec.source_identity() in c_source and rec.perturbation_kind != "control":
            leaks.append({"type": "source_identity_leak", "observation_id": oid, "source_identity": list(rec.source_identity())})
    development_overlap = []
    if development_target_conditions:
        overlap = set(development_target_conditions) & set(target_conditions)
        if overlap:
            development_overlap = sorted(overlap)
    status = "ok"
    flags = []
    if leaks:
        status = "reject"
        flags.append("canonical_or_condition_overlap_with_T")
        if leaked_c or any(item.get("type") == "self_control" for item in leaks):
            flags.append("control_role_contains_treated_measurement")
        if any(item.get("type") == "source_identity_leak" for item in leaks):
            flags.append("source_identity_leak")
    if development_overlap:
        if public_data:
            status = "pilot_not_private_independent_test"
            flags.append("evaluation_target_also_in_public_development")
        else:
            status = "reject"
            flags.append("official_eval_target_in_development")
    if protocol == "within_study_celltype_ood_v1":
        flags.append("single_public_study_fold_related")
    audit = {
        "status": status,
        "protocol": protocol,
        "protocol_is_not": [v for k, v in PROTOCOL_NAMES.items() if k != protocol],
        "n_observed_obs": len(observed_obs),
        "n_queryable_obs": len(queryable_obs),
        "n_target_obs": len(target_obs),
        "n_control_obs": len(control_obs),
        "leaks": leaks,
        "development_target_overlap": development_overlap,
        "flags": flags,
        "public_source_data": public_data,
        "fingerprint": sha256_json(
            {
                "O": sorted(observed_conditions),
                "Q": sorted(queryable_conditions),
                "T": sorted(target_conditions),
            }
        ),
    }
    if status == "reject":
        raise SplitLeakError("split leak detected", details=audit)
    return audit


def write_split_audit(audit: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(audit), indent=2), encoding="utf-8")


def _records_by_condition(records: Sequence[ExperimentRecord]) -> dict[str, ExperimentRecord]:
    out: dict[str, ExperimentRecord] = {}
    for rec in records:
        out[rec.condition_id or rec.condition_key()] = rec
    return out


def audit_chemical_dose_split(
    records: Sequence[ExperimentRecord],
    *,
    observed_conditions: Sequence[str],
    queryable_conditions: Sequence[str],
    target_conditions: Sequence[str],
    split_variant: str = "dose_interpolation",
    **kwargs: Any,
) -> dict[str, Any]:
    audit = audit_split(
        records,
        observed_conditions=observed_conditions,
        queryable_conditions=queryable_conditions,
        target_conditions=target_conditions,
        split_variant=split_variant,
        **{k: v for k, v in kwargs.items() if k not in {"split_variant"}},
    )
    by_cid = _records_by_condition(records)
    visible = set(observed_conditions) | set(queryable_conditions)
    from pertbench_long.schemas.conditions import canonical_dose_nm

    for tid in target_conditions:
        rec = by_cid[tid]
        if rec.dose is None:
            raise SplitLeakError("dose target missing dose", details={"condition": tid})
        target_nm = canonical_dose_nm(rec.dose, rec.dose_unit)
        same = [
            by_cid[cid]
            for cid in visible
            if by_cid[cid].perturbation_id == rec.perturbation_id and by_cid[cid].resolved_context_id() == rec.resolved_context_id() and by_cid[cid].dose is not None
        ]
        doses = [canonical_dose_nm(item.dose, item.dose_unit) for item in same]
        if split_variant == "dose_interpolation":
            if not doses or not any(d < target_nm for d in doses) or not any(d > target_nm for d in doses):
                raise SplitLeakError("dose_interpolation requires both lower and higher visible doses", details={"condition": tid})
        elif split_variant == "dose_extrapolation":
            if any(d >= target_nm for d in doses):
                raise SplitLeakError("dose_extrapolation cannot expose an equal or higher visible dose", details={"condition": tid})
        else:
            raise SplitLeakError(f"unknown dose split_variant {split_variant}")
    audit["split_variant"] = split_variant
    return audit


def audit_genetic_pair_split(
    records: Sequence[ExperimentRecord],
    *,
    observed_conditions: Sequence[str],
    queryable_conditions: Sequence[str],
    target_conditions: Sequence[str],
    split_variant: str = "pair_holdout_seen_genes",
    **kwargs: Any,
) -> dict[str, Any]:
    audit = audit_split(
        records,
        observed_conditions=observed_conditions,
        queryable_conditions=queryable_conditions,
        target_conditions=target_conditions,
        split_variant=split_variant,
        **{k: v for k, v in kwargs.items() if k not in {"split_variant"}},
    )
    by_cid = _records_by_condition(records)
    visible = set(observed_conditions) | set(queryable_conditions)
    visible_genes: set[str] = set()
    visible_pairs: set[tuple[str, ...]] = set()
    for cid in visible:
        rec = by_cid[cid]
        comps = rec.resolved_components()
        if rec.perturbation_kind == "genetic_pair" or len(comps) == 2:
            visible_pairs.add(tuple(sorted(comps)))
        visible_genes.update(comps)
    for tid in target_conditions:
        rec = by_cid[tid]
        comps = tuple(sorted(rec.resolved_components()))
        if rec.perturbation_kind == "genetic_pair" and comps in visible_pairs:
            raise SplitLeakError("target pair is queryable or observed", details={"condition": tid})
        if split_variant == "pair_holdout_seen_genes":
            missing = [g for g in comps if g not in visible_genes]
            if missing:
                raise SplitLeakError("pair_holdout_seen_genes requires support for every target gene", details={"missing": missing})
        elif split_variant == "gene_heldout":
            overlap = [g for g in comps if g in visible_genes]
            if overlap:
                raise SplitLeakError("gene_heldout target genes overlap visible genes", details={"overlap": overlap})
    audit["split_variant"] = split_variant
    return audit


def audit_context_campaign_split(
    records: Sequence[ExperimentRecord],
    *,
    observed_conditions: Sequence[str],
    queryable_conditions: Sequence[str],
    target_conditions: Sequence[str],
    split_variant: str = "few_shot_target_context",
    **kwargs: Any,
) -> dict[str, Any]:
    audit = audit_split(
        records,
        observed_conditions=observed_conditions,
        queryable_conditions=queryable_conditions,
        target_conditions=target_conditions,
        split_variant=split_variant,
        **{k: v for k, v in kwargs.items() if k not in {"split_variant"}},
    )
    by_cid = _records_by_condition(records)
    target_ctx = {by_cid[cid].resolved_context_id() for cid in target_conditions}
    q_ctx = {by_cid[cid].resolved_context_id() for cid in queryable_conditions}
    if split_variant == "few_shot_target_context":
        if not (target_ctx & q_ctx):
            raise SplitLeakError("few-shot context campaign requires Q access to target contexts", details={"target_ctx": sorted(target_ctx)})
        audit["claim"] = "target_context_few_shot_adaptation"
    elif split_variant == "zero_shot_target_context":
        if target_ctx & q_ctx:
            raise SplitLeakError("zero-shot context campaign cannot expose Q in target contexts", details={"overlap": sorted(target_ctx & q_ctx)})
        audit["claim"] = "zero_shot_context_extrapolation"
    audit["split_variant"] = split_variant
    return audit


EVAL_PARTITIONS = frozenset({"eval", "evaluation"})
DEV_PARTITIONS = frozenset({"dev", "development", "pilot", "train"})


def audit_release_cross_contamination(
    episodes: Sequence[Mapping[str, Any]],
    *,
    allow_shared_support: bool = False,
) -> dict[str, Any]:
    """Detect a development public condition becoming an evaluation target.

    Missing/unknown partition is a failure, not a pass. Shared support may
    reuse O/Q calibration conditions but never a hidden eval T.
    """
    by_partition: dict[str, dict[str, set[str]]] = {}
    for item in episodes:
        raw = item.get("partition")
        if raw in {None, "", "unknown"}:
            raise SplitLeakError("release partition is missing", details={"episode": item.get("episode_id")})
        part = str(raw)
        bucket = by_partition.setdefault(part, {"T": set(), "Q": set(), "O": set()})
        bucket["T"].update(item.get("target_condition_ids") or [])
        bucket["Q"].update(item.get("queryable_condition_ids") or [])
        bucket["O"].update(item.get("observed_condition_ids") or [])
    leaks = []
    public_dev: set[str] = set()
    for part, bucket in by_partition.items():
        if part in DEV_PARTITIONS or part not in EVAL_PARTITIONS:
            public_dev |= bucket["O"] | bucket["Q"] | bucket["T"]
    eval_t: set[str] = set()
    for part, bucket in by_partition.items():
        if part in EVAL_PARTITIONS:
            eval_t |= bucket["T"]
    hidden_as_public = eval_t & public_dev
    if hidden_as_public:
        leaks.append({"type": "eval_target_in_public_dev", "conditions": sorted(hidden_as_public)[:32]})
    dev = by_partition.get("dev") or by_partition.get("development")
    ev = by_partition.get("eval") or by_partition.get("evaluation")
    if dev and ev:
        leak_t = (dev["T"] & ev["T"]) | (dev["T"] & ev["Q"])
        if leak_t:
            leaks.append({"type": "dev_answer_in_eval", "conditions": sorted(leak_t)[:32]})
        if not allow_shared_support and (dev["T"] & ev["O"]):
            leaks.append({"type": "dev_target_in_eval_observed", "conditions": sorted(dev["T"] & ev["O"])[:32]})
    if leaks:
        raise SplitLeakError("release cross-contamination", details={"leaks": leaks})
    return {"status": "ok", "partitions": {k: {rk: len(rv) for rk, rv in v.items()} for k, v in by_partition.items()}}

