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
) -> dict[str, Any]:
    validate_disjoint_roles(observed_obs, queryable_obs, target_obs, layer="observation")
    validate_disjoint_roles(observed_conditions, queryable_conditions, target_conditions, layer="condition")
    rec_by_id = {r.observation_id: r for r in records}
    leaks = []
    for oid in list(observed_obs) + list(queryable_obs):
        rec = rec_by_id.get(oid)
        if rec and rec.condition_key() in set(target_conditions):
            leaks.append({"type": "condition_exposure", "observation_id": oid, "condition": rec.condition_key()})
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
