"""Condition-table feasibility audit. Does not invent missing observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.schemas.conditions import (
    PERTURBATION_CHEMICAL,
    PERTURBATION_CONTROL,
    PERTURBATION_GENETIC_PAIR,
    PERTURBATION_GENETIC_SINGLE,
    canonical_dose_nm,
)

REASON_INSUFFICIENT = "insufficient_eligible_conditions"
REASON_NO_CONTROL = "control_unmatched"
REASON_UNKNOWN_UNIT = "unknown_unit_refused"
REASON_AMBIGUOUS = "ambiguous_condition_selector"


def _condition_meta(store: CanonicalStore) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cid, rows in store.condition_to_rows.items():
        rec = store.records[rows[0]]
        donors = {store.records[i].donor for i in rows if store.records[i].donor}
        batches = {store.records[i].batch_id for i in rows if store.records[i].batch_id}
        reps = {store.records[i].replicate_id for i in rows if store.records[i].replicate_id}
        out[cid] = {
            "condition_id": cid,
            "context_id": rec.resolved_context_id(),
            "cell_type": rec.cell_type,
            "perturbation_id": rec.perturbation_id,
            "perturbation_kind": rec.perturbation_kind,
            "perturbation_components": list(rec.resolved_components()),
            "dose": rec.dose,
            "dose_unit": rec.dose_unit,
            "time": rec.time,
            "time_unit": rec.time_unit,
            "n_cells": len(rows),
            "n_donors": len(donors),
            "n_batches": len(batches),
            "n_replicates": len(reps) or 1,
            "control_group_id": rec.control_group_id,
        }
    return out


def match_controls(store: CanonicalStore, *, control_kinds: Iterable[str] | None = None) -> dict[str, str | None]:
    kinds = set(control_kinds or {PERTURBATION_CONTROL})
    meta = _condition_meta(store)
    controls_by_ctx: dict[str, list[str]] = defaultdict(list)
    for cid, row in meta.items():
        if row["perturbation_kind"] in kinds or str(row["perturbation_id"]).lower() in {"control", "vehicle", "dmso", "ntc"}:
            controls_by_ctx[row["context_id"]].append(cid)
    mapping: dict[str, str | None] = {}
    for cid, row in meta.items():
        if row["perturbation_kind"] in kinds:
            mapping[cid] = cid
            continue
        candidates = controls_by_ctx.get(row["context_id"], [])
        mapping[cid] = candidates[0] if len(candidates) == 1 else (candidates[0] if candidates else None)
        if len(candidates) > 1:
            # Prefer same batch later; first version records reuse but requires an explicit mapping for official builds.
            mapping[cid] = candidates[0]
    return mapping


def audit_conditions(
    store: CanonicalStore,
    *,
    min_cells: int = 5,
    min_candidates: int = 24,
    min_targets: int = 8,
    protocol: str | None = None,
) -> dict[str, Any]:
    meta = _condition_meta(store)
    controls = match_controls(store)
    treated = [cid for cid, row in meta.items() if row["perturbation_kind"] not in {PERTURBATION_CONTROL} and str(row["perturbation_id"]).lower() not in {"control", "vehicle", "dmso", "ntc"}]
    eligible = [cid for cid in treated if meta[cid]["n_cells"] >= min_cells and controls.get(cid)]
    unmatched = [cid for cid in treated if not controls.get(cid)]
    small = [cid for cid in treated if meta[cid]["n_cells"] < min_cells]
    reasons: list[str] = []
    if unmatched:
        reasons.append(REASON_NO_CONTROL)
    if len(eligible) < min_candidates + min_targets:
        reasons.append(REASON_INSUFFICIENT)

    dose_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pair_graph = {"singles": [], "pairs": [], "support": {}}
    for cid in eligible:
        row = meta[cid]
        if row["perturbation_kind"] == PERTURBATION_CHEMICAL and row["dose"] is not None:
            try:
                nm = float(canonical_dose_nm(row["dose"], row["dose_unit"]))
            except Exception:
                reasons.append(REASON_UNKNOWN_UNIT)
                continue
            dose_groups[(row["context_id"], row["perturbation_id"])].append({"condition_id": cid, "dose_nm": nm})
        if row["perturbation_kind"] == PERTURBATION_GENETIC_SINGLE:
            pair_graph["singles"].append(cid)
        if row["perturbation_kind"] == PERTURBATION_GENETIC_PAIR:
            pair_graph["pairs"].append(cid)
            genes = tuple(row["perturbation_components"])
            pair_graph["support"][cid] = list(genes)

    compounds_with_doses = {key: sorted(vals, key=lambda x: x["dose_nm"]) for key, vals in dose_groups.items() if len(vals) >= 2}
    interpolation_ready = sum(1 for vals in compounds_with_doses.values() if len(vals) >= 3)
    status = "ok" if not reasons else "blocked"
    return {
        "status": status,
        "protocol": protocol,
        "n_conditions": len(meta),
        "n_treated": len(treated),
        "n_eligible": len(eligible),
        "n_unmatched_controls": len(unmatched),
        "n_below_min_cells": len(small),
        "min_cells": min_cells,
        "min_candidates": min_candidates,
        "min_targets": min_targets,
        "reasons": sorted(set(reasons)),
        "control_match_rate": (len(treated) - len(unmatched)) / len(treated) if treated else 0.0,
        "dose_compounds_with_gradient": len(compounds_with_doses),
        "interpolation_ready_compounds": interpolation_ready,
        "n_singles": len(pair_graph["singles"]),
        "n_pairs": len(pair_graph["pairs"]),
        "conditions": meta,
        "control_mapping": {k: v for k, v in controls.items() if v},
        "eligible_condition_ids": eligible,
    }


def require_scale(audit: dict[str, Any], *, min_candidates: int, min_targets: int) -> None:
    from pertbench_long.errors import InsufficientEligible

    n = int(audit.get("n_eligible") or 0)
    if n < min_candidates + min_targets or REASON_INSUFFICIENT in audit.get("reasons", []):
        raise InsufficientEligible(
            f"{REASON_INSUFFICIENT}: eligible={n} need_candidates={min_candidates} need_targets={min_targets}",
            details={"reason": REASON_INSUFFICIENT, "audit": {k: audit[k] for k in audit if k != "conditions"}},
        )
