"""T2: single-gene to pair inference. Hidden pair outcomes are never synthesized."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.data.audit_conditions import audit_conditions, match_controls
from pertbench_long.episodes.builders.common import build_episode_from_condition_ids
from pertbench_long.errors import InsufficientEligible
from pertbench_long.schemas.conditions import PERTURBATION_CONTROL, PERTURBATION_GENETIC_PAIR, PERTURBATION_GENETIC_SINGLE
from pertbench_long.schemas.types import PROTOCOL_GENETIC_PAIR

PAIR_OBJECTIVE = (
    "Predict transcriptional responses of held-out two-gene activation combinations. "
    "Initial evidence includes some single-gene and pair experiments. You may purchase at most "
    "the listed budget of additional catalog experiments. Allocate budget among single-gene baselines, "
    "related pairs, and possible non-additive responses. Cite unlocked experiment IDs; do not claim a unique mechanism."
)


def _kind(store: CanonicalStore, cid: str) -> str:
    return store.records[store.condition_to_rows[cid][0]].perturbation_kind


def _comps(store: CanonicalStore, cid: str) -> tuple[str, ...]:
    return store.records[store.condition_to_rows[cid][0]].resolved_components()


def assign_pair_roles(
    store: CanonicalStore,
    *,
    split_variant: str,
    min_candidates: int,
    min_targets: int,
    min_cells: int,
    seed: int = 1701,
) -> dict[str, Any]:
    audit = audit_conditions(store, min_cells=min_cells, min_candidates=min_candidates, min_targets=min_targets, protocol=PROTOCOL_GENETIC_PAIR)
    controls = match_controls(store)
    eligible = list(audit["eligible_condition_ids"])
    singles = [cid for cid in eligible if _kind(store, cid) == PERTURBATION_GENETIC_SINGLE]
    pairs = [cid for cid in eligible if _kind(store, cid) == PERTURBATION_GENETIC_PAIR]
    if split_variant != "pair_holdout_seen_genes":
        raise InsufficientEligible(f"split_variant {split_variant} is not implemented in this builder")
    gene_to_single = {}
    for cid in singles:
        gene_to_single[_comps(store, cid)[0]] = cid
    usable_pairs = []
    for cid in pairs:
        genes = _comps(store, cid)
        if all(g in gene_to_single for g in genes):
            usable_pairs.append(cid)
    if len(usable_pairs) < min_targets:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: only {len(usable_pairs)} pairs have single-gene support",
            details={"reason": "insufficient_eligible_conditions"},
        )
    usable_pairs = sorted(usable_pairs)
    rng = random.Random(int(seed))
    shuffled = usable_pairs[:]
    rng.shuffle(shuffled)
    t_ids = sorted(shuffled[:min_targets])
    t_set = set(t_ids)
    support_singles = []
    for cid in t_ids:
        for gene in _comps(store, cid):
            support_singles.append(gene_to_single[gene])
    support_singles = list(dict.fromkeys(support_singles))
    other_pairs = [cid for cid in pairs if cid not in t_set]
    other_singles = [cid for cid in singles if cid not in support_singles]
    o_ids = support_singles[: max(2, len(support_singles) // 3)] + other_pairs[:2]
    remaining = [cid for cid in support_singles + other_singles + other_pairs if cid not in o_ids and cid not in t_set]
    q_ids = remaining
    if len(q_ids) < min_candidates:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: Q={len(q_ids)} < {min_candidates}",
            details={"reason": "insufficient_eligible_conditions", "Q": len(q_ids)},
        )
    all_used = o_ids + q_ids + t_ids
    c_ids = sorted({controls[cid] for cid in all_used if controls.get(cid)})
    mapping = {cid: controls[cid] for cid in all_used if controls.get(cid)}
    return {"O": o_ids, "Q": q_ids, "T": t_ids, "C": c_ids, "control_mapping": mapping, "audit": audit}


def build_genetic_pair_episode(
    store: CanonicalStore,
    *,
    dest: Path,
    episode_id: str,
    split_variant: str = "pair_holdout_seen_genes",
    experimental_budget: int = 8,
    min_candidates: int = 24,
    min_targets: int = 8,
    n_panel: int | None = 256,
    synthetic: bool = False,
    study_name: str = "norman2019",
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
    development_episodes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    roles = assign_pair_roles(
        store,
        split_variant=split_variant,
        min_candidates=min_candidates,
        min_targets=min_targets,
        min_cells=min_cells,
        seed=seed,
    )
    return build_episode_from_condition_ids(
        store,
        dest=dest,
        episode_id=episode_id,
        protocol=PROTOCOL_GENETIC_PAIR,
        observed_condition_ids=roles["O"],
        queryable_condition_ids=roles["Q"],
        target_condition_ids=roles["T"],
        control_condition_ids=roles["C"],
        control_mapping=roles["control_mapping"],
        split_variant=split_variant,
        objective=PAIR_OBJECTIVE,
        synthetic=synthetic,
        study_name=study_name,
        data_release_id=data_release_id,
        experimental_budget=experimental_budget,
        min_candidates=min_candidates,
        min_targets=min_targets,
        n_panel=n_panel,
        a_family=a_family,
        notes=["intervention=CRISPRa", f"split_variant={split_variant}", f"actual_Q={len(roles['Q'])}", f"actual_T={len(roles['T'])}"],
        partition=partition or ("synthetic" if synthetic else None),
        requested_track=requested_track,
        label_estimand=label_estimand,
        provenance_verified=provenance_verified,
        source_verified=source_verified,
        scoring_scale_hash=scoring_scale_hash,
        development_episodes=development_episodes,
        seed=seed,
        resource_profile=resource_profile,
    )
