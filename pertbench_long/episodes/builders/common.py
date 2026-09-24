"""Build an episode from explicit condition IDs. First-match cell_type selectors are forbidden."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.data.preprocess import public_export_fingerprint, select_panel_from_visible, to_effect_space
from pertbench_long.data.sparse_ops import extract_dense_block, finite_check_sample
from pertbench_long.episodes.export import export_public_bundle
from pertbench_long.episodes.readiness import resolve_scoring_eligibility
from pertbench_long.episodes.split import audit_release_cross_contamination, write_split_audit
from pertbench_long.errors import AmbiguousCondition, InsufficientEligible, SplitLeakError, UnsupportedProfile
from pertbench_long.evaluation.contract import CONTRACT_FILENAME, public_submission_contract
from pertbench_long.evaluation.labels import build_lognorm_cellmean_delta_labels
from pertbench_long.hashes import gene_order_hash, sha256_file, sha256_json
from pertbench_long.schemas.types import (
    COST_POLICY_UNIT_V1,
    LABEL_LOGNORM_CELLMEAN_DELTA_V1,
    SCHEMA_VERSION_V2,
    SCORING_CONTINUOUS_AUBC_V1,
    CandidateExperiment,
    PublicEpisodeSpec,
    TargetDescription,
)
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload
from pertbench_long.tasks.registry import get_task


def require_condition(store: CanonicalStore, condition_id: str, *, where: str) -> str:
    rows = store.condition_to_rows.get(condition_id)
    if not rows:
        raise AmbiguousCondition(f"{where}: missing condition {condition_id}")
    keys = {store.records[i].condition_id or store.records[i].condition_key() for i in rows}
    if len(keys) != 1:
        raise AmbiguousCondition(f"{where}: condition {condition_id} is internally ambiguous")
    return condition_id


def _obs_for(store: CanonicalStore, condition_id: str) -> list[str]:
    return [store.records[i].observation_id for i in store.condition_to_rows[condition_id]]


def _matrix_for(store: CanonicalStore, condition_id: str, gene_index: list[int], effect_matrix) -> np.ndarray:
    rows = store.condition_to_rows[condition_id]
    block = extract_dense_block(effect_matrix, rows, gene_index)
    return block


def _record_for(store: CanonicalStore, condition_id: str):
    return store.records[store.condition_to_rows[condition_id][0]]


def build_episode_from_condition_ids(
    store: CanonicalStore,
    *,
    dest: Path,
    episode_id: str,
    protocol: str,
    observed_condition_ids: Sequence[str],
    queryable_condition_ids: Sequence[str],
    target_condition_ids: Sequence[str],
    control_condition_ids: Sequence[str],
    control_mapping: Mapping[str, str],
    split_variant: str,
    objective: str,
    synthetic: bool,
    study_name: str,
    data_release_id: str,
    experimental_budget: int,
    min_candidates: int,
    min_targets: int,
    n_panel: int | None = 256,
    resource_profile: str = "long_cpu_v1",
    reference_policy: str = "matched_vehicle_v1",
    a_family: float = 0.5,
    split_audit: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    partition: str | None = None,
    requested_track: str = "pilot",
    label_estimand: str = "auto",
    provenance_verified: bool = False,
    source_verified: bool = False,
    scoring_scale_hash: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    development_episodes: Sequence[Mapping[str, Any]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    task = get_task(protocol)
    allowed_policies = {"matched_vehicle_v1", "matched_ntc_v1", "target_control_available_v1"}
    allowed_estimands = {"auto", "replicate_equal_weight", "cell_weighted_descriptive", "donor_equal_weight"}
    if reference_policy not in allowed_policies:
        raise UnsupportedProfile(f"unknown reference_policy {reference_policy!r}")
    if label_estimand not in allowed_estimands:
        raise UnsupportedProfile(f"unknown label_estimand {label_estimand!r}")
    if experimental_budget < 0:
        raise UnsupportedProfile("experimental_budget must be >= 0")
    if len(queryable_condition_ids) < min_candidates:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: Q={len(queryable_condition_ids)} < min_candidates={min_candidates}"
        )
    if len(target_condition_ids) < min_targets:
        raise InsufficientEligible(
            f"insufficient_eligible_conditions: T={len(target_condition_ids)} < min_targets={min_targets}"
        )
    if experimental_budget > len(queryable_condition_ids):
        raise UnsupportedProfile("experimental_budget cannot exceed the number of candidate experiments")

    o_ids = [require_condition(store, cid, where="O") for cid in observed_condition_ids]
    q_ids = [require_condition(store, cid, where="Q") for cid in queryable_condition_ids]
    t_ids = [require_condition(store, cid, where="T") for cid in target_condition_ids]
    c_ids = [require_condition(store, cid, where="C") for cid in control_condition_ids]
    if len(set(o_ids + q_ids + t_ids)) != len(o_ids) + len(q_ids) + len(t_ids):
        raise AmbiguousCondition("duplicate condition IDs across O/Q/T")
    for cid in list(o_ids) + list(q_ids) + list(t_ids):
        ctrl = control_mapping.get(cid)
        if not ctrl:
            raise UnsupportedProfile(f"control mapping missing for {cid}")
        if ctrl == cid:
            raise SplitLeakError("target or query condition mapped to itself as control", details={"condition": cid})
        require_condition(store, ctrl, where="control_mapping")
        if ctrl not in set(c_ids):
            raise SplitLeakError("control mapping does not point at a declared C condition", details={"condition": cid, "control": ctrl})
    treated_qt = set(q_ids) | set(t_ids)
    if set(c_ids) & treated_qt:
        raise SplitLeakError(
            "C contains Q/T treated conditions",
            details={"overlap": sorted(set(c_ids) & treated_qt)},
        )

    resolved_partition = partition or ("synthetic" if synthetic else None)
    if not synthetic and not resolved_partition:
        raise UnsupportedProfile("real episodes require an explicit partition")
    if development_episodes is not None:
        audit_release_cross_contamination(
            [
                *list(development_episodes),
                {
                    "episode_id": episode_id,
                    "partition": resolved_partition,
                    "target_condition_ids": t_ids,
                    "queryable_condition_ids": q_ids,
                    "observed_condition_ids": o_ids,
                },
            ]
        )

    eligibility = resolve_scoring_eligibility(
        synthetic=synthetic,
        matrix_kind=store.summary.matrix_kind,
        data_release_id=data_release_id,
        requested_track=requested_track,
        provenance_verified=provenance_verified,
        source_verified=source_verified,
        split_audited=bool((evidence or {}).get("split_audited")),
        labels_validated=store.summary.matrix_kind not in {"unknown", "scaled"},
        scoring_scale_hash=scoring_scale_hash,
        evidence=evidence,
    )
    scoring_track = eligibility["scoring_track"]
    if store.summary.matrix_kind in {"unknown", "scaled"}:
        if task.label_profile == LABEL_LOGNORM_CELLMEAN_DELTA_V1:
            raise UnsupportedProfile("unknown/scaled matrices cannot enter lognorm_cellmean_delta_v1")
        finite_check_sample(store.matrix)
        effect_matrix = store.matrix
    else:
        effect_matrix = to_effect_space(store.matrix, store.summary.matrix_kind, full_universe=store.matrix)

    o_obs, q_obs, t_obs, c_obs = [], [], [], []
    for cid in o_ids:
        o_obs.extend(_obs_for(store, cid))
    for cid in q_ids:
        q_obs.extend(_obs_for(store, cid))
    for cid in t_ids:
        t_obs.extend(_obs_for(store, cid))
    for cid in c_ids:
        c_obs.extend(_obs_for(store, cid))

    rec_index = {r.observation_id: i for i, r in enumerate(store.records)}
    visible_rows = [rec_index[oid] for oid in o_obs + c_obs]
    genes = list(store.gene_ids)
    panel_policy = "frozen_predefined"
    if n_panel is not None and n_panel < len(genes):
        genes = select_panel_from_visible(effect_matrix, store.gene_ids, visible_rows, n_genes=n_panel)
        panel_policy = "episode_specific_from_O_plus_C"
    gene_index = [store.gene_ids.index(g) for g in genes]

    computed_audit = task.split_validator(
        store.records,
        observed_obs=o_obs,
        queryable_obs=q_obs,
        target_obs=t_obs,
        control_obs=c_obs,
        observed_conditions=o_ids,
        queryable_conditions=q_ids,
        target_conditions=t_ids,
        control_conditions=c_ids,
        protocol=protocol,
        public_data=not synthetic,
        split_variant=split_variant,
        control_mapping=dict(control_mapping),
    )
    if split_audit is not None and split_audit.get("fingerprint") != computed_audit.get("fingerprint"):
        raise SplitLeakError("stale split_audit does not match the current O/Q/T assignment")
    audit = computed_audit

    dest = Path(dest)
    public_dir = dest / "public"
    private_dir = dest / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    def _bundle(condition_ids: Sequence[str], artifact_id: str, filename: str, *, public: bool) -> dict[str, Any]:
        rows, recs = [], []
        for cid in condition_ids:
            for i in store.condition_to_rows[cid]:
                rows.append(extract_dense_block(effect_matrix, [i], gene_index)[0])
                recs.append(asdict(store.records[i]))
        matrix = np.vstack(rows) if rows else np.zeros((0, len(genes)))
        root = public_dir if public else private_dir
        path = root / ("evidence" if public else "queryable") / filename
        export_public_bundle(
            matrix=matrix,
            records=recs,
            gene_ids=genes,
            dest=path,
            artifact_id=artifact_id,
            uns={"artifact_id": artifact_id, "synthetic": synthetic},
        )
        return {
            "artifact_id": artifact_id,
            "relative_path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "condition_ids": list(condition_ids),
        }

    initial = _bundle(o_ids, "ev_initial_001", "ev_initial_001.h5ad", public=True)
    controls_art = _bundle(c_ids, "ev_controls_001", "ev_controls_001.h5ad", public=True)

    q_catalog = []
    exp_to_cond = {}
    for i, cid in enumerate(q_ids, start=1):
        exp_id = f"q_{i:02d}"
        rec = _record_for(store, cid)
        art = _bundle([cid], f"ev_{exp_id}", f"ev_{exp_id}.h5ad", public=False)
        q_catalog.append(
            {
                "experiment_id": exp_id,
                "cell_type": rec.cell_type,
                "perturbation": rec.perturbation_id,
                "cost": 1,
                "dose": rec.dose,
                "dose_unit": rec.dose_unit,
                "time": rec.time,
                "time_unit": rec.time_unit,
                "context_id": rec.resolved_context_id(),
                "condition_id": cid,
                "perturbation_kind": rec.perturbation_kind,
                "perturbation_components": list(rec.resolved_components()),
                "artifact": art,
            }
        )
        exp_to_cond[exp_id] = cid

    gene_path = public_dir / "genes_v1.tsv"
    gene_path.write_text("gene_id\n" + "\n".join(genes) + "\n", encoding="utf-8")
    (public_dir / "gene_panel.tsv").write_text(gene_path.read_text(encoding="utf-8"), encoding="utf-8")
    contract = public_submission_contract(effect_unit="lognorm_cellmean_delta", require_direction=False)
    contract_path = public_dir / CONTRACT_FILENAME
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    targets_meta = []
    target_to_cond = {}
    stim_ctrl = {}
    donors: dict[str, tuple[list, list]] = {}
    replicates: dict[str, tuple[list, list]] = {}
    for i, cid in enumerate(t_ids, start=1):
        tid = f"t_{i:02d}"
        rec = _record_for(store, cid)
        ctrl_id = control_mapping[cid]
        stim_ctrl[tid] = (
            _matrix_for(store, cid, gene_index, effect_matrix),
            _matrix_for(store, ctrl_id, gene_index, effect_matrix),
        )
        donors[tid] = (
            [store.records[j].donor for j in store.condition_to_rows[cid]],
            [store.records[j].donor for j in store.condition_to_rows[ctrl_id]],
        )
        def _group(rec) -> str | None:
            if not rec.replicate_id:
                return None
            if rec.batch_id:
                return f"{rec.batch_id}|{rec.replicate_id}"
            return str(rec.replicate_id)

        replicates[tid] = (
            [_group(store.records[j]) for j in store.condition_to_rows[cid]],
            [_group(store.records[j]) for j in store.condition_to_rows[ctrl_id]],
        )
        targets_meta.append(
            TargetDescription(
                target_id=tid,
                cell_type=rec.cell_type,
                perturbation=rec.perturbation_id,
                dose=rec.dose,
                dose_unit=rec.dose_unit,
                time=rec.time,
                time_unit=rec.time_unit,
                context_id=rec.resolved_context_id(),
                context_type=rec.context_type,
                condition_id=cid,
                perturbation_kind=rec.perturbation_kind,
                perturbation_components=rec.resolved_components(),
                control_group_id=ctrl_id,
            )
        )
        target_to_cond[tid] = cid

    labels = build_lognorm_cellmean_delta_labels(
        gene_ids=genes,
        targets=stim_ctrl,
        donors=donors,
        replicates=replicates,
        estimand=label_estimand,
    )
    label_path = private_dir / "labels.parquet"
    labels.to_parquet(label_path)
    label_card = {
        "label_name": "lognorm_cellmean_delta",
        "profile": LABEL_LOGNORM_CELLMEAN_DELTA_V1,
        "effect_unit": "lognorm_cellmean_delta",
        "aggregation": labels.aggregation,
        "notes": labels.notes,
        "a_family": a_family,
    }
    (private_dir / "label_card.json").write_text(json.dumps(label_card, indent=2), encoding="utf-8")

    public_bytes_fp = public_export_fingerprint(genes, extract_dense_block(effect_matrix, visible_rows, gene_index))
    scoring_fp = sha256_json(
        {
            "profile": SCORING_CONTINUOUS_AUBC_V1,
            "label_profile": LABEL_LOGNORM_CELLMEAN_DELTA_V1,
            "a_family": a_family,
            "label_estimand": label_estimand,
            "aggregation": labels.aggregation,
            "control_mapping": dict(sorted(control_mapping.items())),
            "matrix_kind": store.summary.matrix_kind,
            "scale_source": eligibility["scale_source"],
            "scoring_scale_hash": scoring_scale_hash,
        }
    )
    candidates = [
        CandidateExperiment(
            experiment_id=item["experiment_id"],
            cell_type=item["cell_type"],
            perturbation=item["perturbation"],
            cost=1,
            dose=item.get("dose"),
            dose_unit=item.get("dose_unit") or "none",
            time=item.get("time"),
            time_unit=item.get("time_unit") or "none",
            context_id=item.get("context_id") or "",
            condition_id=item.get("condition_id") or "",
            perturbation_kind=item.get("perturbation_kind") or "unknown",
            perturbation_components=tuple(item.get("perturbation_components") or ()),
            control_group_id=control_mapping.get(item["condition_id"]),
        )
        for item in q_catalog
    ]
    public = PublicEpisodeSpec(
        schema_version=SCHEMA_VERSION_V2,
        episode_id=episode_id,
        protocol=protocol,
        protocol_version="1.0",
        label_profile=LABEL_LOGNORM_CELLMEAN_DELTA_V1,
        objective=objective,
        initial_evidence=["ev_initial_001"],
        reference_evidence=["ev_controls_001"],
        candidate_experiments=candidates,
        targets=targets_meta,
        target_control_available=True,
        gene_universe_artifact="genes_v1.tsv",
        experimental_budget=int(experimental_budget),
        cost_unit="credit",
        resource_profile=resource_profile,
        data_release_id=data_release_id,
        public_split_fingerprint=audit["fingerprint"],
        public_scoring_fingerprint=scoring_fp,
        reference_policy=reference_policy,
        gene_panel_policy=panel_policy,
        related_family=f"{study_name}:{task.task_family}:{split_variant}",
        synthetic=synthetic,
        notes=list(notes or [])
        + (["SYNTHETIC fixture; not for scientific conclusions"] if synthetic else [])
        + [eligibility["a_family_note"]],
        artifact_metadata={
            "initial": initial,
            "controls": controls_art,
            "public_bytes_fingerprint": public_bytes_fp,
            "gene_universe_sha256": sha256_file(gene_path),
            "submission_contract_sha256": sha256_file(contract_path),
            "declared_public_evidence": ["ev_initial_001.h5ad", "ev_controls_001.h5ad"],
        },
        canonical_units={"effect": "lognorm_cellmean_delta", "cost": "credit"},
        task_family=task.task_family,
        split_variant=split_variant,
        scoring_profile=SCORING_CONTINUOUS_AUBC_V1,
        cost_policy=COST_POLICY_UNIT_V1,
        readiness=eligibility["readiness"],
        data_provenance={
            "study": study_name,
            "synthetic": synthetic,
            "matrix_kind": store.summary.matrix_kind,
            "partition": resolved_partition,
            "seed": seed,
            **{k: eligibility[k] for k in ("source_verified", "labels_validated", "split_audited", "scale_source")},
        },
    )
    public_path = public_dir / "episode.json"
    public_path.write_text(json.dumps(public.to_dict(), indent=2), encoding="utf-8")
    validate_public_payload(json.loads(public_path.read_text()))

    conditions_rows = []
    for role, ids in (("O", o_ids), ("Q", q_ids), ("T", t_ids), ("C", c_ids)):
        for cid in ids:
            rec = _record_for(store, cid)
            conditions_rows.append(
                {
                    "condition_id": cid,
                    "role": role if role != "C" else "C",
                    "context_id": rec.resolved_context_id(),
                    "perturbation": rec.perturbation_id,
                    "perturbation_kind": rec.perturbation_kind,
                    "dose": rec.dose,
                    "dose_unit": rec.dose_unit,
                    "purchasable": role == "Q",
                }
            )
    import pandas as pd

    public_conditions = [row for row in conditions_rows if row["role"] != "T"]
    pd.DataFrame(public_conditions).to_parquet(public_dir / "conditions.parquet", index=False)
    pd.DataFrame([asdict(t) for t in targets_meta]).to_parquet(public_dir / "targets.parquet", index=False)
    (public_dir / "task.md").write_text(objective + "\n", encoding="utf-8")
    mapping_path = public_dir / "reference_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "policy": reference_policy,
                "mapping": dict(control_mapping),
                "notes": ["IDs only; Q/T measurements are not included"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    extra_public = {
        "task.md": sha256_file(public_dir / "task.md"),
        "conditions.parquet": sha256_file(public_dir / "conditions.parquet"),
        "targets.parquet": sha256_file(public_dir / "targets.parquet"),
        "gene_panel.tsv": sha256_file(public_dir / "gene_panel.tsv"),
        "reference_mapping.json": sha256_file(mapping_path),
    }
    public.artifact_metadata["public_files"] = extra_public
    public.artifact_metadata["reference_mapping"] = "reference_mapping.json"
    public_path.write_text(json.dumps(public.to_dict(), indent=2), encoding="utf-8")
    validate_public_payload(json.loads(public_path.read_text()))

    private_payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "episode_id": episode_id,
        "public": public.to_dict(),
        "observed_condition_ids": o_ids,
        "queryable_condition_ids": q_ids,
        "target_condition_ids": t_ids,
        "control_condition_ids": c_ids,
        "observation_ids_by_role": {"O": o_obs, "Q": q_obs, "T": t_obs, "C": c_obs},
        "experiment_id_to_condition": exp_to_cond,
        "target_id_to_condition": target_to_cond,
        "label_artifact": "labels.parquet",
        "source_inventory_id": store.summary.gene_order_hash,
        "split_config": {"split_variant": split_variant, "protocol": protocol, "partition": resolved_partition, "seed": seed},
        "scoring_config": {
            "metric": "continuous_effect",
            "scoring_profile": SCORING_CONTINUOUS_AUBC_V1,
            "label_profile": LABEL_LOGNORM_CELLMEAN_DELTA_V1,
            "scoring_track": scoring_track,
            "a_family": a_family,
            "label_estimand": label_estimand,
            "scale_source": eligibility["scale_source"],
            "scoring_scale_hash": scoring_scale_hash,
        },
        "private_data_hash": sha256_file(label_path),
        "development_group": "synthetic" if synthetic else study_name,
        "evaluation_group": episode_id,
        "gene_ids": genes,
        "gene_order_hash": gene_order_hash(genes),
        "matrix_kind": store.summary.matrix_kind,
        "donor_metadata_present": store.summary.donor_metadata_present,
        "provenance": {
            "import_summary": store.summary.to_dict(),
            "queryable_artifacts": q_catalog,
            "study": study_name,
        },
        "control_mapping": dict(control_mapping),
        "readiness": public.readiness,
        "label_validity": eligibility["label_validity"],
        "protocol_implementation": "implemented",
    }
    private_spec = validate_private_payload(private_payload)
    manifest_path = private_dir / "manifest.json"
    manifest_path.write_text(json.dumps(private_spec.to_dict(), indent=2), encoding="utf-8")
    write_split_audit(audit, private_dir / "split_audit.json")
    (private_dir / "condition_mapping.json").write_text(
        json.dumps({"control_mapping": dict(control_mapping), "target_id_to_condition": target_to_cond}, indent=2),
        encoding="utf-8",
    )
    (public_dir / "README.txt").write_text(
        "Public pack: O+C only. Queryable results are purchased via the oracle. T is never unlocked.\n",
        encoding="utf-8",
    )
    return {
        "public_dir": str(public_dir),
        "private_manifest": str(manifest_path),
        "episode_id": episode_id,
        "synthetic": synthetic,
        "n_genes": len(genes),
        "n_candidates": len(q_ids),
        "n_targets": len(t_ids),
        "split_audit": audit,
    }
