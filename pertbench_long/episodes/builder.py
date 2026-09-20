"""Build public/private episode packs from a canonical store."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.data.preprocess import profile_for_kind, public_export_fingerprint, select_panel_from_visible, to_effect_space
from pertbench_long.episodes.export import export_public_bundle
from pertbench_long.episodes.split import audit_split, write_split_audit
from pertbench_long.episodes.synthetic import O_TYPES, Q_TYPES, T_TYPES, build_synthetic_store
from pertbench_long.errors import UnsupportedProfile, UnsupportedProtocol
from pertbench_long.evaluation.labels import TAU_DEFAULT, build_effect_proxy_labels
from pertbench_long.hashes import gene_order_hash, sha256_file, sha256_json
from pertbench_long.schemas.types import (
    IMPLEMENTED_PROTOCOLS,
    LABEL_EFFECT_PROXY_V1,
    PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD,
    SCHEMA_VERSION,
    CandidateExperiment,
    PublicEpisodeSpec,
    TargetDescription,
)
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload

DEFAULT_ALIASES = {
    "CD4T": "CD4T",
    "CD8T": "CD8T",
    "CD14+Mono": "CD14+Mono",
    "Dendritic": "Dendritic",
    "FCGR3A+Mono": "FCGR3A+Mono",
    "B": "B",
    "NK": "NK",
}


def _condition_for(store: CanonicalStore, cell_type: str, perturbation: str) -> str:
    for rec in store.records:
        if rec.cell_type == cell_type and rec.perturbation_id == perturbation:
            return rec.condition_key()
    raise KeyError(f"missing condition {cell_type}/{perturbation}")


def _obs_for(store: CanonicalStore, condition_id: str) -> list[str]:
    return [store.records[i].observation_id for i in store.condition_to_rows.get(condition_id, [])]


def _matrix_for(store: CanonicalStore, condition_id: str, gene_index: list[int], effect_matrix: np.ndarray) -> np.ndarray:
    rows = store.condition_to_rows[condition_id]
    return effect_matrix[np.array(rows)][:, gene_index]


def build_episode_from_store(
    store: CanonicalStore,
    *,
    dest: Path,
    episode_id: str,
    o_types: Sequence[str],
    q_types: Sequence[str],
    t_types: Sequence[str],
    perturbation: str = "IFN",
    control_name: str = "Control",
    protocol: str = PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD,
    synthetic: bool = False,
    n_panel: int | None = None,
    study_name: str = "kang_pbmc_ifn_public",
    data_release_id: str = "local_unreleased",
    experimental_budget: int = 2,
    label_profile: str = LABEL_EFFECT_PROXY_V1,
) -> dict[str, Any]:
    if protocol not in IMPLEMENTED_PROTOCOLS:
        raise UnsupportedProtocol(f"protocol {protocol!r} is not implemented")
    if label_profile != LABEL_EFFECT_PROXY_V1:
        raise UnsupportedProfile(f"label_profile {label_profile!r} is not implemented")
    if experimental_budget < 0:
        raise UnsupportedProfile("experimental_budget must be >= 0")
    scoring_track = "official"
    if store.summary.matrix_kind in {"unknown", "scaled"}:
        scoring_track = "diagnostic"
    profile_for_kind(store.summary.matrix_kind) if scoring_track == "official" else None
    if scoring_track == "official":
        effect_matrix = to_effect_space(store.matrix, store.summary.matrix_kind, full_universe=store.matrix)
    else:
        # Diagnostic: do not guess a transform from numeric range. Keep values as imported.
        effect_matrix = np.asarray(store.matrix, dtype=np.float64)
        if not np.all(np.isfinite(effect_matrix)):
            raise UnsupportedProfile("diagnostic matrix contains NaN/Inf")
    o_stim = [_condition_for(store, ct, perturbation) for ct in o_types]
    q_stim = [_condition_for(store, ct, perturbation) for ct in q_types]
    t_stim = [_condition_for(store, ct, perturbation) for ct in t_types]
    controls = []
    for ct in list(o_types) + list(q_types) + list(t_types):
        controls.append(_condition_for(store, ct, control_name))
    controls = list(dict.fromkeys(controls))

    o_obs, q_obs, t_obs, c_obs = [], [], [], []
    for cid in o_stim:
        o_obs.extend(_obs_for(store, cid))
    for cid in q_stim:
        q_obs.extend(_obs_for(store, cid))
    for cid in t_stim:
        t_obs.extend(_obs_for(store, cid))
    for cid in controls:
        c_obs.extend(_obs_for(store, cid))

    rec_index = {r.observation_id: i for i, r in enumerate(store.records)}
    visible_rows = [rec_index[oid] for oid in o_obs + c_obs]
    genes = list(store.gene_ids)
    panel_policy = "frozen_predefined"
    if n_panel is not None and n_panel < len(genes):
        genes = select_panel_from_visible(effect_matrix, store.gene_ids, visible_rows, n_genes=n_panel)
        panel_policy = "episode_specific_from_O_plus_C"
    gene_index = [store.gene_ids.index(g) for g in genes]

    audit = audit_split(
        store.records,
        observed_obs=o_obs,
        queryable_obs=q_obs,
        target_obs=t_obs,
        control_obs=c_obs,
        observed_conditions=o_stim,
        queryable_conditions=q_stim,
        target_conditions=t_stim,
        protocol=protocol,
        public_data=not synthetic,
    )

    dest = Path(dest)
    public_dir = dest / "public"
    private_dir = dest / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    def _bundle(condition_ids: Sequence[str], artifact_id: str, filename: str) -> dict[str, Any]:
        rows, recs = [], []
        for cid in condition_ids:
            for i in store.condition_to_rows[cid]:
                rows.append(effect_matrix[i, gene_index])
                recs.append(asdict(store.records[i]))
        matrix = np.vstack(rows) if rows else np.zeros((0, len(genes)))
        path = public_dir / "evidence" / filename
        export_public_bundle(matrix=matrix, records=recs, gene_ids=genes, dest=path, artifact_id=artifact_id, uns={"artifact_id": artifact_id, "synthetic": synthetic})
        return {
            "artifact_id": artifact_id,
            "relative_path": str(Path("evidence") / filename),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "condition_ids": list(condition_ids),
        }

    initial = _bundle(o_stim, "ev_initial_001", "ev_initial_001.h5ad")
    controls_art = _bundle(controls, "ev_controls_001", "ev_controls_001.h5ad")

    q_catalog = []
    exp_to_cond = {}
    for i, (ct, cid) in enumerate(zip(q_types, q_stim), start=1):
        exp_id = f"q_{i:02d}"
        art = _bundle([cid], f"ev_{exp_id}", f"ev_{exp_id}.h5ad")
        # Keep Q artifacts in private materialization store, not the public initial mount.
        private_q = private_dir / "queryable" / f"ev_{exp_id}.h5ad"
        private_q.parent.mkdir(parents=True, exist_ok=True)
        src = public_dir / "evidence" / f"ev_{exp_id}.h5ad"
        private_q.write_bytes(src.read_bytes())
        src.unlink()
        q_catalog.append(
            {
                "experiment_id": exp_id,
                "cell_type": ct,
                "perturbation": perturbation,
                "cost": 1,
                "artifact": {**art, "relative_path": str(Path("queryable") / f"ev_{exp_id}.h5ad")},
            }
        )
        exp_to_cond[exp_id] = cid

    gene_path = public_dir / "genes_v1.tsv"
    gene_path.write_text("gene_id\n" + "\n".join(genes) + "\n", encoding="utf-8")
    from pertbench_long.evaluation.contract import CONTRACT_FILENAME, public_submission_contract

    contract = public_submission_contract(effect_unit="log1p_mean_diff")
    contract_path = public_dir / CONTRACT_FILENAME
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    targets_meta = []
    target_to_cond = {}
    stim_ctrl_matrices = {}
    donors: dict[str, tuple[list, list]] = {}
    for i, (ct, cid) in enumerate(zip(t_types, t_stim), start=1):
        tid = f"t_{i:02d}"
        ctrl_id = _condition_for(store, ct, control_name)
        stim_ctrl_matrices[tid] = (
            _matrix_for(store, cid, gene_index, effect_matrix),
            _matrix_for(store, ctrl_id, gene_index, effect_matrix),
        )
        donor_stim = [store.records[i].donor for i in store.condition_to_rows[cid]]
        donor_ctrl = [store.records[i].donor for i in store.condition_to_rows[ctrl_id]]
        donors[tid] = (donor_stim, donor_ctrl)
        targets_meta.append(TargetDescription(target_id=tid, cell_type=ct, perturbation=perturbation))
        target_to_cond[tid] = cid

    if experimental_budget > len(q_catalog):
        raise UnsupportedProfile("experimental_budget cannot exceed the number of candidate experiments")
    labels = build_effect_proxy_labels(gene_ids=genes, targets=stim_ctrl_matrices, tau=TAU_DEFAULT, donors=donors)
    label_path = private_dir / "labels.parquet"
    labels.to_parquet(label_path)
    label_card = {
        "label_name": "effect_direction_proxy",
        "profile": LABEL_EFFECT_PROXY_V1,
        "tau": TAU_DEFAULT,
        "effect_unit": "log1p_mean_diff",
        "aggregation": labels.aggregation,
        "distribution": labels.frame["effect_direction_proxy"].value_counts().to_dict(),
        "notes": labels.notes,
    }
    (private_dir / "label_card.json").write_text(json.dumps(label_card, indent=2), encoding="utf-8")

    public_bytes_fp = public_export_fingerprint(genes, effect_matrix[visible_rows][:, gene_index])
    scoring_fp = sha256_json({"profile": LABEL_EFFECT_PROXY_V1, "tau": TAU_DEFAULT, "metric": "direction_score"})
    candidates = [
        CandidateExperiment(experiment_id=item["experiment_id"], cell_type=item["cell_type"], perturbation=perturbation, cost=1)
        for item in q_catalog
    ]
    public = PublicEpisodeSpec(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        protocol=protocol,
        protocol_version="1.0",
        label_profile=LABEL_EFFECT_PROXY_V1,
        objective="Predict held-out IFN responses using observed data and at most two additional condition bundles.",
        initial_evidence=["ev_initial_001"],
        reference_evidence=["ev_controls_001"],
        candidate_experiments=candidates,
        targets=targets_meta,
        target_control_available=True,
        gene_universe_artifact="genes_v1.tsv",
        experimental_budget=int(experimental_budget),
        cost_unit="credit",
        resource_profile="cpu_pilot_v1",
        data_release_id=data_release_id,
        public_split_fingerprint=audit["fingerprint"],
        public_scoring_fingerprint=scoring_fp,
        gene_panel_policy=panel_policy,
        related_family=f"{study_name}:celltype_ood_ifn",
        synthetic=synthetic,
        notes=(["SYNTHETIC fixture; not for scientific conclusions"] if synthetic else ["public Kang/PBMC-derived pilot; not a private independent test"])
        + (["scoring_track=diagnostic; processing history unknown"] if scoring_track == "diagnostic" else []),
        artifact_metadata={
            "initial": initial,
            "controls": controls_art,
            "public_bytes_fingerprint": public_bytes_fp,
            "gene_universe_sha256": sha256_file(gene_path),
            "submission_contract_sha256": sha256_file(contract_path),
            "declared_public_evidence": ["ev_initial_001.h5ad", "ev_controls_001.h5ad"],
        },
        canonical_units={"effect": "log1p_mean_diff", "cost": "credit"},
    )
    public_path = public_dir / "episode.json"
    public_path.write_text(json.dumps(public.to_dict(), indent=2), encoding="utf-8")
    validate_public_payload(json.loads(public_path.read_text()))

    private_payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "public": public.to_dict(),
        "observed_condition_ids": o_stim,
        "queryable_condition_ids": q_stim,
        "target_condition_ids": t_stim,
        "control_condition_ids": controls,
        "observation_ids_by_role": {"O": o_obs, "Q": q_obs, "T": t_obs, "C": c_obs},
        "experiment_id_to_condition": exp_to_cond,
        "target_id_to_condition": target_to_cond,
        "label_artifact": "labels.parquet",
        "source_inventory_id": store.summary.gene_order_hash,
        "split_config": {"o_types": list(o_types), "q_types": list(q_types), "t_types": list(t_types)},
        "scoring_config": {
            "metric": "direction_score",
            "tau": TAU_DEFAULT,
            "label_profile": LABEL_EFFECT_PROXY_V1,
            "scoring_track": scoring_track,
        },
        "private_data_hash": sha256_file(label_path),
        "development_group": "public_pbmc_study" if not synthetic else "synthetic",
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
    }
    private_spec = validate_private_payload(private_payload)
    manifest_path = private_dir / "manifest.json"
    manifest_path.write_text(json.dumps(private_spec.to_dict(), indent=2), encoding="utf-8")
    write_split_audit(audit, private_dir / "split_audit.json")
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
        "split_audit": audit,
    }


def build_synthetic_episode(dest: Path, *, episode_id: str = "synthetic_pilot_001", hidden_shift: float = 0.0, experimental_budget: int = 2) -> dict[str, Any]:
    store = build_synthetic_store(hidden_shift=hidden_shift)
    return build_episode_from_store(
        store,
        dest=dest,
        episode_id=episode_id,
        o_types=O_TYPES,
        q_types=Q_TYPES,
        t_types=T_TYPES,
        synthetic=True,
        study_name="synthetic_pbmc_v1",
        data_release_id="synthetic_v1",
        experimental_budget=experimental_budget,
    )
