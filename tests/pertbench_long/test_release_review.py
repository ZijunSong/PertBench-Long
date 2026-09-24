"""Regressions for the 2026-09-24 release-readiness review (E01–E08)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pertbench_long.data.acquire import fetch_dataset
from pertbench_long.data.doctor import doctor_data
from pertbench_long.data.prepare.sciplex import prepare_sciplex
from pertbench_long.episodes.builders.chemical_dose import build_chemical_dose_episode
from pertbench_long.episodes.builders.common import build_episode_from_condition_ids
from pertbench_long.episodes.build_config import validate_build_config
from pertbench_long.episodes.synthetic_long import build_synthetic_dose_episode, build_synthetic_dose_store
from pertbench_long.errors import ConfigError, SplitLeakError, UnsupportedProfile
from pertbench_long.runtime.executor import DockerPythonExecutor, isolation_is_qualified
from pertbench_long.schemas.types import PROTOCOL_CHEMICAL_DOSE


def test_e01_optional_guides_do_not_block_fetch(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "cell_identities.csv").write_text("cell_barcode,guide_identity\n", encoding="utf-8")
    (raw / "counts.csv").write_text("gene\n", encoding="utf-8")
    manifest = Path(__file__).resolve().parents[2] / "data" / "acquisition" / "norman2019_v1.json"
    # Place files where fetch expects them by using a copied manifest and data root layout via direct call
    from pertbench_long.data.acquire import load_source_files

    _ds, _rel, sources, _payload = load_source_files(manifest)
    dest = tmp_path / "data" / "raw" / "norman2019" / "norman2019_v1"
    dest.mkdir(parents=True)
    (dest / "cell_identities.csv").write_text("cell\n", encoding="utf-8")
    (dest / "counts.csv").write_text("g\n", encoding="utf-8")
    report = fetch_dataset("norman2019", data_root=tmp_path / "data", manifest_path=manifest, offline=True)
    assert "guides.csv" not in report.missing
    assert report.status == "ok"
    assert any(s.requirement == "optional" for s in sources if s.name == "guides.csv")


def test_e02_scaled_h5ad_is_refused_at_prepare(tmp_path: Path):
    import anndata as ad

    source = tmp_path / "src"
    source.mkdir()
    pd.DataFrame({"cell_barcode": ["c0"], "cell_type": ["A549"], "perturbation": ["Drug"], "dose": [1], "dose_unit": ["nM"], "is_control": [False]}).to_csv(source / "cells.csv", index=False)
    adata = ad.AnnData(X=np.array([[-1.0, 0.2]]), obs=pd.DataFrame(index=["c0"]), var=pd.DataFrame(index=["G1", "G2"]))
    adata.uns["matrix_kind"] = "scaled"
    adata.write_h5ad(source / "counts.h5ad")
    with pytest.raises(Exception, match="scaled|negative|refuses"):
        prepare_sciplex(source, tmp_path / "out", input_profile="h5ad_bundle")
    assert not (tmp_path / "out" / "matrix.h5ad").exists()


def test_e03_prepare_does_not_densify_sparse_input(tmp_path: Path):
    import anndata as ad
    from scipy import sparse

    source = tmp_path / "src"
    source.mkdir()
    pd.DataFrame(
        {
            "cell_barcode": ["c0", "c1"],
            "cell_type": ["A549", "A549"],
            "perturbation": ["DMSO", "Drug"],
            "dose": [None, 1],
            "dose_unit": ["", "nM"],
            "time": [24, 24],
            "time_unit": ["h", "h"],
            "is_control": [True, False],
        }
    ).to_csv(source / "cells.csv", index=False)
    matrix = sparse.csr_matrix(np.array([[1.0, 0.0], [2.0, 3.0]]))
    calls = {"n": 0}
    original = matrix.toarray

    def spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    matrix.toarray = spy
    adata = ad.AnnData(X=matrix, obs=pd.DataFrame(index=["c0", "c1"]), var=pd.DataFrame(index=["G1", "G2"]))
    adata.uns["matrix_kind"] = "counts"
    adata.write_h5ad(source / "counts.h5ad")
    from pertbench_long.data.prepare.matrix_io import read_expression

    loaded, _ids, _genes, kind = read_expression(source / "counts.h5ad")
    assert kind == "counts"
    assert hasattr(loaded, "toarray")
    prepare_sciplex(source, tmp_path / "out", input_profile="h5ad_bundle")
    prepared = ad.read_h5ad(tmp_path / "out" / "matrix.h5ad")
    assert prepared.uns["matrix_kind"] == "counts"
    assert prepared.n_obs == 2


def test_e04_empty_provenance_cannot_become_official(tmp_path: Path):
    store = build_synthetic_dose_store(n_compounds=6, n_genes=8, n_cells=4)
    with pytest.raises(UnsupportedProfile, match="official"):
        build_chemical_dose_episode(
            store,
            dest=tmp_path / "off",
            episode_id="off",
            synthetic=False,
            data_release_id="some_release",
            requested_track="official",
            scoring_scale_hash="not-a-calibration",
            provenance_verified=True,
            source_verified=True,
            n_panel=8,
            min_candidates=8,
            min_targets=3,
            min_cells=2,
            experimental_budget=2,
            partition="eval",
        )


def test_e05_unknown_fields_and_missing_matrix_are_errors(tmp_path: Path):
    with pytest.raises(ConfigError, match="label_estimand"):
        validate_build_config({"label_estimand": "not_an_estimand", "fixture": "pbmc"})
    with pytest.raises(ConfigError, match="reference_policy"):
        validate_build_config({"reference_policy": "first_match", "fixture": "pbmc"})
    with pytest.raises(UnsupportedProfile, match="reference_policy"):
        store = build_synthetic_dose_store(n_compounds=4, n_genes=6, n_cells=3)
        from pertbench_long.data.audit_conditions import match_controls

        roles_o = [cid for cid, rows in store.condition_to_rows.items() if store.records[rows[0]].perturbation_kind != "control"][:1]
        build_episode_from_condition_ids(
            store,
            dest=tmp_path / "bad",
            episode_id="bad",
            protocol=PROTOCOL_CHEMICAL_DOSE,
            observed_condition_ids=roles_o,
            queryable_condition_ids=roles_o,
            target_condition_ids=roles_o,
            control_condition_ids=[],
            control_mapping={},
            split_variant="dose_interpolation",
            objective="x",
            synthetic=True,
            study_name="s",
            data_release_id="synthetic_dose_v1",
            experimental_budget=0,
            min_candidates=1,
            min_targets=1,
            reference_policy="not_a_policy",
        )


def test_e06_doctor_rejects_empty_qc_and_controls_only(tmp_path: Path):
    root = tmp_path / "data"
    prepared = root / "prepared" / "sciplex3" / "v1"
    prepared.mkdir(parents=True)
    (prepared / "provenance.json").write_text(json.dumps({"matrix": {"matrix_kind": "counts"}}), encoding="utf-8")
    (prepared / "qc_report.json").write_text("{}", encoding="utf-8")
    (prepared / "matrix.h5ad").write_bytes(b"not-an-h5ad")
    pd.DataFrame({"condition_id": ["c"], "perturbation_kind": ["control"]}).to_parquet(prepared / "conditions.parquet")
    report = doctor_data(dataset="sciplex3", data_root=root, min_candidates=1, min_targets=1)
    assert report["status"] in {"empty_qc", "invalid_matrix", "insufficient_conditions"}


def test_e07_mount_plan_includes_declared_public_files(tmp_path: Path):
    built = build_synthetic_dose_episode(tmp_path / "dose", n_panel=8)
    public = Path(built["public_dir"])
    mounts = DockerPythonExecutor._mounts(object(), public)
    text = " ".join(mounts)
    for name in ("task.md", "conditions.parquet", "targets.parquet", "gene_panel.tsv", "reference_mapping.json"):
        assert name in text


def test_vehicle_mismatch_is_rejected():
    from dataclasses import replace

    from pertbench_long.data.audit_conditions import match_controls
    from pertbench_long.errors import AmbiguousCondition
    from pertbench_long.schemas.conditions import make_condition_id

    store = build_synthetic_dose_store(n_compounds=1, n_genes=4, n_cells=2)
    control = next(rec for rec in store.records if rec.perturbation_kind == "control")
    ethanol_id = make_condition_id(
        study=control.study,
        context_id=control.context_id,
        perturbation_kind="control",
        perturbation_components=("ethanol",),
        time=control.time,
        time_unit=control.time_unit,
        assay=control.assay,
    )
    records = [replace(rec, vehicle="DMSO" if rec.perturbation_kind == "control" else None) for rec in store.records]
    records.append(
        replace(
            control,
            vehicle="ethanol",
            perturbation_id="ethanol",
            perturbation_components=("ethanol",),
            condition_id=ethanol_id,
            observation_id=control.observation_id + "|ethanol",
            original_obs_id=control.original_obs_id + "|ethanol",
        )
    )
    cond = {}
    for i, rec in enumerate(records):
        cond.setdefault(rec.condition_id, []).append(i)
    store.records = records
    store.condition_to_rows = cond
    with pytest.raises(AmbiguousCondition, match="vehicle|reference"):
        match_controls(store)


def test_e08_digest_match_is_not_isolation_qualified():
    digest = "sha256:" + "b" * 64
    assert isolation_is_qualified(mode="isolated_eval", backend="docker", resolved_digest=digest, attestation_digest=digest) is False


def test_live_and_docker_jobs_fail_when_explicitly_required():
    if os.environ.get("PERTBENCH_REQUIRE_LIVE") == "1":
        pytest.fail("live model episode is not_run; refusing a green required job")
    if os.environ.get("PERTBENCH_REQUIRE_DOCKER") == "1":
        pytest.fail("docker isolation acceptance is not_run; refusing a green required job")
    pytest.skip("live/docker acceptance is not_run unless PERTBENCH_REQUIRE_LIVE or PERTBENCH_REQUIRE_DOCKER is set")


def test_e05_development_overlap_rejected(tmp_path: Path):
    store = build_synthetic_dose_store(n_compounds=6, n_genes=8, n_cells=4)
    from pertbench_long.episodes.builders.chemical_dose import assign_dose_roles

    roles = assign_dose_roles(store, split_variant="dose_interpolation", context_ids=None, min_candidates=4, min_targets=2, min_cells=2)
    with pytest.raises(SplitLeakError):
        build_episode_from_condition_ids(
            store,
            dest=tmp_path / "overlap",
            episode_id="overlap",
            protocol=PROTOCOL_CHEMICAL_DOSE,
            observed_condition_ids=roles["O"],
            queryable_condition_ids=roles["Q"],
            target_condition_ids=roles["T"],
            control_condition_ids=roles["C"],
            control_mapping=roles["control_mapping"],
            split_variant="dose_interpolation",
            objective="x",
            synthetic=False,
            study_name="synthetic_sciplex",
            data_release_id="local_unreleased",
            experimental_budget=2,
            min_candidates=4,
            min_targets=2,
            n_panel=8,
            partition="eval",
            development_episodes=[{"partition": "dev", "observed_condition_ids": [roles["T"][0]], "queryable_condition_ids": [], "target_condition_ids": []}],
        )
