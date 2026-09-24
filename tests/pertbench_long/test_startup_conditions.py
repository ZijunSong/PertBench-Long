"""Regressions for the 5bf28d6 startup review (C01–C10)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pertbench_long.cli import main
from pertbench_long.data.acquire import fetch_dataset
from pertbench_long.data.doctor import doctor_data
from pertbench_long.data.prepare.matrix_io import read_expression
from pertbench_long.data.prepare.norman import prepare_norman
from pertbench_long.episodes.evidence import verify_official_evidence
from pertbench_long.episodes.synthetic_long import build_synthetic_dose_episode
from pertbench_long.errors import IntegrityError, SchemaError, UnsupportedProfile
from pertbench_long.runtime.config import resolve_run_config
from pertbench_long.runtime.executor import isolation_is_qualified


def _load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_c01_run_templates_expand_data_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PERTBENCH_DATA_ROOT", str(tmp_path))
    built = build_synthetic_dose_episode(tmp_path / "episodes" / "sciplex_dose_pilot_0001", n_panel=8)
    repo = Path(__file__).resolve().parents[2]
    for name in ("run_sciplex_dose.yaml", "run_norman_pair.yaml", "run_sciplex_context.yaml"):
        cfg = _load(repo / "configs" / "pertbench_long" / name)
        if name != "run_sciplex_dose.yaml":
            # Only the dose episode is built; the other templates must still expand.
            resolved = resolve_run_config(yaml_cfg=cfg, config_path=repo / "configs" / "pertbench_long" / name)
            assert "${" not in resolved["public_dir"]
            assert resolved["public_dir"].startswith(str(tmp_path))
            continue
        resolved = resolve_run_config(yaml_cfg=cfg, config_path=repo / "configs" / "pertbench_long" / name)
        assert Path(resolved["public_dir"]).is_dir()
        assert Path(resolved["private_manifest"]).is_file()
    rc = main(
        [
            "run",
            "--config",
            str(repo / "configs" / "pertbench_long" / "run_sciplex_dose.yaml"),
            "--agent",
            "scripted_mock",
            "--mode",
            "local_trusted_debug",
            "--allow-unqualified",
            "--output",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 0
    assert built["episode_id"] == "sciplex_dose_synth_0001" or Path(built["public_dir"]).exists()


def test_c01_missing_env_and_cli_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("PERTBENCH_DATA_ROOT", raising=False)
    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(Exception, match="PERTBENCH_DATA_ROOT"):
        resolve_run_config(
            yaml_cfg=_load(repo / "configs" / "pertbench_long" / "run_sciplex_dose.yaml"),
            config_path=repo / "configs" / "pertbench_long" / "run_sciplex_dose.yaml",
        )
    episode = tmp_path / "rel" / "public"
    episode.mkdir(parents=True)
    private = tmp_path / "rel" / "private" / "manifest.json"
    private.parent.mkdir()
    private.write_text("{}", encoding="utf-8")
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "mode: local_trusted_debug\nagent: scripted_mock\npublic_dir: rel/public\nprivate_manifest: rel/private/manifest.json\n",
        encoding="utf-8",
    )
    resolved = resolve_run_config(yaml_cfg=_load(cfg_path), config_path=cfg_path)
    assert Path(resolved["public_dir"]) == episode.resolve()
    override = resolve_run_config(
        cli={"public_dir": str(tmp_path / "other"), "private_manifest": str(private)},
        yaml_cfg={"mode": "local_trusted_debug", "public_dir": "${UNSET_DATA_ROOT}/x", "private_manifest": "${UNSET_DATA_ROOT}/y"},
        config_path=cfg_path,
    )
    assert override["public_dir"] == str(tmp_path / "other")


def _mex(source: Path, matrix: np.ndarray, barcodes: list[str], genes: list[str]) -> None:
    from scipy import io as spio
    from scipy import sparse

    source.mkdir(parents=True)
    spio.mmwrite(source / "matrix.mtx", sparse.csr_matrix(matrix))
    (source / "barcodes.tsv").write_text("\n".join(barcodes) + "\n", encoding="utf-8")
    (source / "genes.tsv").write_text("\n".join(genes) + "\n", encoding="utf-8")
    pd.DataFrame({"cell_barcode": barcodes, "guide_identity": ["NTC"] * len(barcodes), "cell_type": ["K562"] * len(barcodes)}).to_csv(
        source / "cell_identities.csv", index=False
    )


def test_c02_mex_three_by_two_becomes_cells_by_genes(tmp_path: Path):
    import anndata as ad

    _mex(tmp_path / "src", np.arange(6, dtype=float).reshape(3, 2), ["b0", "b1"], ["g0", "g1", "g2"])
    prepare_norman(tmp_path / "src", tmp_path / "out", input_profile="mex_bundle")
    adata = ad.read_h5ad(tmp_path / "out" / "matrix.h5ad")
    assert adata.n_obs == 2 and adata.n_vars == 3


def test_c03_mex_square_values_are_transposed(tmp_path: Path):
    import anndata as ad

    _mex(tmp_path / "src", np.array([[1.0, 11.0], [2.0, 12.0]]), ["b0", "b1"], ["g0", "g1"])
    prepare_norman(tmp_path / "src", tmp_path / "out", input_profile="mex_bundle")
    adata = ad.read_h5ad(tmp_path / "out" / "matrix.h5ad")
    got = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    assert np.allclose(got, np.array([[1.0, 2.0], [11.0, 12.0]]))
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert "barcodes.tsv" in provenance["source_files"]
    assert "genes.tsv" in provenance["source_files"]


def test_c04_undeclared_h5ad_is_not_counts(tmp_path: Path):
    import anndata as ad

    path = tmp_path / "counts.h5ad"
    values = np.log1p(np.array([[2.0, 9.0], [0.0, 3.0]]))
    ad.AnnData(X=values, obs=pd.DataFrame(index=["c0", "c1"]), var=pd.DataFrame(index=["G1", "G2"])).write_h5ad(path)
    with pytest.raises(SchemaError, match="undeclared"):
        read_expression(path)


def test_c04_explicit_counts_layer_is_accepted(tmp_path: Path):
    import anndata as ad

    adata = ad.AnnData(X=np.array([[-1.0, 0.2], [0.1, -0.3]]), obs=pd.DataFrame(index=["c0", "c1"]), var=pd.DataFrame(index=["G1", "G2"]))
    adata.uns["matrix_kind"] = "scaled"
    adata.layers["counts"] = np.array([[2.0, 0.0], [3.0, 1.0]])
    adata.uns["layer_kinds"] = {"counts": "counts"}
    adata.write_h5ad(tmp_path / "counts.h5ad")
    matrix, _ids, _genes, kind = read_expression(tmp_path / "counts.h5ad", layer="counts")
    assert kind == "counts"
    assert np.allclose(np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix), np.array([[2.0, 0.0], [3.0, 1.0]]))


def test_c05_doctor_blocks_failed_qc_and_missing_controls(tmp_path: Path):
    import anndata as ad

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ad.AnnData(X=np.ones((1, 1)), obs=pd.DataFrame(index=["c0"]), var=pd.DataFrame(index=["G1"])).write_h5ad(prepared / "matrix.h5ad")
    (prepared / "provenance.json").write_text(json.dumps({"provenance_version": "prepared_v1", "matrix": {"matrix_kind": "counts"}}), encoding="utf-8")
    (prepared / "qc_report.json").write_text(json.dumps({"status": "failed", "n_cells": 1}), encoding="utf-8")
    pd.DataFrame({"condition_id": [f"t{i}" for i in range(32)], "perturbation_kind": ["chemical"] * 32, "n_cells": [1] * 32}).to_parquet(prepared / "conditions.parquet")
    (prepared / "prepared_manifest.json").write_text("{}", encoding="utf-8")
    report = doctor_data(dataset="sciplex3", data_root=tmp_path, config={"data_dir": str(prepared)}, min_candidates=1, min_targets=1)
    assert report["status"] != "ready_for_pilot"
    assert report["reason"] == "qc_failed"


def test_c06_locked_flag_without_files_is_not_official(tmp_path: Path):
    data = tmp_path / "prepared"
    data.mkdir()
    (data / "matrix.h5ad").write_bytes(b"x")
    (data / "conditions.parquet").write_bytes(b"x")
    (data / "qc_report.json").write_text(json.dumps({"status": "ok", "n_cells": 1}), encoding="utf-8")
    provenance = {
        "provenance_version": "prepared_v1",
        "source_lock_status": "locked",
        "source_files": {"missing.csv": {"sha256": "not-a-hash", "path": str(tmp_path / "missing.csv")}},
        "matrix": {"matrix_kind": "counts"},
    }
    (data / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    from pertbench_long.hashes import sha256_file

    manifest = {
        "manifest_version": "prepared_manifest_v1",
        "provenance_sha256": sha256_file(data / "provenance.json"),
        "outputs": {
            "matrix.h5ad": sha256_file(data / "matrix.h5ad"),
            "conditions.parquet": sha256_file(data / "conditions.parquet"),
            "qc_report.json": sha256_file(data / "qc_report.json"),
        },
    }
    (data / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calibration = tmp_path / "scale.json"
    calibration.write_text(
        json.dumps({"calibration_version": "scoring_calibration_v1", "label_profile": "lognorm_cellmean_delta_v1", "a_family": 0.5, "fit_on": "fixture"}),
        encoding="utf-8",
    )
    with pytest.raises((UnsupportedProfile, IntegrityError)):
        verify_official_evidence(
            data,
            calibration_path=calibration,
            a_family=0.5,
            label_profile="lognorm_cellmean_delta_v1",
            split_manifest={"partition": "development"},
            declared_scale_hash=None,
            partition="eval",
            protocol="chemical_dose_acquisition_v1",
        )


def test_c07_h5ad_profile_does_not_require_counts_csv(tmp_path: Path):
    root = tmp_path / "data"
    dest = root / "raw" / "sciplex3" / "sciplex3_v1"
    dest.mkdir(parents=True)
    (dest / "cells.csv").write_text("cell_barcode\n", encoding="utf-8")
    (dest / "counts.h5ad").write_bytes(b"h5")
    manifest = Path(__file__).resolve().parents[2] / "data" / "acquisition" / "sciplex3_v1.json"
    report = fetch_dataset("sciplex3", data_root=root, manifest_path=manifest, offline=True, input_profile="h5ad_bundle")
    assert report.status == "ok"
    assert "counts.csv" not in report.missing
    with pytest.raises(Exception, match="input_profile"):
        fetch_dataset("sciplex3", data_root=root, manifest_path=manifest, offline=True, input_profile="not_a_profile")


def test_c09_only_a_matching_passed_report_qualifies():
    digest = "sha256:" + "c" * 64
    passed = {
        "report_version": "isolation_acceptance_v1",
        "status": "passed",
        "image_digest": digest,
        "mount_policy": "declared_public_v1",
        "tests": {name: "passed" for name in ("public_read", "public_readonly", "host_sentinel_unreadable", "unpurchased_unreadable", "no_network")},
    }
    assert isolation_is_qualified(mode="isolated_eval", backend="docker", resolved_digest=digest, attestation_digest=digest, acceptance_report=passed)
    assert isolation_is_qualified(mode="isolated_eval", backend="docker", resolved_digest=digest, attestation_digest=digest, acceptance_report=None) is False
    stale = dict(passed)
    stale["mount_policy"] = "old"
    assert isolation_is_qualified(mode="isolated_eval", backend="docker", resolved_digest=digest, attestation_digest=digest, acceptance_report=stale) is False


def test_c08_acceptance_script_invokes_docker_when_configured(tmp_path: Path, monkeypatch):
    import scripts.isolation_acceptance as acceptance

    monkeypatch.setattr(acceptance.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setenv("PERTBENCH_ANALYSIS_IMAGE", "example@sha256:" + "d" * 64)
    calls = []

    class Proc:
        returncode = 1
        stdout = ""
        stderr = "blocked"

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return Proc()

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)
    code = acceptance.main(["isolation_acceptance.py", str(tmp_path / "report.json")])
    assert calls and calls[0][0] == "docker"
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert code == 3
