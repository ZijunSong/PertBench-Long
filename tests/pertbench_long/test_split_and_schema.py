from __future__ import annotations

import json
from pathlib import Path

import pytest

from pertbench_long.errors import SchemaError, SplitLeakError
from pertbench_long.episodes.builder import build_episode_from_store, build_synthetic_episode
from pertbench_long.episodes.export import contains_marker_bytes, export_public_bundle
from pertbench_long.episodes.split import audit_split
from pertbench_long.episodes.synthetic import build_synthetic_store
from pertbench_long.schemas.validate import FieldError, validate_public_payload


def test_t01_overlap_rejected():
    store = build_synthetic_store()
    recs = store.records
    o = [r.observation_id for r in recs if r.cell_type == "B" and r.perturbation_id == "IFN"]
    t = o
    with pytest.raises((SplitLeakError, FieldError, SchemaError)):
        audit_split(
            recs,
            observed_obs=o,
            queryable_obs=[],
            target_obs=t,
            control_obs=[],
            observed_conditions=["same"],
            queryable_conditions=[],
            target_conditions=["same"],
            protocol="within_study_celltype_ood_v1",
            public_data=True,
        )


def test_t02_development_overlap_is_pilot_not_private(tmp_path: Path):
    store = build_synthetic_store()
    t_cond = [r.condition_key() for r in store.records if r.cell_type == "B" and r.perturbation_id == "IFN"][:1]
    audit = audit_split(
        store.records,
        observed_obs=[],
        queryable_obs=[],
        target_obs=[],
        control_obs=[],
        observed_conditions=["o"],
        queryable_conditions=["q"],
        target_conditions=t_cond,
        development_target_conditions=t_cond,
        protocol="within_study_celltype_ood_v1",
        public_data=True,
    )
    assert audit["status"] == "pilot_not_private_independent_test"


def test_t03_hidden_values_do_not_change_public_fingerprint(tmp_path: Path):
    a = build_synthetic_episode(tmp_path / "a", hidden_shift=0.0)
    b = build_synthetic_episode(tmp_path / "b", hidden_shift=1.5)
    pub_a = json.loads((Path(a["public_dir"]) / "episode.json").read_text())
    pub_b = json.loads((Path(b["public_dir"]) / "episode.json").read_text())
    assert pub_a["artifact_metadata"]["public_bytes_fingerprint"] == pub_b["artifact_metadata"]["public_bytes_fingerprint"]
    import anndata as ad

    xa = ad.read_h5ad(Path(a["public_dir"]) / "evidence" / "ev_initial_001.h5ad").X
    xb = ad.read_h5ad(Path(b["public_dir"]) / "evidence" / "ev_initial_001.h5ad").X
    import numpy as np

    assert np.allclose(xa, xb)


def test_t04_scrub_hides_raw_uns_layers(tmp_path: Path):
    import anndata as ad
    import numpy as np
    import pandas as pd

    marker = "TARGET_MARKER_SECRET"
    X = np.zeros((2, 2), dtype=np.float32)
    obs = pd.DataFrame({"observation_id": ["a", "b"], "study": ["s", "s"], "species": ["synthetic", "synthetic"], "cell_type": ["CD4T", "CD4T"], "perturbation_id": ["IFN", "Control"], "dose": [None, None], "dose_unit": ["none", "none"], "time": [None, None], "time_unit": ["none", "none"], "assay": ["scrna", "scrna"], "original_obs_id": ["a", "b"], "sample_id": ["s", "s"]})
    obs.index = obs["observation_id"]
    dest = tmp_path / "public.h5ad"
    export_public_bundle(
        matrix=X,
        records=obs.to_dict(orient="records"),
        gene_ids=["G1", "G2"],
        uns={"ok": True, "de_table_target": marker, "target_marker": marker},
        dest=dest,
        artifact_id="ev",
    )
    assert not contains_marker_bytes(dest, marker)
    loaded = ad.read_h5ad(dest)
    assert loaded.raw is None
    assert list(loaded.layers.keys()) == []
    assert "de_table_target" not in loaded.uns
    assert "target_marker" not in loaded.uns


def test_unknown_public_field_errors():
    payload = {
        "schema_version": "1.0",
        "episode_id": "x",
        "protocol": "within_study_celltype_ood_v1",
        "protocol_version": "1.0",
        "label_profile": "effect_proxy_v1",
        "objective": "obj",
        "initial_evidence": ["e"],
        "reference_evidence": ["c"],
        "candidate_experiments": [{"experiment_id": "q_01", "cell_type": "B", "perturbation": "IFN", "cost": 1}],
        "targets": [{"target_id": "t_01", "cell_type": "NK", "perturbation": "IFN"}],
        "target_control_available": True,
        "gene_universe_artifact": "genes_v1.tsv",
        "experimental_budget": 1,
        "cost_unit": "credit",
        "resource_profile": "cpu_pilot_v1",
        "data_release_id": "d",
        "public_split_fingerprint": "a",
        "public_scoring_fingerprint": "b",
        "label_path": "/secret",
    }
    with pytest.raises(FieldError) as exc:
        validate_public_payload(payload)
    assert "label_path" in str(exc.value)


def test_incompatible_schema_version_errors():
    with pytest.raises(FieldError) as exc:
        validate_public_payload(
            {
                "schema_version": "9.9",
                "episode_id": "x",
                "protocol": "within_study_celltype_ood_v1",
                "protocol_version": "1.0",
                "label_profile": "effect_proxy_v1",
                "objective": "obj",
                "initial_evidence": ["e"],
                "reference_evidence": ["c"],
                "candidate_experiments": [{"experiment_id": "q_01", "cell_type": "B", "perturbation": "IFN", "cost": 1}],
                "targets": [{"target_id": "t_01", "cell_type": "NK", "perturbation": "IFN"}],
                "target_control_available": True,
                "gene_universe_artifact": "genes_v1.tsv",
                "experimental_budget": 1,
                "cost_unit": "credit",
                "resource_profile": "cpu_pilot_v1",
                "data_release_id": "d",
                "public_split_fingerprint": "a",
                "public_scoring_fingerprint": "b",
            }
        )
    assert "schema_version" in str(exc.value)
