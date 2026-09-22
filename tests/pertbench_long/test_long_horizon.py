from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pertbench_long.data.audit_conditions import REASON_INSUFFICIENT, audit_conditions
from pertbench_long.data.preprocess import to_effect_space
from pertbench_long.data.sparse_ops import extract_dense_block, library_sizes
from pertbench_long.episodes.builders.chemical_dose import build_chemical_dose_episode
from pertbench_long.episodes.split import audit_chemical_dose_split, audit_release_cross_contamination
from pertbench_long.episodes.synthetic import build_synthetic_store
from pertbench_long.episodes.synthetic_long import (
    build_synthetic_context_episode,
    build_synthetic_dose_episode,
    build_synthetic_dose_store,
    build_synthetic_pair_episode,
    build_synthetic_pair_store,
)
from pertbench_long.errors import InsufficientEligible, SplitLeakError, UnknownUnit, UnsupportedProtocol
from pertbench_long.evaluation.scoring import aubc, score_continuous_predictions
from pertbench_long.evaluation.submission import validate_predictions
from pertbench_long.runtime.runner import EpisodeRunner, score_run
from pertbench_long.schemas.conditions import doses_equivalent, make_condition_id, normalize_components
from pertbench_long.schemas.types import IMPLEMENTED_PROTOCOLS, PROTOCOL_CHEMICAL_DOSE, PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload
from pertbench_long.tasks.registry import get_task


def test_v01_old_pbmc_protocol_still_implemented():
    assert PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD in IMPLEMENTED_PROTOCOLS
    spec = get_task(PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD)
    assert spec.label_profile == "effect_proxy_v1"


def test_v03_unit_and_pair_identity():
    assert doses_equivalent(1, "uM", 1000, "nM")
    assert make_condition_id(study="s", context_id="A549", perturbation_kind="chemical", perturbation_components=["DrugX"], dose=1, dose_unit="uM") == make_condition_id(
        study="s", context_id="A549", perturbation_kind="chemical", perturbation_components=["DrugX"], dose=1000, dose_unit="nM"
    )
    assert make_condition_id(study="s", context_id="A549", perturbation_kind="chemical", perturbation_components=["DrugX"], dose=1, dose_unit="uM") != make_condition_id(
        study="s", context_id="K562", perturbation_kind="chemical", perturbation_components=["DrugX"], dose=1, dose_unit="uM"
    )
    assert normalize_components(["B", "A"], kind="genetic_pair") == ("A", "B")
    assert make_condition_id(study="s", context_id="K562", perturbation_kind="genetic_pair", perturbation_components=["B", "A"]) == make_condition_id(
        study="s", context_id="K562", perturbation_kind="genetic_pair", perturbation_components="A+B"
    )
    with pytest.raises(UnknownUnit):
        make_condition_id(study="s", context_id="A549", perturbation_kind="chemical", perturbation_components=["DrugX"], dose=1, dose_unit="unknown", require_exact_dose=True)


def test_v02_builder_does_not_take_first_match(tmp_path: Path):
    store = build_synthetic_dose_store()
    same_drug = [r for r in store.records if r.perturbation_id == "Cpd01"]
    assert len({r.condition_id for r in same_drug}) == 4
    built = build_chemical_dose_episode(store, dest=tmp_path / "dose", episode_id="dose_test", synthetic=True, n_panel=16)
    pub = validate_public_payload(__import__("json").loads(Path(built["public_dir"], "episode.json").read_text()))
    doses = [c.dose for c in pub.candidate_experiments if c.perturbation == "Cpd01"]
    assert len(set(doses)) >= 1


def test_v05_sparse_library_size_matches_dense():
    dense = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0]])
    from scipy import sparse

    csr = sparse.csr_matrix(dense)
    assert np.allclose(library_sizes(dense), library_sizes(csr))
    assert np.allclose(extract_dense_block(csr, [1], [0, 2]), dense[1:2, [0, 2]])
    z_dense = to_effect_space(dense, "counts", full_universe=dense)
    z_sparse = to_effect_space(csr, "counts", full_universe=csr)
    assert np.allclose(z_dense, z_sparse)


def test_v10_extrapolation_rejects_higher_visible_dose():
    store = build_synthetic_dose_store()
    recs = store.records
    by = {}
    for r in recs:
        by.setdefault(r.condition_id, r)
    treated = [r for r in by.values() if r.perturbation_id == "Cpd01"]
    treated = sorted(treated, key=lambda r: r.dose or 0)
    with pytest.raises(SplitLeakError):
        audit_chemical_dose_split(
            recs,
            observed_obs=[],
            queryable_obs=[],
            target_obs=[],
            control_obs=[],
            observed_conditions=[treated[0].condition_id],
            queryable_conditions=[treated[-1].condition_id],
            target_conditions=[treated[1].condition_id],
            protocol="chemical_dose_acquisition_v1",
            split_variant="dose_extrapolation",
            public_data=False,
        )


def test_v12_aubc_full_curve():
    scores = {i: 0.1 * i for i in range(0, 9)}
    got = aubc(scores, 8)
    expect = (0.5 * scores[0] + sum(scores[b] for b in range(1, 8)) + 0.5 * scores[8]) / 8
    assert got == pytest.approx(expect)


def test_v16_continuous_submission_rejects_nan_and_missing():
    genes = ["G1", "G2"]
    frame = pd.DataFrame({"target_id": ["t_01", "t_01"], "gene_id": ["G1", "G2"], "predicted_effect": [0.1, np.nan]})
    with pytest.raises(Exception):
        validate_predictions(frame, target_ids=["t_01"], gene_ids=genes, require_direction=False)
    short = pd.DataFrame({"target_id": ["t_01"], "gene_id": ["G1"], "predicted_effect": [0.1]})
    with pytest.raises(Exception):
        validate_predictions(short, target_ids=["t_01"], gene_ids=genes, require_direction=False)


def test_v17_constant_correlation_is_na():
    labels = pd.DataFrame(
        {
            "target_id": ["t_01", "t_01"],
            "gene_id": ["G1", "G2"],
            "reference_effect": [0.2, 0.2],
        }
    )
    pred = pd.DataFrame(
        {
            "target_id": ["t_01", "t_01"],
            "gene_id": ["G1", "G2"],
            "predicted_effect": [0.1, 0.3],
        }
    )
    scored = score_continuous_predictions(pred, labels, target_ids=["t_01"], gene_ids=["G1", "G2"], a_family=0.5)
    assert scored["per_target"][0]["pearson_delta"] is None
    assert scored["per_target"][0]["pearson_delta_na_reason"] == "constant_vector"


def test_v22_insufficient_scale_is_blocked():
    store = build_synthetic_store()
    report = audit_conditions(store, min_candidates=24, min_targets=8)
    assert REASON_INSUFFICIENT in report["reasons"]
    with pytest.raises(InsufficientEligible):
        build_chemical_dose_episode(store, dest=Path("/tmp/nope"), episode_id="x", synthetic=True, min_candidates=24, min_targets=8)


def test_v09_release_cross_contamination():
    with pytest.raises(SplitLeakError):
        audit_release_cross_contamination(
            [
                {"partition": "dev", "target_condition_ids": ["c1"], "queryable_condition_ids": [], "observed_condition_ids": []},
                {"partition": "eval", "target_condition_ids": [], "queryable_condition_ids": ["c1"], "observed_condition_ids": []},
            ]
        )


def test_unknown_protocol_fails():
    with pytest.raises(UnsupportedProtocol):
        get_task("not_a_protocol_v9")


def test_dose_and_pair_fixtures_build_and_mock(tmp_path: Path):
    dose = build_synthetic_dose_episode(tmp_path / "dose", n_panel=16)
    pair = build_synthetic_pair_episode(tmp_path / "pair", n_panel=12)
    ctx = build_synthetic_context_episode(tmp_path / "ctx", n_panel=12)
    for built in (dose, pair, ctx):
        pub = validate_public_payload(__import__("json").loads(Path(built["public_dir"], "episode.json").read_text()))
        priv = validate_private_payload(__import__("json").loads(Path(built["private_manifest"]).read_text()))
        assert pub.synthetic is True
        assert priv.public.experimental_budget in {8, 16}
        assert len(pub.candidate_experiments) >= pub.experimental_budget
        assert pub.label_profile == "lognorm_cellmean_delta_v1"
        run = EpisodeRunner(
            public_dir=Path(built["public_dir"]),
            private_manifest=Path(built["private_manifest"]),
            run_root=tmp_path / f"run_{built['episode_id']}",
            agent="scripted_mock",
            agent_kwargs={"policy": "two_query"},
        ).run()
        assert run["outcome"] == "completed", run
        scores = score_run(Path(run["run_dir"]), Path(built["private_manifest"]))
        assert scores["score_status"] == "scored"
        assert "scores_by_budget" in scores
        assert set(map(int, scores["scores_curve"])) >= {0}
        assert scores["synthetic"] is True


def test_adapter_roundtrip_and_shuffled_order(tmp_path: Path):
    from pertbench_long.data.adapters.sciplex import import_sciplex
    from pertbench_long.data.adapters.norman import import_norman
    import anndata as ad

    store = build_synthetic_dose_store(n_compounds=3, n_genes=8, n_cells=4)
    obs = pd.DataFrame([{**r.__dict__, "perturbation_components": "+".join(r.perturbation_components)} for r in store.records])
    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].map(lambda v: "" if v is None else v)
    obs["is_control"] = obs["perturbation_kind"] == "control"
    adata = ad.AnnData(X=store.matrix, obs=obs, var=pd.DataFrame(index=store.gene_ids))
    path = tmp_path / "matrix.h5ad"
    adata.write_h5ad(path)
    loaded = import_sciplex(tmp_path, declared_matrix_kind="counts", require_exact_dose=False)
    assert loaded.summary.n_kept == store.summary.n_kept
    assert len(loaded.condition_ids()) == len(store.condition_ids())
    assert loaded.gene_ids == store.gene_ids

    store2 = build_synthetic_pair_store(n_genes=8, n_cells=4, extra_pairs=6)
    obs2 = pd.DataFrame([{**r.__dict__, "perturbation_components": "+".join(r.perturbation_components)} for r in store2.records])
    for col in obs2.columns:
        if obs2[col].dtype == object:
            obs2[col] = obs2[col].map(lambda v: "" if v is None else v)
    obs2["is_control"] = obs2["perturbation_kind"] == "control"
    ad.AnnData(X=store2.matrix, obs=obs2, var=pd.DataFrame(index=store2.gene_ids)).write_h5ad(path)
    loaded2 = import_norman(tmp_path, declared_matrix_kind="counts")
    assert any(r.perturbation_kind == "genetic_pair" for r in loaded2.records)
