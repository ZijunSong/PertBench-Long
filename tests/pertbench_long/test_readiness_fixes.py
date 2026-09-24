"""Regressions for the 2026-09-22 new-task readiness review (A01–A20)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pertbench_long.data.acquire import fetch_dataset
from pertbench_long.data.audit_conditions import match_controls
from pertbench_long.data.doctor import doctor_data
from pertbench_long.data.parsing import parse_bool
from pertbench_long.data.prepare.norman import prepare_norman, resolve_norman_identity
from pertbench_long.data.prepare.sciplex import prepare_sciplex
from pertbench_long.data.preprocess import to_effect_space
from pertbench_long.episodes.builders.chemical_dose import assign_dose_roles, build_chemical_dose_episode
from pertbench_long.episodes.builders.common import build_episode_from_condition_ids
from pertbench_long.episodes.builders.genetic_pair import assign_pair_roles
from pertbench_long.episodes.split import audit_release_cross_contamination
from pertbench_long.episodes.synthetic_long import (
    build_synthetic_context_episode,
    build_synthetic_dose_episode,
    build_synthetic_dose_store,
    build_synthetic_pair_episode,
    build_synthetic_pair_store,
)
from pertbench_long.errors import (
    AmbiguousCondition,
    ConfigError,
    IntegrityError,
    SchemaError,
    SplitLeakError,
    UnknownUnit,
    UnsupportedProfile,
)
from pertbench_long.evaluation.labels import build_lognorm_cellmean_delta_labels
from pertbench_long.hashes import sha256_file
from pertbench_long.runtime.release import check_release, copy_declared_public_inputs
from pertbench_long.runtime.runner import EpisodeRunner, score_run
from pertbench_long.schemas.conditions import PERTURBATION_CONTROL, PERTURBATION_GENETIC_SINGLE
from pertbench_long.schemas.types import PROTOCOL_CHEMICAL_DOSE
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload


def _write_author_sciplex(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    genes = ["G1", "G2", "G3"]
    rows = []
    meta = []
    barcode = 0
    for ctx, drug, dose, unit, time, ctrl, vehicle in [
        ("A549", "DMSO", None, None, 24, True, "DMSO"),
        ("A549", "DrugA", 1, "uM", 24, False, "DMSO"),
        ("A549", "DrugA", 1000, "nM", 24, False, "DMSO"),
        ("A549", "DrugB", 10, "nM", 24, False, "DMSO"),
    ]:
        for _ in range(4):
            bid = f"c{barcode}"
            barcode += 1
            rows.append([2, 0, 1] if ctrl else [2 + int(dose or 0) // 100, 1, 1])
            meta.append(
                {
                    "cell_barcode": bid,
                    "cell_type": ctx,
                    "perturbation": drug,
                    "dose": dose,
                    "dose_unit": unit,
                    "time": time,
                    "time_unit": "h",
                    "vehicle": vehicle,
                    "is_control": ctrl,
                    "sample_id": "s1",
                    "replicate_id": "r1",
                }
            )
    pd.DataFrame(meta).to_csv(root / "cells.csv", index=False)
    pd.DataFrame(rows, index=[m["cell_barcode"] for m in meta], columns=genes).to_csv(root / "counts.csv")
    return root


def _write_author_norman(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    genes = ["G1", "G2"]
    meta = []
    rows = []
    items = [("NTC", True), ("TF1", False), ("TF2", False), ("TF1+NTC", False), ("TF1+TF2", False)]
    n = 0
    for ident, ctrl in items:
        for _ in range(3):
            bid = f"b{n}"
            n += 1
            meta.append({"cell_barcode": bid, "guide_identity": ident, "cell_type": "K562", "is_control": ctrl})
            rows.append([1, 0] if ctrl else [3, 1])
    pd.DataFrame(meta).to_csv(root / "cell_identities.csv", index=False)
    pd.DataFrame(rows, index=[m["cell_barcode"] for m in meta], columns=genes).to_csv(root / "counts.csv")
    pd.DataFrame([{"guide": "NTC", "gene": "control", "is_ntc": True}]).to_csv(root / "guides.csv", index=False)
    return root


def test_a01_doctor_data_diagnoses_missing_and_empty(tmp_path: Path):
    missing = doctor_data(dataset="sciplex3", data_root=tmp_path / "nope")
    assert missing["status"] == "missing_directory"
    empty = tmp_path / "root"
    (empty / "raw" / "sciplex3" / "sciplex3_v1").mkdir(parents=True)
    report = doctor_data(dataset="sciplex3", data_root=empty)
    assert report["status"] in {"missing_source_files", "not_prepared"}
    (empty / "prepared" / "sciplex3" / "v1").mkdir(parents=True)
    again = doctor_data(dataset="sciplex3", data_root=empty)
    assert again["status"] == "not_prepared"


def test_a02_fetch_hash_and_offline(tmp_path: Path, monkeypatch):
    src = tmp_path / "remote"
    src.mkdir()
    payload = b"hello-source"
    (src / "cells.csv").write_bytes(payload)
    digest = sha256_file(src / "cells.csv")
    manifest = tmp_path / "man.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "sciplex3",
                "accession": "GSM4150378",
                "release_id": "sciplex3_v1",
                "required_files": ["matrix.h5ad"],
                "raw_sources": [
                    {
                        "name": "cells.csv",
                        "role": "cell_metadata",
                        "url": (src / "cells.csv").as_uri(),
                        "sha256": digest,
                    }
                ],
            }
        )
    )
    root = tmp_path / "data"
    report = fetch_dataset("sciplex3", data_root=root, manifest_path=manifest)
    assert report.status == "ok"
    dest = Path(report.dest) / "cells.csv"
    assert dest.read_bytes() == payload
    reused = fetch_dataset("sciplex3", data_root=root, manifest_path=manifest, offline=True)
    assert reused.status == "ok"
    dest.write_bytes(b"truncated")
    with pytest.raises(IntegrityError):
        fetch_dataset("sciplex3", data_root=root, manifest_path=manifest)


def test_a03_prepare_author_format_slices(tmp_path: Path):
    sci = prepare_sciplex(_write_author_sciplex(tmp_path / "sci_raw"), tmp_path / "sci_prep")
    assert sci["status"] == "ok"
    assert (tmp_path / "sci_prep" / "provenance.json").exists()
    cond = pd.read_parquet(tmp_path / "sci_prep" / "conditions.parquet")
    assert set(cond["perturbation"]) >= {"DMSO", "DrugA", "DrugB"}
    nor = prepare_norman(_write_author_norman(tmp_path / "nor_raw"), tmp_path / "nor_prep")
    assert nor["qc"]["n_pairs"] >= 1
    assert nor["qc"]["n_singles"] >= 2
    assert resolve_norman_identity("TF1+NTC") == (PERTURBATION_GENETIC_SINGLE, ("TF1",))


def test_a04_layer_must_be_chosen(tmp_path: Path):
    import anndata as ad
    from pertbench_long.data.adapters.sciplex import import_sciplex

    obs = pd.DataFrame(
        {
            "cell_type": ["A549", "A549"],
            "perturbation": ["DMSO", "DrugA"],
            "dose": [np.nan, 1.0],
            "dose_unit": ["", "uM"],
            "time": [24.0, 24.0],
            "time_unit": ["h", "h"],
            "is_control": [True, False],
        },
        index=["c0", "c1"],
    )
    adata = ad.AnnData(X=np.array([[0.1, 0.2], [0.3, 0.4]]), obs=obs, var=pd.DataFrame(index=["G1", "G2"]))
    adata.layers["counts"] = np.array([[2.0, 0.0], [3.0, 1.0]])
    adata.write_h5ad(tmp_path / "matrix.h5ad")
    with pytest.raises(SchemaError, match="layers"):
        import_sciplex(tmp_path, declared_matrix_kind="counts")
    loaded = import_sciplex(tmp_path, declared_matrix_kind="counts", matrix_layer="counts", require_exact_dose=True)
    assert float(np.max(loaded.matrix if not hasattr(loaded.matrix, "toarray") else loaded.matrix.toarray())) >= 2.0


def test_a05_undeclared_kind_and_missing_unit(tmp_path: Path):
    import anndata as ad
    from pertbench_long.data.adapters.sciplex import import_sciplex

    obs = pd.DataFrame(
        {"cell_type": ["A549"], "perturbation": ["DrugA"], "dose": [1.0], "is_control": [False]},
        index=["c0"],
    )
    ad.AnnData(X=np.array([[1.5, 2.2]]), obs=obs, var=pd.DataFrame(index=["G1", "G2"])).write_h5ad(tmp_path / "matrix.h5ad")
    with pytest.raises(SchemaError, match="matrix_kind"):
        import_sciplex(tmp_path)
    with pytest.raises(UnknownUnit):
        import_sciplex(tmp_path, declared_matrix_kind="counts", require_exact_dose=True)


def test_a06_false_string_is_not_control():
    assert parse_bool("false") is False
    assert parse_bool("true") is True
    assert parse_bool(0) is False
    assert parse_bool(None) is None
    with pytest.raises(SchemaError):
        parse_bool("maybe")


def test_a06_adapter_false_string(tmp_path: Path):
    import anndata as ad
    from pertbench_long.data.adapters.sciplex import import_sciplex

    obs = pd.DataFrame(
        {
            "cell_type": ["A549", "A549"],
            "perturbation": ["DMSO", "DrugA"],
            "dose": [np.nan, 1.0],
            "dose_unit": ["", "uM"],
            "time": [24, 24],
            "time_unit": ["h", "h"],
            "is_control": ["true", "false"],
        },
        index=["a", "b"],
    )
    ad.AnnData(X=np.ones((2, 2)), obs=obs, var=pd.DataFrame(index=["G1", "G2"])).write_h5ad(tmp_path / "matrix.h5ad")
    store = import_sciplex(tmp_path, declared_matrix_kind="counts")
    kinds = {r.perturbation_id: r.perturbation_kind for r in store.records}
    assert kinds["DMSO"] == "control"
    assert kinds["DrugA"] != "control"


def test_a07_time_and_vehicle_controls_are_not_first_match():
    store = build_synthetic_dose_store(n_compounds=2, n_genes=6, n_cells=3)
    extra = []
    from dataclasses import replace

    for rec in list(store.records):
        if rec.perturbation_kind == PERTURBATION_CONTROL:
            extra.append(replace(rec, time=0.0, observation_id=rec.observation_id + "|t0", original_obs_id=rec.original_obs_id + "|t0"))
    records = extra + list(store.records)
    from pertbench_long.schemas.conditions import make_condition_id

    rewritten = []
    for rec in records:
        cid = make_condition_id(
            study=rec.study,
            context_id=rec.resolved_context_id(),
            perturbation_kind=rec.perturbation_kind,
            perturbation_components=rec.perturbation_components,
            dose=rec.dose,
            dose_unit=rec.dose_unit,
            time=rec.time,
            time_unit=rec.time_unit,
            assay=rec.assay,
            require_exact_dose=rec.perturbation_kind != PERTURBATION_CONTROL and rec.dose is not None,
        )
        rewritten.append(replace(rec, condition_id=cid))
    cond = {}
    for i, rec in enumerate(rewritten):
        cond.setdefault(rec.condition_id, []).append(i)
    store.records = rewritten
    store.condition_to_rows = cond
    mapping = match_controls(store)
    treated = [r for r in rewritten if r.perturbation_kind != PERTURBATION_CONTROL][0]
    ctrl = next(r for r in rewritten if r.condition_id == mapping[treated.condition_id])
    assert ctrl.time == treated.time
    reversed_cond = {k: cond[k] for k in reversed(list(cond))}
    store.condition_to_rows = reversed_cond
    mapping2 = match_controls(store)
    assert mapping2[treated.condition_id] == mapping[treated.condition_id]


def test_a08_target_in_controls_fails_build_and_preflight(tmp_path: Path):
    store = build_synthetic_dose_store(n_compounds=4, n_genes=8, n_cells=4)
    roles = assign_dose_roles(store, split_variant="dose_interpolation", context_ids=None, min_candidates=4, min_targets=2, min_cells=2)
    leak_t = roles["T"][0]
    with pytest.raises(SplitLeakError):
        build_episode_from_condition_ids(
            store,
            dest=tmp_path / "leak",
            episode_id="leak",
            protocol=PROTOCOL_CHEMICAL_DOSE,
            observed_condition_ids=roles["O"],
            queryable_condition_ids=roles["Q"],
            target_condition_ids=roles["T"],
            control_condition_ids=roles["C"] + [leak_t],
            control_mapping={**roles["control_mapping"], leak_t: leak_t},
            split_variant="dose_interpolation",
            objective="x",
            synthetic=True,
            study_name="synthetic_sciplex",
            data_release_id="synthetic_dose_v1",
            experimental_budget=2,
            min_candidates=4,
            min_targets=2,
            n_panel=8,
        )


def test_a09_source_identity_leak(tmp_path: Path):
    from pertbench_long.episodes.split import audit_split

    store = build_synthetic_dose_store(n_compounds=4, n_genes=8, n_cells=4)
    roles = assign_dose_roles(store, split_variant="dose_interpolation", context_ids=None, min_candidates=4, min_targets=2, min_cells=2)
    t_cid = roles["T"][0]
    from dataclasses import replace

    t_row = store.condition_to_rows[t_cid][0]
    alias = replace(
        store.records[t_row],
        observation_id=store.records[t_row].observation_id + "|alias",
        condition_id=roles["C"][0],
        perturbation_kind="chemical",
    )
    records = list(store.records) + [alias]
    o_obs = [store.records[i].observation_id for cid in roles["O"] for i in store.condition_to_rows[cid]]
    q_obs = [store.records[i].observation_id for cid in roles["Q"] for i in store.condition_to_rows[cid]]
    t_obs = [store.records[i].observation_id for cid in roles["T"] for i in store.condition_to_rows[cid]]
    c_obs = [store.records[i].observation_id for cid in roles["C"] for i in store.condition_to_rows[cid]] + [alias.observation_id]
    with pytest.raises(SplitLeakError):
        audit_split(
            records,
            observed_obs=o_obs,
            queryable_obs=q_obs,
            target_obs=t_obs,
            control_obs=c_obs,
            observed_conditions=roles["O"],
            queryable_conditions=roles["Q"],
            target_conditions=roles["T"],
            control_conditions=roles["C"],
            protocol=PROTOCOL_CHEMICAL_DOSE,
            control_mapping=roles["control_mapping"],
        )


def test_a10_replicate_equal_weight():
    stim = np.vstack([np.zeros((9, 2)), np.full((1, 2), 10.0)])
    control = np.zeros((10, 2))
    groups_s = ["r1"] * 9 + ["r2"]
    groups_c = ["r1"] * 5 + ["r2"] * 5
    labels = build_lognorm_cellmean_delta_labels(
        gene_ids=["G1", "G2"],
        targets={"t_01": (stim, control)},
        replicates={"t_01": (groups_s, groups_c)},
        estimand="replicate_equal_weight",
    )
    assert labels.frame.loc[labels.frame["gene_id"] == "G1", "reference_effect"].iloc[0] == pytest.approx(5.0)
    cell = build_lognorm_cellmean_delta_labels(
        gene_ids=["G1", "G2"],
        targets={"t_01": (stim, control)},
        estimand="cell_weighted_descriptive",
    )
    assert cell.frame.loc[cell.frame["gene_id"] == "G1", "reference_effect"].iloc[0] == pytest.approx(1.0)


def test_a11_sparse_no_full_densify():
    from scipy import sparse

    dense = np.array([[1.0, 2.0, 0.0], [0.0, 4.0, 1.0]])
    csr = sparse.csr_matrix(dense)
    called = {"full": 0}
    original = csr.toarray

    def spy():
        called["full"] += 1
        return original()

    csr.toarray = spy  # type: ignore[method-assign]
    z = to_effect_space(csr, "counts", full_universe=csr)
    assert called["full"] == 0
    assert np.allclose(np.asarray(z.todense() if hasattr(z, "todense") else z), to_effect_space(dense, "counts", full_universe=dense))


def test_a12_dev_public_cannot_become_eval_target():
    with pytest.raises(SplitLeakError):
        audit_release_cross_contamination(
            [
                {"partition": "dev", "observed_condition_ids": ["c1"], "queryable_condition_ids": [], "target_condition_ids": []},
                {"partition": "eval", "observed_condition_ids": [], "queryable_condition_ids": [], "target_condition_ids": ["c1"]},
            ]
        )
    with pytest.raises(SplitLeakError):
        audit_release_cross_contamination(
            [
                {"partition": None, "observed_condition_ids": ["a"], "queryable_condition_ids": [], "target_condition_ids": ["b"]},
                {"partition": "eval", "observed_condition_ids": [], "queryable_condition_ids": [], "target_condition_ids": ["c"]},
            ]
        )
    ok = audit_release_cross_contamination(
        [
            {"partition": "dev", "observed_condition_ids": ["o1"], "queryable_condition_ids": ["q1"], "target_condition_ids": ["t1"]},
            {"partition": "eval", "observed_condition_ids": ["o2"], "queryable_condition_ids": ["q2"], "target_condition_ids": ["t2"]},
        ]
    )
    assert ok["status"] == "ok"
    shared = audit_release_cross_contamination(
        [
            {"partition": "dev", "observed_condition_ids": ["support"], "queryable_condition_ids": [], "target_condition_ids": ["t1"]},
            {"partition": "eval", "observed_condition_ids": ["support"], "queryable_condition_ids": [], "target_condition_ids": ["t2"]},
        ],
        allow_shared_support=True,
    )
    assert shared["status"] == "ok"


def test_a13_seed_stable_after_reorder():
    store = build_synthetic_pair_store()
    a = assign_pair_roles(store, split_variant="pair_holdout_seen_genes", min_candidates=8, min_targets=4, min_cells=2, seed=7)
    store.condition_to_rows = {k: store.condition_to_rows[k] for k in reversed(list(store.condition_to_rows))}
    b = assign_pair_roles(store, split_variant="pair_holdout_seen_genes", min_candidates=8, min_targets=4, min_cells=2, seed=7)
    assert set(a["T"]) == set(b["T"])
    c = assign_pair_roles(store, split_variant="pair_holdout_seen_genes", min_candidates=8, min_targets=4, min_cells=2, seed=99)
    assert set(c["T"]) != set(a["T"]) or len(a["T"]) < 2


def test_a14_dose_groups_do_not_mix_times():
    store = build_synthetic_dose_store(n_compounds=3, n_genes=6, n_cells=3)
    from dataclasses import replace
    from pertbench_long.schemas.conditions import make_condition_id

    extra = []
    for rec in store.records:
        if rec.perturbation_id == "Cpd01" and rec.dose == 10.0:
            cid = make_condition_id(
                study=rec.study,
                context_id=rec.context_id,
                perturbation_kind=rec.perturbation_kind,
                perturbation_components=rec.perturbation_components,
                dose=rec.dose,
                dose_unit=rec.dose_unit,
                time=0.0,
                time_unit="h",
                assay=rec.assay,
                require_exact_dose=True,
            )
            extra.append(replace(rec, time=0.0, condition_id=cid, observation_id=rec.observation_id + "|0h"))
    store.records.extend(extra)
    for i, rec in enumerate(store.records):
        store.condition_to_rows.setdefault(rec.condition_id, [])
        if i not in store.condition_to_rows[rec.condition_id]:
            store.condition_to_rows[rec.condition_id].append(i)
    roles = assign_dose_roles(store, split_variant="dose_interpolation", context_ids=None, min_candidates=3, min_targets=2, min_cells=2)
    by_cid = {r.condition_id: r for r in store.records}
    for tid in roles["T"]:
        rec = by_cid[tid]
        visible = [by_cid[cid] for cid in roles["O"] + roles["Q"] if by_cid[cid].perturbation_id == rec.perturbation_id]
        assert all(v.time == rec.time for v in visible)


def test_a15_unknown_builder_kwargs_rejected(tmp_path: Path):
    with pytest.raises(ConfigError):
        build_synthetic_dose_episode(tmp_path / "x", n_panel=8, not_a_real_flag=1)


def test_a16_local_unreleased_is_not_official(tmp_path: Path):
    store = build_synthetic_dose_store(n_compounds=6, n_genes=8, n_cells=4)
    built = build_chemical_dose_episode(
        store,
        dest=tmp_path / "realish",
        episode_id="local",
        synthetic=False,
        data_release_id="local_unreleased",
        n_panel=8,
        min_candidates=8,
        min_targets=3,
        min_cells=2,
        experimental_budget=2,
        partition="pilot",
        requested_track="pilot",
    )
    pub = validate_public_payload(json.loads(Path(built["public_dir"], "episode.json").read_text()))
    priv = validate_private_payload(json.loads(Path(built["private_manifest"]).read_text()))
    assert pub.readiness != "official"
    assert priv.scoring_config["scoring_track"] != "official"
    with pytest.raises(UnsupportedProfile):
        build_chemical_dose_episode(
            store,
            dest=tmp_path / "off",
            episode_id="off",
            synthetic=False,
            data_release_id="local_unreleased",
            n_panel=8,
            min_candidates=8,
            min_targets=3,
            min_cells=2,
            experimental_budget=2,
            partition="pilot",
            requested_track="official",
        )


def test_a17_public_artifacts_copied_and_hashed(tmp_path: Path):
    built = build_synthetic_dose_episode(tmp_path / "dose", n_panel=8)
    public = Path(built["public_dir"])
    for name in ("task.md", "conditions.parquet", "targets.parquet", "gene_panel.tsv", "reference_mapping.json"):
        assert (public / name).exists()
    spec = validate_public_payload(json.loads((public / "episode.json").read_text()))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    copied = copy_declared_public_inputs(public, workspace, spec)
    assert "task.md" in copied
    assert "reference_mapping.json" in copied
    (public / "task.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        check_release(public, Path(built["private_manifest"]))


def test_a18_baseline_uses_public_mapping(tmp_path: Path):
    built = build_synthetic_dose_episode(tmp_path / "dose", n_panel=8, experimental_budget=2)
    public = Path(built["public_dir"])
    mapping = json.loads((public / "reference_mapping.json").read_text())["mapping"]
    assert mapping
    runner = EpisodeRunner(
        public_dir=public,
        private_manifest=Path(built["private_manifest"]),
        run_root=tmp_path / "run",
        agent="nearest_dose_no_query",
        agent_kwargs={},
    )
    result = runner.run()
    assert result["outcome"] == "completed"


def test_a19_full_budget_loops(tmp_path: Path):
    cases = [
        (build_synthetic_dose_episode, 8, "dose"),
        (build_synthetic_pair_episode, 8, "pair"),
        (build_synthetic_context_episode, 16, "ctx"),
    ]
    for builder, budget, name in cases:
        built = builder(tmp_path / name, n_panel=8)
        pub = validate_public_payload(json.loads(Path(built["public_dir"], "episode.json").read_text()))
        assert pub.experimental_budget == budget
        run = EpisodeRunner(
            public_dir=Path(built["public_dir"]),
            private_manifest=Path(built["private_manifest"]),
            run_root=tmp_path / f"run_{name}",
            agent="fixed_order_query",
        ).run()
        assert run["outcome"] == "completed"
        scores = score_run(Path(run["run_dir"]), Path(built["private_manifest"]))
        assert scores["score_status"] == "scored"
        curve = {int(k) for k in scores["scores_curve"]}
        assert curve >= set(range(budget + 1))


def test_a20_live_model_not_run():
    pytest.skip("live local/API model episode is not_run without a configured service or key")
