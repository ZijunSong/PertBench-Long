"""Synthetic long-horizon fixtures. Marked synthetic; not scientific evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from pertbench_long.data.adapter import CanonicalStore, ImportSummary
from pertbench_long.hashes import gene_order_hash
from pertbench_long.schemas.conditions import (
    CONTEXT_CELL_LINE,
    PERTURBATION_CHEMICAL,
    PERTURBATION_CONTROL,
    PERTURBATION_GENETIC_PAIR,
    PERTURBATION_GENETIC_SINGLE,
    make_condition_id,
    normalize_components,
)
from pertbench_long.schemas.types import ExperimentRecord


def _record(**kwargs) -> ExperimentRecord:
    rec = ExperimentRecord(**kwargs)
    return rec.__class__(**{**asdict(rec), "condition_id": rec.condition_id or rec.condition_key()})


def _store(records, matrix, genes, notes) -> CanonicalStore:
    cond: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        cond.setdefault(rec.condition_id, []).append(i)
    summary = ImportSummary(
        n_source_rows=matrix.shape[0],
        n_kept=matrix.shape[0],
        matrix_kind="counts",
        gene_order_hash=gene_order_hash(genes),
        notes=["SYNTHETIC fixture; not for scientific conclusions", *notes],
    )
    return CanonicalStore(matrix=matrix, records=records, gene_ids=list(genes), summary=summary, condition_to_rows=cond)


def build_synthetic_dose_store(
    *,
    n_cells: int = 6,
    n_genes: int = 32,
    n_compounds: int = 12,
    doses: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
    context_id: str = "A549",
    seed: int = 1,
) -> CanonicalStore:
    rng = np.random.default_rng(seed)
    genes = [f"G{i+1}" for i in range(n_genes)]
    records = []
    rows = []
    compounds = [f"Cpd{i+1:02d}" for i in range(n_compounds)]

    def add(pert, kind, components, dose, unit, base):
        cid = make_condition_id(
            study="synthetic_sciplex",
            context_id=context_id,
            perturbation_kind=kind,
            perturbation_components=components,
            dose=dose,
            dose_unit=unit,
            assay="scrna",
            require_exact_dose=kind == PERTURBATION_CHEMICAL,
        )
        noise = rng.integers(0, 4, size=(n_cells, n_genes))
        block = np.clip(np.round(base.reshape(1, -1) + noise), 0, None).astype(np.float64)
        rows.append(block)
        for i in range(n_cells):
            records.append(
                _record(
                    observation_id=f"synthetic_sciplex|{context_id}|{pert}|{dose}|{i}",
                    study="synthetic_sciplex",
                    species="synthetic",
                    cell_type=context_id,
                    perturbation_id="Control" if kind == PERTURBATION_CONTROL else pert,
                    dose=dose,
                    dose_unit=unit,
                    time=24.0,
                    time_unit="h",
                    assay="scrna",
                    original_obs_id=f"{pert}-{i}",
                    sample_id=context_id,
                    condition_id=cid,
                    source_file="synthetic://dose",
                    matrix_kind="counts",
                    context_id=context_id,
                    context_type=CONTEXT_CELL_LINE,
                    perturbation_kind=kind,
                    perturbation_components=tuple(components),
                    identity_version="cid_v1",
                )
            )

    add("vehicle", PERTURBATION_CONTROL, ("control",), None, "none", np.full(n_genes, 8.0))
    for j, cpd in enumerate(compounds):
        slope = 0.4 + 0.05 * j
        for dose in doses:
            base = np.full(n_genes, 8.0)
            base[0] += slope * np.log10(dose)
            base[1] += 0.2 * j
            add(cpd, PERTURBATION_CHEMICAL, (cpd,), dose, "nM", base)
    return _store(records, np.vstack(rows), genes, ["synthetic chemical-dose condition table"])


def build_synthetic_pair_store(
    *,
    n_cells: int = 6,
    n_genes: int = 24,
    gene_names: Sequence[str] | None = None,
    extra_pairs: int = 20,
    seed: int = 2,
) -> CanonicalStore:
    rng = np.random.default_rng(seed)
    pert_genes = list(gene_names or [f"TF{i+1}" for i in range(10)])
    genes = [f"G{i+1}" for i in range(n_genes)]
    records = []
    rows = []
    context_id = "K562"

    def add(pert, kind, components, base):
        cid = make_condition_id(
            study="synthetic_norman",
            context_id=context_id,
            perturbation_kind=kind,
            perturbation_components=components,
            assay="scrna",
        )
        noise = rng.integers(0, 3, size=(n_cells, n_genes))
        block = np.clip(np.round(base.reshape(1, -1) + noise), 0, None).astype(np.float64)
        rows.append(block)
        for i in range(n_cells):
            records.append(
                _record(
                    observation_id=f"synthetic_norman|{pert}|{i}",
                    study="synthetic_norman",
                    species="synthetic",
                    cell_type=context_id,
                    perturbation_id=pert,
                    dose=None,
                    dose_unit="none",
                    time=None,
                    time_unit="none",
                    assay="scrna",
                    original_obs_id=f"{pert}-{i}",
                    sample_id=context_id,
                    condition_id=cid,
                    source_file="synthetic://pair",
                    matrix_kind="counts",
                    context_id=context_id,
                    context_type=CONTEXT_CELL_LINE,
                    perturbation_kind=kind,
                    perturbation_components=tuple(components),
                    identity_version="cid_v1",
                )
            )

    add("Control", PERTURBATION_CONTROL, ("control",), np.full(n_genes, 10.0))
    single_effect = {g: rng.normal(1.5, 0.2, size=n_genes) for g in pert_genes}
    for g in pert_genes:
        add(g, PERTURBATION_GENETIC_SINGLE, (g,), np.full(n_genes, 10.0) + np.clip(single_effect[g], 0, None))
    pairs = []
    for i, a in enumerate(pert_genes):
        for b in pert_genes[i + 1 :]:
            pairs.append((a, b))
    for a, b in pairs[: extra_pairs + 12]:
        comps = normalize_components((a, b), kind=PERTURBATION_GENETIC_PAIR)
        base = np.full(n_genes, 10.0) + np.clip(single_effect[a] + single_effect[b], 0, None)
        base[0] += 1.0  # non-additive residual on gene 0
        add("+".join(comps), PERTURBATION_GENETIC_PAIR, comps, base)
    return _store(records, np.vstack(rows), genes, ["synthetic Norman-like CRISPRa pairs"])


def build_synthetic_context_store(
    *,
    n_cells: int = 5,
    n_genes: int = 24,
    n_compounds: int = 30,
    contexts: Sequence[str] = ("A549", "K562", "MCF7"),
    seed: int = 3,
) -> CanonicalStore:
    rng = np.random.default_rng(seed)
    genes = [f"G{i+1}" for i in range(n_genes)]
    compounds = [f"Cpd{i+1:02d}" for i in range(n_compounds)]
    records = []
    rows = []

    def add(ctx, pert, kind, components, dose, base):
        cid = make_condition_id(
            study="synthetic_sciplex_ctx",
            context_id=ctx,
            perturbation_kind=kind,
            perturbation_components=components,
            dose=dose,
            dose_unit="none" if dose is None else "uM",
            assay="scrna",
            require_exact_dose=kind == PERTURBATION_CHEMICAL,
        )
        noise = rng.integers(0, 3, size=(n_cells, n_genes))
        block = np.clip(np.round(base.reshape(1, -1) + noise), 0, None).astype(np.float64)
        rows.append(block)
        for i in range(n_cells):
            records.append(
                _record(
                    observation_id=f"synthetic_ctx|{ctx}|{pert}|{i}",
                    study="synthetic_sciplex_ctx",
                    species="synthetic",
                    cell_type=ctx,
                    perturbation_id="Control" if kind == PERTURBATION_CONTROL else pert,
                    dose=dose,
                    dose_unit="none" if dose is None else "uM",
                    time=24.0,
                    time_unit="h",
                    assay="scrna",
                    original_obs_id=f"{ctx}-{pert}-{i}",
                    sample_id=ctx,
                    condition_id=cid,
                    source_file="synthetic://context",
                    matrix_kind="counts",
                    context_id=ctx,
                    context_type=CONTEXT_CELL_LINE,
                    perturbation_kind=kind,
                    perturbation_components=tuple(components),
                    identity_version="cid_v1",
                )
            )

    for ctx_i, ctx in enumerate(contexts):
        add(ctx, "vehicle", PERTURBATION_CONTROL, ("control",), None, np.full(n_genes, 7.0 + ctx_i))
        for j, cpd in enumerate(compounds):
            base = np.full(n_genes, 7.0 + ctx_i)
            base[0] += 0.8 + 0.1 * ctx_i + 0.03 * j
            add(ctx, cpd, PERTURBATION_CHEMICAL, (cpd,), 1.0, base)
    return _store(records, np.vstack(rows), genes, ["synthetic multi-context campaign"])


_SYNTH_KEYS = {
    "episode_id",
    "n_panel",
    "experimental_budget",
    "min_candidates",
    "min_targets",
    "a_family",
    "seed",
    "split_variant",
    "min_cells",
    "resource_profile",
    "context_ids",
    "source_context",
    "target_contexts",
}


def _synth_kwargs(kwargs: dict) -> dict:
    from pertbench_long.errors import ConfigError

    unknown = set(kwargs) - _SYNTH_KEYS
    if unknown:
        raise ConfigError(f"unknown builder arguments: {sorted(unknown)}")
    return kwargs


def build_synthetic_dose_episode(dest: Path, **kwargs):
    from pertbench_long.episodes.builders.chemical_dose import build_chemical_dose_episode

    kwargs = _synth_kwargs(kwargs)
    return build_chemical_dose_episode(
        build_synthetic_dose_store(),
        dest=dest,
        episode_id=kwargs.pop("episode_id", "sciplex_dose_synth_0001"),
        synthetic=True,
        study_name="synthetic_sciplex",
        data_release_id="synthetic_dose_v1",
        **kwargs,
    )


def build_synthetic_pair_episode(dest: Path, **kwargs):
    from pertbench_long.episodes.builders.genetic_pair import build_genetic_pair_episode

    kwargs = _synth_kwargs(kwargs)
    return build_genetic_pair_episode(
        build_synthetic_pair_store(),
        dest=dest,
        episode_id=kwargs.pop("episode_id", "norman_pair_synth_0001"),
        synthetic=True,
        study_name="synthetic_norman",
        data_release_id="synthetic_pair_v1",
        **kwargs,
    )


def build_synthetic_context_episode(dest: Path, **kwargs):
    from pertbench_long.episodes.builders.context_campaign import build_context_campaign_episode

    kwargs = _synth_kwargs(kwargs)
    return build_context_campaign_episode(
        build_synthetic_context_store(),
        dest=dest,
        episode_id=kwargs.pop("episode_id", "sciplex_context_synth_0001"),
        synthetic=True,
        study_name="synthetic_sciplex_ctx",
        data_release_id="synthetic_context_v1",
        **kwargs,
    )
