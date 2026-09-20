"""Synthetic fixture used for engineering smoke tests. Outputs are labeled synthetic."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from pertbench_long.data.adapter import CanonicalStore, ExperimentRecord, ImportSummary
from pertbench_long.hashes import gene_order_hash

CELL_TYPES = ["CD4T", "CD8T", "CD14+Mono", "Dendritic", "FCGR3A+Mono", "B", "NK"]
GENES_DEFAULT = ["G1", "G2", "G3", "G4", "G5", "G6"]
O_TYPES = ["CD4T", "CD8T"]
Q_TYPES = ["CD14+Mono", "Dendritic", "FCGR3A+Mono"]
T_TYPES = ["B", "NK"]


def _effect_vector(cell_type: str, n_genes: int) -> np.ndarray:
    """Designed so querying myeloid Q items can change T predictions vs T-only O mean."""
    vec = np.zeros(n_genes, dtype=np.float64)
    vec[0] = 0.8  # shared IFN response
    if cell_type in {"CD4T", "CD8T"}:
        vec[1] = 0.05  # T-specific near-neutral
        vec[2] = 0.02
    if cell_type in {"CD14+Mono", "Dendritic", "FCGR3A+Mono"}:
        vec[1] = 1.2  # myeloid, also true for B/NK
        vec[3] = 0.9
    if cell_type in {"B", "NK"}:
        vec[1] = 1.2
        vec[3] = 0.9
        vec[4] = 0.6 if cell_type == "B" else -0.7
    vec[5] = 0.0
    return vec


def build_synthetic_store(
    *,
    n_cells: int = 8,
    gene_ids: Sequence[str] | None = None,
    seed: int = 0,
    hidden_shift: float = 0.0,
) -> CanonicalStore:
    rng = np.random.default_rng(seed)
    genes = list(gene_ids or GENES_DEFAULT)
    n_genes = len(genes)
    rows = []
    records = []
    for ct in CELL_TYPES:
        for pert in ("Control", "IFN"):
            base = np.zeros(n_genes) if pert == "Control" else _effect_vector(ct, n_genes)
            if pert == "IFN" and ct in T_TYPES:
                base = base + hidden_shift
            noise = rng.normal(0.0, 0.02, size=(n_cells, n_genes))
            block = np.clip(base.reshape(1, -1) + noise, 0, None)
            rows.append(block)
            for i in range(n_cells):
                original = f"{ct}-{i:03d}-{'stimulated' if pert == 'IFN' else 'control'}"
                rec = ExperimentRecord(
                    observation_id=f"synthetic_pbmc_v1|{ct}|{original}",
                    study="synthetic_pbmc_v1",
                    species="synthetic",
                    cell_type=ct,
                    perturbation_id=pert,
                    dose=None,
                    dose_unit="none",
                    time=None,
                    time_unit="none",
                    assay="scrna",
                    original_obs_id=original,
                    sample_id=ct,
                    donor=None,
                    source_file="synthetic://memory",
                    matrix_kind="log1p",
                )
                rec = rec.__class__(**{**asdict(rec), "condition_id": rec.condition_key()})
                records.append(rec)
    matrix = np.vstack(rows)
    summary = ImportSummary(
        n_source_rows=matrix.shape[0],
        n_kept=matrix.shape[0],
        matrix_kind="log1p",
        gene_order_hash=gene_order_hash(genes),
        notes=["SYNTHETIC fixture; not for scientific conclusions"],
    )
    cond: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        cond.setdefault(rec.condition_key(), []).append(i)
    return CanonicalStore(matrix=matrix, records=records, gene_ids=genes, summary=summary, condition_to_rows=cond)
