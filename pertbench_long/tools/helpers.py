"""CPU helpers that only see authorized evidence. Gene IDs are aligned before arithmetic."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pertbench_long.baselines.mapping import effects_to_probabilities
from pertbench_long.errors import SchemaError


def load_h5ad_matrix(path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    import anndata as ad

    adata = ad.read_h5ad(path)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=np.float64)
    return np.asarray(X, dtype=np.float64), adata.obs.copy(), list(adata.var_names)


def align_to_genes(matrix: np.ndarray, source_genes: Sequence[str], target_genes: Sequence[str]) -> np.ndarray:
    index = {g: i for i, g in enumerate(source_genes)}
    missing = [g for g in target_genes if g not in index]
    if missing:
        raise SchemaError(f"gene alignment failed; missing {len(missing)} genes")
    if len(source_genes) != len(set(source_genes)):
        raise SchemaError("duplicate gene IDs in source matrix")
    indexer = [index[g] for g in target_genes]
    return np.asarray(matrix, dtype=np.float64)[:, indexer]


def visible_effect_table(evidence_paths: Sequence[Path], control_path: Path, gene_ids: Sequence[str]) -> pd.DataFrame:
    cX, cobs, cgenes = load_h5ad_matrix(control_path)
    cX = align_to_genes(cX, cgenes, gene_ids)
    rows = []
    for path in evidence_paths:
        X, obs, genes = load_h5ad_matrix(path)
        X = align_to_genes(X, genes, gene_ids)
        for ct, pert in obs[["cell_type", "perturbation_id"]].drop_duplicates().itertuples(index=False):
            stim_mask = (obs["cell_type"] == ct) & (obs["perturbation_id"] == pert)
            ctrl_mask = (cobs["cell_type"] == ct) & (cobs["perturbation_id"] == "Control")
            if stim_mask.sum() == 0 or ctrl_mask.sum() == 0:
                continue
            if pert == "Control":
                continue
            delta = X[np.asarray(stim_mask)].mean(axis=0) - cX[np.asarray(ctrl_mask)].mean(axis=0)
            for gene, value in zip(gene_ids, delta):
                rows.append({"cell_type": ct, "perturbation": pert, "gene_id": gene, "effect": float(value)})
    return pd.DataFrame(rows)


def control_similarity(control_path: Path, target_cell_types: Sequence[str], candidate_cell_types: Sequence[str]) -> list[str]:
    X, obs, genes = load_h5ad_matrix(control_path)
    means = {}
    for ct in obs["cell_type"].astype(str).unique():
        mask = (obs["cell_type"].astype(str) == ct) & (obs["perturbation_id"].astype(str) == "Control")
        if mask.any():
            means[ct] = X[np.asarray(mask)].mean(axis=0)
    available = [ct for ct in target_cell_types if ct in means]
    if not available:
        return []
    target_mean = np.mean([means[ct] for ct in available], axis=0)
    scored = []
    for ct in candidate_cell_types:
        if ct not in means:
            continue
        a = means[ct]
        denom = np.linalg.norm(a) * np.linalg.norm(target_mean)
        sim = float(np.dot(a, target_mean) / denom) if denom else 0.0
        scored.append((sim, ct))
    scored.sort(reverse=True)
    return [ct for _, ct in scored]


def mean_delta_from_effects(effect_table: pd.DataFrame, target_ids: Sequence[str], gene_ids: Sequence[str], sigma: float = 0.15):
    if effect_table.empty:
        effects = np.zeros(len(target_ids) * len(gene_ids))
    else:
        gene_mean = effect_table.groupby("gene_id")["effect"].mean()
        effects = np.array([float(gene_mean.get(g, 0.0)) for _t in target_ids for g in gene_ids], dtype=np.float64)
    probs = effects_to_probabilities(effects, sigma=sigma)
    return effects, probs
