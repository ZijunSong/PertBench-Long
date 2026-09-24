"""Prepare-time matrix IO. Full-matrix toarray() is refused."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pertbench_long.data.sparse_ops import as_csr, is_sparse, refuse_full_densify
from pertbench_long.errors import SchemaError

COUNTS_KINDS = frozenset({"", "counts"})
REFUSED_KINDS = frozenset({"scaled", "log1p", "normalized", "unknown"})


def _values_look_like_counts(matrix: Any) -> None:
    if is_sparse(matrix):
        refuse_full_densify(matrix)
        data = np.asarray(as_csr(matrix).data, dtype=np.float64)
    else:
        data = np.asarray(matrix, dtype=np.float64).ravel()
    if data.size == 0:
        return
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise SchemaError("counts prepare refuses NaN/Inf or negative values; scaled/lognorm inputs are not relabeled as counts")


def read_expression(path: Path, *, layer: str | None = None) -> tuple[Any, list[str], list[str], str]:
    """Return matrix, obs ids, genes, and the declared matrix kind. Sparse stays sparse."""
    if path.suffix.lower() == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        declared = str((adata.uns or {}).get("matrix_kind") or "")
        if declared in REFUSED_KINDS:
            raise SchemaError(
                f"prepare refuses matrix_kind={declared!r}; it is not rewritten as counts"
            )
        if declared not in COUNTS_KINDS:
            raise SchemaError(f"unsupported prepare matrix_kind {declared!r}")
        layer_names = [str(k) for k in getattr(adata, "layers", {}) if k is not None]
        if layer:
            chosen = layer
        elif declared == "counts" and "counts" in adata.layers:
            chosen = "counts"
        elif layer_names and "counts" not in layer_names:
            raise SchemaError(f"h5ad layers {layer_names} need an explicit counts layer; X is not assumed to be counts")
        elif "counts" in layer_names:
            chosen = "counts"
        else:
            chosen = "X"
        matrix = adata.X if chosen == "X" else adata.layers[chosen]
        if is_sparse(matrix):
            matrix = as_csr(matrix)
        _values_look_like_counts(matrix)
        return matrix, [str(i) for i in adata.obs_names], [str(g) for g in adata.var_names], "counts"
    if path.suffix.lower() == ".mtx":
        from scipy import io as spio
        from scipy import sparse

        matrix = sparse.csr_matrix(spio.mmread(path))
        _values_look_like_counts(matrix)
        return matrix, [], [], "counts"
    frame = __import__("pandas").read_csv(path, index_col=0)
    matrix = frame.to_numpy(dtype=np.float64)
    _values_look_like_counts(matrix)
    return matrix, [str(i) for i in frame.index], [str(c) for c in frame.columns], "counts"


def take_rows(matrix: Any, row_index: list[int]) -> Any:
    if not row_index:
        raise SchemaError("no rows selected")
    if is_sparse(matrix):
        refuse_full_densify(matrix)
        return as_csr(matrix)[row_index]
    return np.asarray(matrix, dtype=np.float64)[row_index]


def align_barcodes(obs_ids: list[str], cell_index: dict[str, int]) -> tuple[list[str], list[int], list[str]]:
    """One pass. Does not call list.index."""
    keep: list[str] = []
    rows: list[int] = []
    missing: list[str] = []
    for i, oid in enumerate(obs_ids):
        if oid in cell_index:
            keep.append(oid)
            rows.append(i)
        else:
            missing.append(oid)
    return keep, rows, missing


def publish_directory(staging: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
