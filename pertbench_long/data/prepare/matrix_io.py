"""Prepare-time matrix IO. Full-matrix toarray() is refused."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pertbench_long.data.sparse_ops import as_csr, is_sparse, refuse_full_densify
from pertbench_long.errors import SchemaError

COUNTS_KINDS = frozenset({"counts"})
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


def _kind_for_selection(uns: dict, chosen: str, *, layer: str | None, declared_kind: str | None) -> str:
    layer_kinds = dict(uns.get("layer_kinds") or {})
    if chosen == "X":
        kind = str(uns.get("matrix_kind") or "")
    else:
        kind = str(layer_kinds.get(chosen) or "")
    if layer and layer == chosen and declared_kind and not kind:
        kind = declared_kind
    if chosen == "X" and not kind and declared_kind and layer in {None, "X"}:
        kind = declared_kind
    if not kind:
        raise SchemaError(
            "h5ad matrix kind is undeclared; non-negative values are not treated as counts"
        )
    if kind in REFUSED_KINDS or kind not in COUNTS_KINDS:
        raise SchemaError(f"prepare refuses matrix_kind={kind!r}; it is not rewritten as counts")
    return kind


def read_expression(
    path: Path,
    *,
    layer: str | None = None,
    declared_kind: str | None = None,
) -> tuple[Any, list[str], list[str], str]:
    """Return matrix, obs ids, genes, and the declared matrix kind. Sparse stays sparse."""
    if path.suffix.lower() == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        uns = dict(adata.uns or {})
        layer_names = [str(k) for k in getattr(adata, "layers", {}) if k is not None]
        x_kind = str(uns.get("matrix_kind") or "")
        if not layer and x_kind in REFUSED_KINDS and "counts" not in layer_names:
            raise SchemaError(f"prepare refuses matrix_kind={x_kind!r}; it is not rewritten as counts")
        if layer:
            if layer != "X" and layer not in adata.layers:
                raise SchemaError(f"requested layer {layer!r} is not in {layer_names}")
            chosen = layer
        elif "counts" in layer_names and str(dict(uns.get("layer_kinds") or {}).get("counts") or declared_kind) == "counts":
            chosen = "counts"
        elif not layer_names and (uns.get("matrix_kind") == "counts" or declared_kind == "counts"):
            chosen = "X"
        elif not uns.get("matrix_kind") and not declared_kind:
            raise SchemaError("h5ad matrix kind is undeclared; non-negative values are not treated as counts")
        else:
            raise SchemaError("h5ad has no declared counts matrix; set layer and matrix_kind explicitly")
        kind = _kind_for_selection(uns, chosen, layer=layer, declared_kind=declared_kind)
        matrix = adata.X if chosen == "X" else adata.layers[chosen]
        if is_sparse(matrix):
            matrix = as_csr(matrix)
        _values_look_like_counts(matrix)
        return matrix, [str(i) for i in adata.obs_names], [str(g) for g in adata.var_names], kind
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


def _read_text_lines(path: Path) -> list[str]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _first_existing(source: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = source / name
        if path.exists():
            return path
    raise SchemaError(f"missing one of {list(names)} in {source}")


def read_mex_bundle(source: Path) -> tuple[Any, list[str], list[str], str, list[Path]]:
    """Cell Ranger MEX is features × barcodes. AnnData needs cells × genes."""
    from scipy import io as spio
    from scipy import sparse

    matrix_path = _first_existing(source, ("matrix.mtx", "matrix.mtx.gz"))
    barcodes_path = _first_existing(source, ("barcodes.tsv", "barcodes.tsv.gz"))
    genes_path = _first_existing(source, ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"))
    barcodes = [line.split("\t")[0] for line in _read_text_lines(barcodes_path)]
    genes = [line.split("\t")[0] for line in _read_text_lines(genes_path)]
    matrix = sparse.csr_matrix(spio.mmread(str(matrix_path)))
    if tuple(matrix.shape) != (len(genes), len(barcodes)):
        raise SchemaError(
            f"MEX shape {tuple(matrix.shape)} is not (n_features={len(genes)}, n_barcodes={len(barcodes)}); "
            "direction is not guessed"
        )
    matrix = matrix.T.tocsr()
    if tuple(matrix.shape) != (len(barcodes), len(genes)):
        raise SchemaError(f"transposed MEX shape {tuple(matrix.shape)} is not (n_cells, n_genes)")
    _values_look_like_counts(matrix)
    return matrix, barcodes, genes, "counts", [matrix_path, barcodes_path, genes_path]


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
