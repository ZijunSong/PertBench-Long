"""Sparse-aware row/column extraction. Full-matrix .toarray() is not used."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def is_sparse(matrix: Any) -> bool:
    return hasattr(matrix, "toarray") and not isinstance(matrix, np.ndarray)


def as_csr(matrix: Any):
    if not is_sparse(matrix):
        return None
    try:
        from scipy import sparse

        if sparse.issparse(matrix):
            return matrix.tocsr()
    except Exception:
        return matrix
    return matrix


def library_sizes(matrix: Any) -> np.ndarray:
    if is_sparse(matrix):
        csr = as_csr(matrix)
        totals = np.asarray(csr.sum(axis=1), dtype=np.float64).reshape(-1)
        return totals
    return np.asarray(matrix, dtype=np.float64).sum(axis=1)


def n_rows(matrix: Any) -> int:
    return int(matrix.shape[0])


def n_cols(matrix: Any) -> int:
    return int(matrix.shape[1])


def extract_dense_block(matrix: Any, row_index: Sequence[int], col_index: Sequence[int] | None = None) -> np.ndarray:
    """Materialize only the requested rows/columns."""
    rows = np.asarray(list(row_index), dtype=np.int64)
    if rows.size == 0:
        cols = n_cols(matrix) if col_index is None else len(col_index)
        return np.zeros((0, cols), dtype=np.float64)
    if is_sparse(matrix):
        csr = as_csr(matrix)
        block = csr[rows]
        if col_index is not None:
            block = block[:, list(col_index)]
        return np.asarray(block.toarray(), dtype=np.float64)
    data = np.asarray(matrix)
    block = data[rows]
    if col_index is not None:
        block = block[:, list(col_index)]
    return np.asarray(block, dtype=np.float64)


def to_dense_if_small(matrix: Any, *, max_values: int = 2_000_000) -> np.ndarray:
    size = int(n_rows(matrix) * n_cols(matrix))
    if size > max_values and is_sparse(matrix):
        raise MemoryError(f"refusing to densify a {matrix.shape} sparse matrix ({size} values)")
    if is_sparse(matrix):
        return np.asarray(as_csr(matrix).toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def finite_check_sample(matrix: Any, *, max_check: int = 10000) -> None:
    rows = min(n_rows(matrix), max(1, max_check // max(n_cols(matrix), 1)))
    sample = extract_dense_block(matrix, range(rows))
    if not np.all(np.isfinite(sample)):
        raise ValueError("matrix sample contains NaN/Inf")


def refuse_full_densify(matrix: Any, *, max_values: int = 2_000_000) -> None:
    """Sentinel: a full-matrix toarray() of a large sparse object is an error."""
    if is_sparse(matrix) and int(n_rows(matrix) * n_cols(matrix)) > max_values:
        raise MemoryError(
            f"refusing full densify of sparse matrix shape={tuple(matrix.shape)} "
            f"({int(n_rows(matrix) * n_cols(matrix))} values)"
        )


def scale_and_log1p_counts(matrix: Any, totals: np.ndarray, *, target: float = 10000.0):
    """Library-size scale + log1p without materializing a full dense copy."""
    if np.any(totals <= 0):
        raise ValueError("zero library size is rejected; it is not silently replaced")
    scale = np.asarray(target / totals, dtype=np.float64)
    if is_sparse(matrix):
        refuse_full_densify(matrix)
        csr = as_csr(matrix).astype(np.float64)
        if csr.data.size:
            if not np.all(np.isfinite(csr.data)) or np.any(csr.data < 0):
                raise ValueError("counts matrix contains NaN/Inf or negative values")
        scaled = csr.multiply(scale.reshape(-1, 1)).tocsr()
        scaled.data = np.log1p(np.asarray(scaled.data, dtype=np.float64))
        return scaled
    data = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError("counts matrix contains NaN/Inf or negative values")
    return np.log1p(data * scale.reshape(-1, 1))
