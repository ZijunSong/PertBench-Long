"""Measurement protocol. Hidden Q/T values must not affect the initial public transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pertbench_long.errors import UnsupportedProfile
from pertbench_long.hashes import gene_order_hash, sha256_bytes


LOG1P_APPLIED_FLAG = "log1p_applied_v1"
COUNTS_NORM_TARGET = 10000.0


@dataclass(frozen=True)
class TransformProfile:
    name: str
    matrix_kind_in: str
    effect_unit: str
    log1p_applied: bool


def profile_for_kind(matrix_kind: str) -> TransformProfile:
    if matrix_kind == "counts":
        return TransformProfile("effect_proxy_v1_from_counts", "counts", "log1p_mean_diff", True)
    if matrix_kind == "log1p":
        return TransformProfile("effect_proxy_v1_existing_log1p", "log1p", "log1p_mean_diff", True)
    if matrix_kind in {"scaled", "unknown"}:
        raise UnsupportedProfile(
            f"matrix_kind={matrix_kind} cannot use effect_proxy_v1; declare an independent profile"
        )
    if matrix_kind == "normalized":
        return TransformProfile("effect_proxy_v1_from_normalized", "normalized", "log1p_mean_diff", True)
    raise UnsupportedProfile(f"unsupported matrix_kind={matrix_kind}")


def library_sizes(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).sum(axis=1)


def normalize_counts_then_log1p(matrix: np.ndarray, *, full_universe: np.ndarray) -> np.ndarray:
    """Normalize using the full assay gene universe, then log1p. Output G is a column subset later."""
    totals = np.asarray(full_universe, dtype=np.float64).sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1e-12)
    normed = np.asarray(matrix, dtype=np.float64) * (COUNTS_NORM_TARGET / totals)
    return np.log1p(normed)


def to_effect_space(matrix: np.ndarray, matrix_kind: str, *, full_universe: np.ndarray | None = None) -> np.ndarray:
    profile = profile_for_kind(matrix_kind)
    data = np.asarray(matrix, dtype=np.float64)
    if matrix_kind == "counts":
        universe = full_universe if full_universe is not None else data
        return normalize_counts_then_log1p(data, full_universe=universe)
    if matrix_kind == "log1p":
        return data
    if matrix_kind == "normalized":
        return np.log1p(np.clip(data, 0, None))
    raise UnsupportedProfile(profile.name)


def detect_repeated_log1p(matrix: np.ndarray, matrix_kind: str) -> bool:
    """Heuristic protocol check: applying log1p again to already-log1p data changes the distribution."""
    data = np.asarray(matrix, dtype=np.float64)
    if matrix_kind != "log1p":
        return False
    again = np.log1p(np.clip(data, 0, None))
    return not np.allclose(data, again)


def select_panel_from_visible(
    matrix: np.ndarray,
    gene_ids: Sequence[str],
    visible_rows: Sequence[int],
    *,
    n_genes: int,
) -> list[str]:
    """Episode-specific panel from O+C only. Must not use Q/T rows."""
    if not visible_rows:
        return list(gene_ids)[:n_genes]
    subset = np.asarray(matrix, dtype=np.float64)[list(visible_rows)]
    var = subset.var(axis=0)
    order = np.argsort(-var)
    n = min(int(n_genes), len(gene_ids))
    chosen = [gene_ids[i] for i in order[:n]]
    chosen_sorted_by_original = [g for g in gene_ids if g in set(chosen)]
    return chosen_sorted_by_original


def public_export_fingerprint(gene_ids: Sequence[str], public_matrix: np.ndarray) -> str:
    payload = gene_order_hash(gene_ids).encode() + np.asarray(public_matrix, dtype=np.float64).tobytes()
    return sha256_bytes(payload)
