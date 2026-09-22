"""Rebuild public AnnData from a whitelist. Copy-then-delete is not used."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

FORBIDDEN_UNS_SUBSTRINGS = ("de_table", "target_hvg", "heldout", "answer", "moa_hidden")
FORBIDDEN_OBS_COLS = ("split_answer", "hidden_label", "target_marker")


def _new_anndata(X: np.ndarray, obs: pd.DataFrame, var: pd.DataFrame, uns: Mapping[str, Any] | None = None):
    import anndata as ad

    adata = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs.copy(), var=var.copy())
    adata.uns = dict(uns or {})
    return adata


def scrub_uns(uns: Mapping[str, Any], *, extra_forbidden: Sequence[str] = ()) -> dict[str, Any]:
    allowed = {}
    forbidden = tuple(FORBIDDEN_UNS_SUBSTRINGS) + tuple(extra_forbidden)
    for key, value in uns.items():
        lowered = str(key).lower()
        if any(tok in lowered for tok in forbidden):
            continue
        allowed[key] = value
    return allowed


def public_obs_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    keep = [
        "observation_id",
        "study",
        "species",
        "cell_type",
        "perturbation_id",
        "dose",
        "dose_unit",
        "time",
        "time_unit",
        "assay",
        "original_obs_id",
        "sample_id",
        "donor",
        "context_id",
        "context_type",
        "condition_id",
        "perturbation_kind",
        "batch_id",
        "replicate_id",
        "control_group_id",
    ]
    rows = []
    for rec in records:
        rows.append({k: rec.get(k) for k in keep})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame.index = [str(r["observation_id"]) for r in rows]
    for col in ("dose", "time"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: "" if v is None else str(v))
    return frame


def export_public_bundle(
    *,
    matrix: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    gene_ids: Sequence[str],
    uns: Mapping[str, Any] | None = None,
    dest: Path,
    artifact_id: str,
) -> Path:
    """Construct a new AnnData from arrays. Hidden layers/raw/graphs are not copied."""
    obs = public_obs_frame(records)
    var = pd.DataFrame(index=list(gene_ids))
    var["gene_id"] = list(gene_ids)
    clean_uns = scrub_uns(uns or {}, extra_forbidden=("TARGET_MARKER", "target_marker"))
    # Never attach .raw or extra layers.
    adata = _new_anndata(np.asarray(matrix, dtype=np.float32), obs, var, clean_uns)
    dest.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(dest)
    _assert_no_marker(dest)
    return dest


def _assert_no_marker(path: Path, marker: str = "TARGET_MARKER_SECRET") -> None:
    data = path.read_bytes()
    if marker.encode("utf-8") in data:
        raise RuntimeError("public export contains hidden marker bytes")
    # Also scan as text-ish
    if marker.encode("utf-8").lower() in data.lower():
        raise RuntimeError("public export contains hidden marker bytes")


def contains_marker_bytes(path: Path, marker: str) -> bool:
    return marker.encode("utf-8") in path.read_bytes()
