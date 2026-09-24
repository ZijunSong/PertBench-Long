"""Diagnose DATA_ROOT readiness. Empty directories are not ready."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pertbench_long.data.acquire import REASON_HASH_MISMATCH, REASON_MISSING_DIR, REASON_MISSING_SOURCE, load_source_files
from pertbench_long.data.data_root import prepared_dir, raw_dir, resolve_data_root
from pertbench_long.hashes import sha256_file

STATUS_MISSING_DIRECTORY = "missing_directory"
STATUS_MISSING_SOURCE = "missing_source_files"
STATUS_NOT_PREPARED = "not_prepared"
STATUS_HASH_MISMATCH = "hash_mismatch"
STATUS_MISSING_METADATA = "missing_metadata"
STATUS_INSUFFICIENT = "insufficient_conditions"
STATUS_READY_PILOT = "ready_for_pilot"


def doctor_data(
    *,
    dataset: str,
    data_root: Path | str | None = None,
    manifest_path: Path | str | None = None,
    min_candidates: int = 24,
    min_targets: int = 8,
) -> dict[str, Any]:
    try:
        root = resolve_data_root(data_root)
    except Exception as exc:
        return {"status": STATUS_MISSING_DIRECTORY, "reason": str(exc), "dataset": dataset}
    if not root.exists():
        return {"status": STATUS_MISSING_DIRECTORY, "dataset": dataset, "data_root": str(root)}
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[2] / "data" / "acquisition" / f"{dataset}_v1.json"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {"status": STATUS_MISSING_METADATA, "dataset": dataset, "missing": [str(manifest_path)]}
    ds, release_id, sources, _payload = load_source_files(manifest_path)
    raw = raw_dir(root, ds or dataset, release_id or "v1")
    prepared = prepared_dir(root, ds or dataset, "v1")
    report: dict[str, Any] = {
        "dataset": ds or dataset,
        "data_root": str(root),
        "raw_dir": str(raw),
        "prepared_dir": str(prepared),
    }
    prepared_exists = prepared.exists()
    raw_has_files = raw.exists() and any(p.is_file() for p in raw.rglob("*"))
    if prepared_exists:
        required_prepared = ["matrix.h5ad", "provenance.json", "conditions.parquet", "qc_report.json"]
        missing_prep = [name for name in required_prepared if not (prepared / name).exists()]
        if missing_prep:
            report.update({"status": STATUS_NOT_PREPARED, "missing": missing_prep})
            return report
    elif raw.exists() and not raw_has_files:
        report.update({"status": STATUS_MISSING_SOURCE, "note": "raw directory exists but contains no files"})
        return report
    missing_raw = [s.name for s in sources if s.requirement == "required" and s.name and not (raw / s.name).exists()]
    if sources and missing_raw and not (prepared / "matrix.h5ad").exists():
        report.update({"status": STATUS_MISSING_SOURCE, "missing": missing_raw})
        return report
    for source in sources:
        path = raw / source.name
        if source.sha256 and path.exists() and sha256_file(path) != source.sha256:
            report.update({"status": STATUS_HASH_MISMATCH, "file": source.name, "reason": REASON_HASH_MISMATCH})
            return report
    try:
        provenance = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {**report, "status": STATUS_MISSING_METADATA, "reason": str(exc)}
    if not provenance.get("matrix", {}).get("matrix_kind"):
        return {**report, "status": STATUS_MISSING_METADATA, "reason": "provenance missing matrix_kind"}
    import pandas as pd

    qc_path = prepared / "qc_report.json"
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except Exception:
        return {**report, "status": "invalid_matrix", "reason": "qc_report_unreadable"}
    if not qc or not qc.get("status") or not qc.get("n_cells"):
        return {**report, "status": "empty_qc", "reason": "empty_qc"}
    try:
        import anndata as ad

        adata = ad.read_h5ad(prepared / "matrix.h5ad")
        n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    except Exception as exc:
        return {**report, "status": "invalid_matrix", "reason": "invalid_matrix", "detail": type(exc).__name__}
    if n_obs <= 0 or n_vars <= 0:
        return {**report, "status": "invalid_matrix", "reason": "invalid_matrix"}
    cond = pd.read_parquet(prepared / "conditions.parquet")
    treated = cond
    if "perturbation_kind" in cond.columns:
        treated = cond[cond["perturbation_kind"].astype(str) != "control"]
    n = int(treated.shape[0])
    if n <= 0:
        return {**report, "status": "insufficient_conditions", "reason": "insufficient_eligible_conditions", "n_treated": 0}
    if n < min_candidates + min_targets:
        report.update(
            {
                "status": STATUS_INSUFFICIENT,
                "n_conditions": n,
                "min_candidates": min_candidates,
                "min_targets": min_targets,
            }
        )
        return report
    report.update(
        {
            "status": STATUS_READY_PILOT,
            "n_conditions": n,
            "provenance_study": provenance.get("dataset_id"),
            "matrix_kind": provenance.get("matrix", {}).get("matrix_kind"),
            "note": "ready for a marked pilot build; this is not official scoring eligibility",
        }
    )
    return report
