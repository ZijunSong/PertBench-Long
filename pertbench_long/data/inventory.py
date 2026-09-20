"""Data inventory utilities. Unknown matrix kinds stay visible; they never become official labels."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_CANDIDATE_ROOTS = [
    "/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus",
    "/data/ppnm/data/PertDiffBench/data_ori/fig1/raw_task1",
    "/data/ppnm/data/PertDiffBench/data/fig1_task1",
    "/data/ppnm/data/PertDiffBench/data_ori/fig4",
    "/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseenMOA",
    "/data/ppnm/PertDiffBench/data/fig2/task2_unseen_celltype_plus",
]


def _sha256_file(path: Path, limit: Optional[int] = None) -> tuple[Optional[str], str]:
    size = path.stat().st_size
    if limit is not None and size > limit:
        return None, "deferred_large_file"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "full"


def inspect_path(path: Path, hash_limit_bytes: int = 150 * 1024 * 1024) -> dict[str, Any]:
    st = path.stat()
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "format": path.suffix.lower().lstrip(".") or "unknown",
        "matrix_kind": "unknown",
        "matrix_kind_reason": "not_opened_in_inventory_pass",
        "source": None,
        "donor_column": None,
        "study_column": None,
        "batch_column": None,
        "condition_columns": None,
        "control_pairing": None,
        "processing_history": "unverified",
    }
    digest, scope = _sha256_file(path, hash_limit_bytes)
    rec["sha256"] = digest
    rec["hash_scope"] = scope
    return rec


def build_inventory(
    roots: Iterable[str | Path] | None = None,
    *,
    extra_notes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    roots = [Path(r) for r in (roots or DEFAULT_CANDIDATE_ROOTS)]
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for root in roots:
        if not root.exists():
            missing.append(str(root))
            continue
        if root.is_file():
            files.append(inspect_path(root))
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".h5ad", ".txt", ".tsv"}:
                files.append(inspect_path(path))
    required = {
        "pbmc_pilot_csv_dir": "/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus",
        "expected_files": [
            "task1_train_B_exp.csv",
            "task1_train_CD4T_exp.csv",
            "task1_train_CD8T_exp.csv",
            "task1_train_CD14+Mono_exp.csv",
            "task1_train_Dendritic_exp.csv",
            "task1_train_FCGR3A+Mono_exp.csv",
            "task1_train_NK_exp.csv",
            "task1_valid_B_exp.csv",
            "task1_valid_CD4T_exp.csv",
            "task1_valid_CD8T_exp.csv",
            "task1_valid_CD14+Mono_exp.csv",
            "task1_valid_Dendritic_exp.csv",
            "task1_valid_FCGR3A+Mono_exp.csv",
            "task1_valid_NK_exp.csv",
        ],
    }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "roots_requested": [str(r) for r in roots],
        "roots_missing": missing,
        "n_files": len(files),
        "files": files,
        "required_inputs": required,
        "notes": extra_notes or {},
    }
    return payload


def write_inventory(path: Path, inventory: dict[str, Any] | None = None) -> Path:
    payload = inventory or build_inventory()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
