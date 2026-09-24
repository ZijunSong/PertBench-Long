"""Prepare a sci-Plex3 standard input from author-format counts + cell metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertbench_long import __version__
from pertbench_long.data.parsing import is_missing, parse_bool, require_dose_pair, require_time_pair
from pertbench_long.data.prepare.matrix_io import align_barcodes, publish_directory, read_expression, take_rows
from pertbench_long.episodes.evidence import PREPARED_MANIFEST_VERSION, PREPARED_PROVENANCE_VERSION
from pertbench_long.errors import ConfigError, SchemaError
from pertbench_long.hashes import sha256_file, sha256_json
from pertbench_long.schemas.conditions import (
    PERTURBATION_CHEMICAL,
    PERTURBATION_CONTROL,
    make_condition_id,
)

from pertbench_long.data.parsing import looks_like_control_name


def _select_counts(source: Path, input_profile: str | None) -> Path:
    csv_path = source / "counts.csv"
    h5_path = source / "counts.h5ad"
    if input_profile == "csv_bundle":
        if not csv_path.exists():
            raise ConfigError("input_profile=csv_bundle requires counts.csv")
        return csv_path
    if input_profile == "h5ad_bundle":
        if not h5_path.exists():
            raise ConfigError("input_profile=h5ad_bundle requires counts.h5ad")
        return h5_path
    if csv_path.exists() and h5_path.exists():
        raise ConfigError("both counts.csv and counts.h5ad exist; set input_profile instead of picking by file order")
    if csv_path.exists():
        return csv_path
    if h5_path.exists():
        return h5_path
    raise ConfigError("sci-Plex prepare requires counts.csv or counts.h5ad")


def _require_col(frame: pd.DataFrame, names: tuple[str, ...], *, what: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise SchemaError(f"sci-Plex prepare is missing {what}; looked for {list(names)}")


def prepare_sciplex(
    source_dir: Path | str,
    dest_dir: Path | str,
    *,
    study: str = "sciplex3",
    accession: str = "GSM4150378",
    unit_basis: str | None = None,
    input_profile: str | None = None,
    matrix_layer: str | None = None,
    matrix_kind: str | None = None,
) -> dict[str, Any]:
    """Align counts with cell/sample/drug/dose/time/vehicle annotations.

    `unit_basis` may be recorded when the source study states a uniform unit.
    It is never inferred from file names.
    """
    source = Path(source_dir)
    dest = Path(dest_dir)
    if not source.exists():
        raise ConfigError(f"sci-Plex source directory does not exist: {source}")
    cells_path = source / "cells.csv"
    counts_path = _select_counts(source, input_profile)
    if not cells_path.exists():
        raise ConfigError("sci-Plex prepare requires cells.csv")
    cells = pd.read_csv(cells_path)
    barcode_col = _require_col(cells, ("cell_barcode", "barcode", "cell", "obs_id"), what="cell barcode")
    cells[barcode_col] = cells[barcode_col].astype(str)
    if cells[barcode_col].duplicated().any():
        raise SchemaError("duplicate cell barcodes in sci-Plex cells.csv; they are not silently dropped")
    declared = "counts" if counts_path.suffix.lower() == ".csv" else matrix_kind
    matrix, obs_ids, genes, matrix_kind = read_expression(counts_path, layer=matrix_layer, declared_kind=declared)
    if len(genes) != len(set(genes)):
        raise SchemaError("duplicate gene IDs in sci-Plex counts; they are not silently dropped")
    if len(obs_ids) != len(set(obs_ids)):
        raise SchemaError("duplicate barcodes in sci-Plex counts")
    cell_index = {str(b): i for i, b in enumerate(cells[barcode_col])}
    known = set(obs_ids)
    extra_meta = [b for b in cells[barcode_col] if b not in known]
    keep_obs, rows, missing_identity = align_barcodes(obs_ids, cell_index)
    if not keep_obs:
        raise SchemaError("no overlapping barcodes between sci-Plex counts and cells.csv")
    matrix = take_rows(matrix, rows)
    ct_col = _require_col(cells, ("cell_type", "cell_line", "context_id"), what="cell context")
    pert_col = _require_col(cells, ("perturbation", "drug", "treatment", "perturbation_id"), what="drug")
    dose_col = _require_col(cells, ("dose", "dose_value"), what="dose")
    dose_unit_col = "dose_unit" if "dose_unit" in cells.columns else None
    time_col = "time" if "time" in cells.columns else None
    time_unit_col = "time_unit" if "time_unit" in cells.columns else None
    vehicle_col = "vehicle" if "vehicle" in cells.columns else None
    sample_col = "sample_id" if "sample_id" in cells.columns else None
    rep_col = "replicate_id" if "replicate_id" in cells.columns else None
    batch_col = "batch_id" if "batch_id" in cells.columns else None
    ctrl_col = "is_control" if "is_control" in cells.columns else None

    obs_rows = []
    condition_ids = []
    for oid in keep_obs:
        row = cells.iloc[cell_index[oid]]
        pert = str(row[pert_col])
        flag = parse_bool(row[ctrl_col], field="is_control") if ctrl_col else None
        is_control = bool(flag) if flag is not None else looks_like_control_name(pert)
        kind = PERTURBATION_CONTROL if is_control else PERTURBATION_CHEMICAL
        if kind == PERTURBATION_CHEMICAL:
            dose, dose_unit = require_dose_pair(row[dose_col], None if dose_unit_col is None else row[dose_unit_col], require_exact=True)
        else:
            dose, dose_unit = None, "none"
        time, time_unit = require_time_pair(None if time_col is None else row[time_col], None if time_unit_col is None else row[time_unit_col])
        vehicle = None if not vehicle_col or is_missing(row[vehicle_col]) else str(row[vehicle_col])
        context = str(row[ct_col])
        components = (vehicle or pert,) if is_control else (pert,)
        cid = make_condition_id(
            study=study,
            context_id=context,
            perturbation_kind=kind,
            perturbation_components=components,
            dose=dose,
            dose_unit=dose_unit,
            time=time,
            time_unit=time_unit,
            assay="scrna",
            require_exact_dose=kind == PERTURBATION_CHEMICAL,
        )
        condition_ids.append(cid)
        obs_rows.append(
            {
                "cell_type": context,
                "context_id": context,
                "perturbation": pert,
                "perturbation_id": pert,
                "dose": None if dose is None else float(dose),
                "dose_unit": dose_unit,
                "time": None if time is None else float(time),
                "time_unit": time_unit,
                "vehicle": vehicle,
                "sample_id": None if not sample_col or is_missing(row[sample_col]) else str(row[sample_col]),
                "replicate_id": None if not rep_col or is_missing(row[rep_col]) else str(row[rep_col]),
                "batch_id": None if not batch_col or is_missing(row[batch_col]) else str(row[batch_col]),
                "is_control": is_control,
                "condition_id": cid,
                "perturbation_kind": kind,
            }
        )
    obs = pd.DataFrame(obs_rows, index=keep_obs)
    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    import anndata as ad

    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    if matrix_kind == "counts":
        adata.layers["counts"] = matrix.copy() if hasattr(matrix, "copy") else matrix
        adata.uns["counts_layer"] = "counts"
    adata.uns["matrix_kind"] = matrix_kind
    adata.uns["study"] = study
    adata.uns["accession"] = accession
    matrix_path = staging / "matrix.h5ad"
    adata.write_h5ad(matrix_path)
    cond = (
        obs.groupby("condition_id", dropna=False)
        .agg(
            n_cells=("perturbation", "size"),
            context_id=("context_id", "first"),
            perturbation=("perturbation", "first"),
            perturbation_kind=("perturbation_kind", "first"),
            dose=("dose", "first"),
            dose_unit=("dose_unit", "first"),
            time=("time", "first"),
            time_unit=("time_unit", "first"),
            n_replicates=("replicate_id", lambda s: s.dropna().nunique()),
        )
        .reset_index()
    )
    cond_path = staging / "conditions.parquet"
    cond.to_parquet(cond_path, index=False)
    provenance = {
        "provenance_version": PREPARED_PROVENANCE_VERSION,
        "dataset_id": study,
        "source_lock_status": "blocked",
        "code_version": __version__,
        "input_profile": input_profile or counts_path.suffix.lstrip("."),
        "accession": accession,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source.resolve()),
        "source_files": {
            "cells.csv": {"sha256": sha256_file(cells_path), "sha256_verified_by": "local_compute"},
            counts_path.name: {"sha256": sha256_file(counts_path), "sha256_verified_by": "local_compute"},
        },
        "matrix": {"file": "matrix.h5ad", "layer": "counts" if matrix_kind == "counts" else "X", "matrix_kind": matrix_kind},
        "unit_basis": unit_basis,
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "n_conditions": int(cond.shape[0]),
        "dropped_missing_identity": len(missing_identity),
        "unused_metadata_barcodes": len(extra_meta),
        "notes": [
            "X and layers['counts'] are the same author-format count matrix.",
            "Replicate IDs stay null when the source table has none; groups are not invented.",
            "This provenance is a local prepare record, not an independently audited GEO lock.",
        ],
    }
    prov_path = staging / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "n_conditions": int(cond.shape[0]),
        "n_treated": int((cond["perturbation_kind"] != PERTURBATION_CONTROL).sum()),
        "n_controls": int((cond["perturbation_kind"] == PERTURBATION_CONTROL).sum()),
        "n_doses": int(cond.loc[cond["perturbation_kind"] != PERTURBATION_CONTROL, "dose"].nunique()),
        "dropped_missing_identity": missing_identity[:20],
        "peak_rss_note": "prepare peak RSS is a data-prep measurement, not the agent resource_profile",
        "provenance_sha256": sha256_json(provenance),
    }
    qc_path = staging / "qc_report.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    manifest = {
        "manifest_version": PREPARED_MANIFEST_VERSION,
        "provenance_sha256": sha256_file(prov_path),
        "outputs": {
            "matrix.h5ad": sha256_file(matrix_path),
            "conditions.parquet": sha256_file(cond_path),
            "qc_report.json": sha256_file(qc_path),
        },
    }
    (staging / "prepared_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    publish_directory(staging, dest)
    return {"status": "ok", "dest": str(dest), "matrix": str(dest / "matrix.h5ad"), "provenance": str(dest / "provenance.json"), "qc": qc}
