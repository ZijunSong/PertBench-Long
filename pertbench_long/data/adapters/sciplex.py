"""sci-Plex3 condition import. Does not treat a third-party h5ad as raw counts by default."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pertbench_long.data.adapter import CanonicalStore, ImportSummary, load_matrix_sparse
from pertbench_long.data.adapters.base import load_acquisition_manifest, require_files, resolve_source_dir
from pertbench_long.data.columns import DEFAULT_ALIASES, resolve_column
from pertbench_long.data.parsing import looks_like_control_name, parse_bool, require_dose_pair, require_matrix_kind, require_time_pair
from pertbench_long.errors import SchemaError
from pertbench_long.hashes import gene_order_hash
from pertbench_long.schemas.conditions import (
    CONTEXT_CELL_LINE,
    PERTURBATION_CHEMICAL,
    PERTURBATION_CONTROL,
    PERTURBATION_GENETIC_SINGLE,
    infer_genetic_kind,
    make_condition_id,
)
from pertbench_long.schemas.types import ExperimentRecord
from pertbench_long.schemas.validate import validate_experiment_record

DEFAULT_FIELD_MAP = {
    "cell_type": "cell_type",
    "context_id": "cell_type",
    "perturbation": "perturbation",
    "dose": "dose",
    "dose_unit": "dose_unit",
    "time": "time",
    "time_unit": "time_unit",
    "sample_id": "sample_id",
    "replicate_id": "replicate_id",
    "batch_id": "batch_id",
    "control_flag": "is_control",
}


def import_sciplex(
    data_dir: Path | str,
    *,
    manifest_path: Path | str | None = None,
    study: str = "sciplex3",
    species: str = "human",
    assay: str = "scrna",
    declared_matrix_kind: str | None = None,
    matrix_filename: str = "matrix.h5ad",
    matrix_layer: str | None = None,
    require_exact_dose: bool = True,
) -> CanonicalStore:
    root = resolve_source_dir(data_dir, dataset_id="sciplex3")
    field_map = dict(DEFAULT_FIELD_MAP)
    required = (matrix_filename,)
    hashes = None
    if manifest_path is not None:
        spec = load_acquisition_manifest(manifest_path)
        field_map.update(spec.field_map)
        if spec.required_files:
            required = spec.required_files
        hashes = getattr(spec, "file_hashes", None)
    files = require_files(root, required, hashes=hashes)
    matrix_path = files.get(matrix_filename) or next(iter(files.values()))
    return _import_h5ad_conditions(
        matrix_path,
        study=study,
        species=species,
        assay=assay,
        declared_matrix_kind=require_matrix_kind(declared_matrix_kind),
        field_map=field_map,
        perturbation_kind_default=PERTURBATION_CHEMICAL,
        context_type=CONTEXT_CELL_LINE,
        require_exact_dose=require_exact_dose,
        matrix_layer=matrix_layer,
        source_notes=["sciplex3 adapter; third-party h5ad is not labeled raw counts unless declared"],
    )


def _import_h5ad_conditions(
    path: Path,
    *,
    study: str,
    species: str,
    assay: str,
    declared_matrix_kind: str,
    field_map: Mapping[str, str],
    perturbation_kind_default: str,
    context_type: str,
    require_exact_dose: bool,
    source_notes: list[str],
    matrix_layer: str | None = None,
) -> CanonicalStore:
    matrix, obs_ids, genes, obs = load_matrix_sparse(path, layer=matrix_layer)
    if obs is None:
        raise SchemaError(f"{path.name} has no observation metadata")
    if len(genes) != len(set(genes)):
        raise SchemaError(f"duplicate gene IDs in {path.name}; they are not silently dropped")
    if matrix.shape[0] != len(obs_ids) or matrix.shape[1] != len(genes):
        raise SchemaError("matrix shape does not match obs/var names")
    aliases = {**DEFAULT_ALIASES, **field_map}
    missing_required = []
    for logical in ("cell_type", "perturbation"):
        col = resolve_column(obs.columns, aliases, logical) or field_map.get(logical)
        if col not in obs.columns:
            missing_required.append(logical)
    if missing_required:
        raise SchemaError(f"missing required metadata columns: {missing_required}")

    def _col(logical: str) -> str | None:
        mapped = field_map.get(logical)
        if mapped and mapped in obs.columns:
            return mapped
        return resolve_column(obs.columns, aliases, logical)

    ct_col = _col("cell_type")
    ctx_col = _col("context_id") or ct_col
    pert_col = _col("perturbation")
    dose_col = _col("dose")
    dose_unit_col = _col("dose_unit")
    time_col = _col("time")
    time_unit_col = _col("time_unit")
    sample_col = _col("sample_id")
    rep_col = _col("replicate_id")
    batch_col = _col("batch_id")
    ctrl_col = _col("control_flag")
    donor_col = _col("donor")

    records: list[ExperimentRecord] = []
    for i, original_id in enumerate(obs_ids):
        row = obs.iloc[i]
        cell_type = str(row[ct_col])
        context_id = str(row[ctx_col]) if ctx_col else cell_type
        perturbation = str(row[pert_col])
        flag = parse_bool(row[ctrl_col], field="is_control") if ctrl_col is not None else None
        is_control = bool(flag) if flag is not None else looks_like_control_name(perturbation)
        kind = PERTURBATION_CONTROL if is_control else perturbation_kind_default
        if not is_control and perturbation_kind_default in {PERTURBATION_GENETIC_SINGLE, "genetic_pair"}:
            kind = infer_genetic_kind(perturbation)
        raw_dose = row[dose_col] if dose_col else None
        raw_dose_unit = row[dose_unit_col] if dose_unit_col else None
        raw_time = row[time_col] if time_col else None
        raw_time_unit = row[time_unit_col] if time_unit_col else None
        if kind != PERTURBATION_CONTROL and require_exact_dose:
            dose, dose_unit = require_dose_pair(raw_dose, raw_dose_unit, require_exact=True)
        elif kind == PERTURBATION_CONTROL:
            dose, dose_unit = None, "none"
        else:
            dose, dose_unit = require_dose_pair(raw_dose, raw_dose_unit, require_exact=False) if raw_dose is not None and str(raw_dose).strip() != "" else (None, "none")
        time, time_unit = require_time_pair(raw_time, raw_time_unit)
        sample_id = None if not sample_col or pd.isna(row[sample_col]) else str(row[sample_col])
        replicate_id = None if not rep_col or pd.isna(row[rep_col]) else str(row[rep_col])
        batch_id = None if not batch_col or pd.isna(row[batch_col]) else str(row[batch_col])
        donor = None if not donor_col or pd.isna(row[donor_col]) else str(row[donor_col])
        if kind == PERTURBATION_CONTROL:
            components = (perturbation,) if perturbation else ("control",)
        elif kind in {PERTURBATION_GENETIC_SINGLE, "genetic_pair"}:
            from pertbench_long.schemas.conditions import normalize_components

            components = normalize_components(perturbation, kind=kind)
        else:
            components = (perturbation,)
        condition_id = make_condition_id(
            study=study,
            context_id=context_id,
            perturbation_kind=kind,
            perturbation_components=components or ("control",),
            dose=None if kind == PERTURBATION_CONTROL else dose,
            dose_unit="none" if kind == PERTURBATION_CONTROL or dose is None else dose_unit,
            time=time,
            time_unit=time_unit if time is not None else "none",
            assay=assay,
            require_exact_dose=require_exact_dose and kind != PERTURBATION_CONTROL and dose is not None,
        )
        rec = ExperimentRecord(
            observation_id=f"{study}|{sample_id or context_id}|{original_id}|{path.name}|{i}",
            study=study,
            species=species,
            cell_type=cell_type,
            perturbation_id=perturbation,
            dose=None if dose is None else float(dose),
            dose_unit=dose_unit if dose is not None else "none",
            time=None if time is None else float(time),
            time_unit=time_unit if time is not None else "none",
            assay=assay,
            original_obs_id=str(original_id),
            sample_id=sample_id,
            donor=donor,
            condition_id=condition_id,
            source_file=str(path),
            matrix_kind=declared_matrix_kind,
            context_id=context_id,
            context_type=context_type,
            perturbation_kind=kind,
            perturbation_components=components or ("control",),
            replicate_id=replicate_id,
            batch_id=batch_id,
            donor_id=donor,
            identity_version="cid_v1",
        )
        validate_experiment_record(asdict(rec), where=f"sciplex[{i}]")
        records.append(rec)

    summary = ImportSummary(
        n_source_rows=len(records),
        n_kept=len(records),
        matrix_kind=declared_matrix_kind,
        gene_order_hash=gene_order_hash(genes),
        source_files=[path.name],
        donor_metadata_present=any(r.donor is not None for r in records),
        notes=list(source_notes),
    )
    cond: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        cond.setdefault(rec.condition_id, []).append(i)
    return CanonicalStore(matrix=matrix, records=records, gene_ids=list(genes), summary=summary, condition_to_rows=cond)
