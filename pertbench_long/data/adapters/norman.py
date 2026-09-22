"""Norman 2019 CRISPRa import. A+B and B+A become the same unordered pair condition."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from pertbench_long.data.adapter import CanonicalStore
from pertbench_long.data.adapters.base import load_acquisition_manifest, require_files, resolve_source_dir
from pertbench_long.data.adapters.sciplex import _import_h5ad_conditions
from pertbench_long.errors import SchemaError
from pertbench_long.schemas.conditions import (
    CONTEXT_CELL_LINE,
    PERTURBATION_CONTROL,
    PERTURBATION_GENETIC_PAIR,
    PERTURBATION_GENETIC_SINGLE,
    infer_genetic_kind,
    make_condition_id,
    normalize_components,
)
from pertbench_long.schemas.types import ExperimentRecord
from dataclasses import asdict, replace


def import_norman(
    data_dir: Path | str,
    *,
    manifest_path: Path | str | None = None,
    study: str = "norman2019",
    species: str = "human",
    assay: str = "scrna",
    declared_matrix_kind: str = "counts",
    matrix_filename: str = "matrix.h5ad",
) -> CanonicalStore:
    root = resolve_source_dir(data_dir, dataset_id="norman2019")
    field_map = {
        "cell_type": "cell_type",
        "context_id": "cell_type",
        "perturbation": "perturbation",
        "sample_id": "sample_id",
        "replicate_id": "replicate_id",
        "batch_id": "batch_id",
        "control_flag": "is_control",
    }
    required = (matrix_filename,)
    if manifest_path is not None:
        spec = load_acquisition_manifest(manifest_path)
        field_map.update(spec.field_map)
        if spec.required_files:
            required = spec.required_files
    files = require_files(root, required)
    matrix_path = files.get(matrix_filename) or next(iter(files.values()))
    store = _import_h5ad_conditions(
        matrix_path,
        study=study,
        species=species,
        assay=assay,
        declared_matrix_kind=declared_matrix_kind,
        field_map=field_map,
        perturbation_kind_default=PERTURBATION_GENETIC_SINGLE,
        context_type=CONTEXT_CELL_LINE,
        require_exact_dose=False,
        source_notes=["norman2019 CRISPRa adapter; pairs are unordered; not all knockouts"],
    )
    rewritten: list[ExperimentRecord] = []
    cond: dict[str, list[int]] = {}
    for i, rec in enumerate(store.records):
        if rec.perturbation_kind == PERTURBATION_CONTROL or rec.perturbation_id.lower() in {"control", "ntc", "non-targeting", "negctrl"}:
            kind = PERTURBATION_CONTROL
            components = ("control",)
        else:
            kind = infer_genetic_kind(rec.perturbation_id)
            components = normalize_components(rec.perturbation_id, kind=kind)
        cid = make_condition_id(
            study=rec.study,
            context_id=rec.resolved_context_id(),
            perturbation_kind=kind,
            perturbation_components=components,
            dose=None,
            dose_unit="none",
            time=rec.time,
            time_unit=rec.time_unit if rec.time is not None else "none",
            assay=rec.assay,
            require_exact_dose=False,
        )
        new = replace(
            rec,
            perturbation_kind=kind,
            perturbation_components=components,
            perturbation_id="+".join(components) if kind != PERTURBATION_CONTROL else "Control",
            condition_id=cid,
            identity_version="cid_v1",
        )
        rewritten.append(new)
        cond.setdefault(cid, []).append(i)
    if not rewritten:
        raise SchemaError("norman import produced zero observations")
    return CanonicalStore(
        matrix=store.matrix,
        records=rewritten,
        gene_ids=store.gene_ids,
        summary=store.summary,
        condition_to_rows=cond,
    )
