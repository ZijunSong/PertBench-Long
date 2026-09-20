"""Canonical observation import. Barcodes are not assumed unique across files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from pertbench_long.data.columns import DEFAULT_ALIASES, resolve_column
from pertbench_long.errors import SchemaError
from pertbench_long.hashes import gene_order_hash, sha256_json
from pertbench_long.schemas.types import ExperimentRecord
from pertbench_long.schemas.validate import validate_experiment_record


CONTROL_ALIASES = {"control": "Control", "ctrl": "Control", "untreated": "Control", "vehicle": "Control"}
STIM_ALIASES = {"stimulated": "IFN", "stim": "IFN", "ifn": "IFN", "ifnb": "IFN", "interferon": "IFN"}


@dataclass
class ImportSummary:
    n_source_rows: int = 0
    n_kept: int = 0
    n_deduplicated: int = 0
    n_conflicts: int = 0
    n_dropped: int = 0
    matrix_kind: str = "unknown"
    gene_order_hash: str = ""
    orientation: str = "cells_x_genes"
    gene_namespace: str = "symbol_as_provided"
    missing_gene_policy: str = "require_intersection"
    conflict_keys: list[str] = field(default_factory=list)
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    donor_metadata_present: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_perturbation_from_index(index: str) -> str:
    token = str(index).rsplit("-", 1)[-1].lower()
    if token in CONTROL_ALIASES or token == "control":
        return "Control"
    if token in STIM_ALIASES or token == "stimulated":
        return "IFN"
    return "unknown"


def infer_matrix_kind(matrix: np.ndarray) -> str:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return "unknown"
    if np.any(finite < 0):
        return "scaled"
    rounded = np.round(finite)
    frac = np.mean(np.abs(finite - rounded) < 1e-8)
    if frac > 0.999 and np.max(finite) >= 20:
        return "counts"
    if np.max(finite) <= 20.0:
        return "log1p"
    return "normalized"


def _to_dense(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def load_table(path: Path, gene_axis: str = "columns") -> tuple[np.ndarray, list[str], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        frame = pd.read_csv(path, index_col=0, sep=sep)
        if gene_axis == "index":
            frame = frame.T
        genes = [str(c) for c in frame.columns]
        obs_ids = [str(i) for i in frame.index]
        return frame.to_numpy(dtype=np.float64), obs_ids, genes
    if suffix == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        matrix = _to_dense(adata.X)
        return matrix, [str(i) for i in adata.obs_names], [str(g) for g in adata.var_names]
    raise SchemaError(f"unsupported import format: {path.suffix}")


def load_obs_frame(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        import anndata as ad

        return ad.read_h5ad(path, backed=None).obs.copy()
    return None


@dataclass
class CanonicalStore:
    matrix: np.ndarray
    records: list[ExperimentRecord]
    gene_ids: list[str]
    summary: ImportSummary
    condition_to_rows: dict[str, list[int]] = field(default_factory=dict)

    def rows_for_condition(self, condition_id: str) -> np.ndarray:
        idx = self.condition_to_rows.get(condition_id, [])
        return self.matrix[idx]

    def condition_ids(self) -> list[str]:
        return sorted(self.condition_to_rows)


def import_tables(
    files: Iterable[Path | str],
    *,
    study: str,
    species: str = "human",
    assay: str = "scrna",
    aliases: Mapping[str, str] | None = None,
    cell_type_from_filename: bool = True,
    default_cell_type: Optional[str] = None,
    source_role: str = "canonical_csv",
) -> CanonicalStore:
    alias_map = dict(DEFAULT_ALIASES)
    if aliases:
        alias_map.update(aliases)
    matrices: list[np.ndarray] = []
    records: list[ExperimentRecord] = []
    gene_sets: list[list[str]] = []
    summary = ImportSummary()
    key_to_vector: dict[str, np.ndarray] = {}
    key_to_index: dict[str, int] = {}

    file_list = [Path(p) for p in files]
    summary.source_files = [str(p) for p in file_list]

    for path in file_list:
        matrix, obs_ids, genes = load_table(path)
        gene_sets.append(genes)
        obs_meta = load_obs_frame(path)
        summary.n_source_rows += matrix.shape[0]
        cell_type_col = None
        pert_col = None
        donor_col = None
        sample_col = None
        if obs_meta is not None:
            cell_type_col = resolve_column(obs_meta.columns, alias_map, "cell_type")
            pert_col = resolve_column(obs_meta.columns, alias_map, "perturbation")
            donor_col = resolve_column(obs_meta.columns, alias_map, "donor")
            sample_col = resolve_column(obs_meta.columns, alias_map, "sample_id")
            if donor_col is not None:
                summary.donor_metadata_present = True

        inferred_ct = default_cell_type
        if cell_type_from_filename and inferred_ct is None:
            inferred_ct = _cell_type_from_name(path.name)

        for i, original_id in enumerate(obs_ids):
            cell_type = inferred_ct or "unknown"
            perturbation = infer_perturbation_from_index(original_id)
            donor = None
            sample_id = inferred_ct or path.stem
            if obs_meta is not None:
                if cell_type_col:
                    cell_type = str(obs_meta.iloc[i][cell_type_col])
                if pert_col:
                    perturbation = str(obs_meta.iloc[i][pert_col])
                if donor_col:
                    raw_donor = obs_meta.iloc[i][donor_col]
                    donor = None if pd.isna(raw_donor) else str(raw_donor)
                if sample_col:
                    sample_id = str(obs_meta.iloc[i][sample_col])
            observation_id = f"{study}|{sample_id}|{original_id}"
            rec = ExperimentRecord(
                observation_id=observation_id,
                study=study,
                species=species,
                cell_type=cell_type,
                perturbation_id=perturbation,
                dose=None,
                dose_unit="none",
                time=None,
                time_unit="none",
                assay=assay,
                original_obs_id=original_id,
                sample_id=sample_id,
                donor=donor,
                source_file=str(path),
            )
            rec = rec.__class__(**{**asdict(rec), "condition_id": rec.condition_key()})
            vector = np.asarray(matrix[i], dtype=np.float64)
            if observation_id in key_to_vector:
                prev = key_to_vector[observation_id]
                if prev.shape == vector.shape and np.allclose(prev, vector, equal_nan=True):
                    summary.n_deduplicated += 1
                    continue
                summary.n_conflicts += 1
                summary.conflict_keys.append(observation_id)
                summary.n_dropped += 1
                summary.dropped_reasons["numeric_conflict"] = summary.dropped_reasons.get("numeric_conflict", 0) + 1
                continue
            key_to_vector[observation_id] = vector
            key_to_index[observation_id] = len(records)
            records.append(rec)
            matrices.append(vector.reshape(1, -1))

    if not records:
        raise SchemaError("import produced zero observations")
    genes = _intersect_genes(gene_sets)
    aligned = []
    for mat, gene_list in zip(_chunk_matrices(matrices, gene_sets), gene_sets):
        indexer = [gene_list.index(g) for g in genes]
        aligned.append(mat[:, indexer])
    full = np.vstack(aligned) if aligned else np.zeros((0, 0))
    # Rebuild matrix in record order. matrices was appended per kept row.
    full = np.vstack(matrices)
    # If gene lists differ, re-read would be needed; require identical genes.
    unique_gene_lists = {tuple(g) for g in gene_sets}
    if len(unique_gene_lists) != 1:
        common = _intersect_genes(gene_sets)
        rebuilt = []
        offset = 0
        # We stored full-width vectors; re-import per file for safety.
        rebuilt_records = []
        rebuilt_rows = []
        for path, gene_list in zip(file_list, gene_sets):
            matrix, obs_ids, _genes = load_table(path)
            col_index = {g: j for j, g in enumerate(gene_list)}
            indexer = [col_index[g] for g in common]
            for rec in records:
                if rec.source_file != str(path):
                    continue
                local = rec.original_obs_id
                try:
                    row_i = obs_ids.index(local)
                except ValueError:
                    continue
                rebuilt_rows.append(matrix[row_i, indexer])
                rebuilt_records.append(rec)
        records = rebuilt_records
        full = np.vstack(rebuilt_rows) if rebuilt_rows else np.zeros((0, len(common)))
        genes = common
        summary.notes.append("aligned_to_gene_intersection")
    else:
        genes = gene_sets[0]
        full = np.vstack(matrices)

    kind = infer_matrix_kind(full)
    summary.matrix_kind = kind
    for rec_i, rec in enumerate(records):
        records[rec_i] = rec.__class__(**{**asdict(rec), "matrix_kind": kind})
        validate_experiment_record(asdict(records[rec_i]), where=f"records[{rec_i}]")
    summary.n_kept = len(records)
    summary.gene_order_hash = gene_order_hash(genes)
    cond: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        cond.setdefault(rec.condition_key(), []).append(i)
    if kind == "unknown":
        summary.notes.append("matrix_kind_unknown_not_eligible_for_official_scoring")
    if not summary.donor_metadata_present:
        summary.notes.append("donor_metadata_missing_recorded_as_null")
    return CanonicalStore(matrix=full, records=records, gene_ids=genes, summary=summary, condition_to_rows=cond)


def _intersect_genes(gene_sets: list[list[str]]) -> list[str]:
    common = set(gene_sets[0])
    for genes in gene_sets[1:]:
        common &= set(genes)
    if not common:
        raise SchemaError("empty gene intersection across imported files")
    # preserve first-file order
    return [g for g in gene_sets[0] if g in common]


def _chunk_matrices(matrices: list[np.ndarray], gene_sets: list[list[str]]) -> list[np.ndarray]:
    return matrices


def _cell_type_from_name(name: str) -> Optional[str]:
    known = ["FCGR3A+Mono", "CD14+Mono", "Dendritic", "CD4T", "CD8T", "NK", "B"]
    for item in known:
        if item in name:
            return item
    return None


def store_to_frames(store: CanonicalStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.DataFrame([asdict(r) for r in store.records])
    var = pd.DataFrame({"gene_id": store.gene_ids}).set_index("gene_id")
    return obs, var


def write_import_summary(summary: ImportSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
