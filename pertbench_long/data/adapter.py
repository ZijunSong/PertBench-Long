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


def numeric_kind_contradiction(matrix: np.ndarray, declared: str) -> list[str]:
    """Range checks only flag contradictions with a declared kind. They never prove log1p."""
    notes: list[str] = []
    finite = np.asarray(matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return ["empty_or_nonfinite_matrix"]
    if declared == "counts" and np.any(finite < 0):
        notes.append("declared_counts_but_negative_values")
    if declared == "counts":
        rounded = np.round(finite)
        frac = float(np.mean(np.abs(finite - rounded) < 1e-8))
        if frac < 0.99:
            notes.append("declared_counts_but_fractional_values")
    if declared in {"log1p", "normalized"} and np.any(finite < 0):
        notes.append(f"declared_{declared}_but_negative_values")
    return notes


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


def _optional_meta(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return None if text in {"", "None", "nan", "NA"} else text


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        raise SchemaError("dose/time must be finite when present")
    return number


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
    declared_matrix_kind: Optional[str] = None,
    conflict_policy: str = "fail_closed",
    sample_id_fallback: str = "filename_declared",
) -> CanonicalStore:
    """Align genes first, then identity-dedup. declared_matrix_kind is required for official scoring."""
    alias_map = dict(DEFAULT_ALIASES)
    if aliases:
        alias_map.update(aliases)
    if conflict_policy not in {"fail_closed", "diagnostic_keep_first"}:
        raise SchemaError("conflict_policy must be fail_closed or diagnostic_keep_first")

    file_list = [Path(p) for p in files]
    summary = ImportSummary()
    summary.source_files = [str(p) for p in file_list]
    loaded: list[dict[str, Any]] = []
    gene_sets: list[list[str]] = []
    declared_kinds: list[str] = []

    for path in file_list:
        matrix, obs_ids, genes = load_table(path)
        if len(genes) != len(set(genes)):
            raise SchemaError(f"duplicate gene IDs in {path.name}")
        if matrix.shape[1] != len(genes):
            raise SchemaError(f"matrix columns do not match gene IDs in {path.name}")
        if not np.all(np.isfinite(matrix)):
            raise SchemaError(f"non-finite values in {path.name}")
        obs_meta = load_obs_frame(path)
        file_kind = declared_matrix_kind
        if obs_meta is not None and "matrix_kind" in obs_meta.columns:
            kinds = {str(v) for v in obs_meta["matrix_kind"].dropna().unique()}
            if len(kinds) > 1:
                raise SchemaError(f"mixed matrix_kind within {path.name}")
            if kinds:
                file_kind = next(iter(kinds))
        declared_kinds.append(file_kind or "undeclared")
        gene_sets.append(genes)
        loaded.append(
            {
                "path": path,
                "matrix": np.asarray(matrix, dtype=np.float64),
                "obs_ids": obs_ids,
                "genes": genes,
                "obs_meta": obs_meta,
                "file_kind": file_kind,
            }
        )
        summary.n_source_rows += matrix.shape[0]

    unique_declared = {k for k in declared_kinds if k != "undeclared"}
    if len(unique_declared) > 1:
        raise SchemaError("mixed declared matrix kinds across files; convert each file before merge")
    kind = declared_matrix_kind or (next(iter(unique_declared)) if unique_declared else None)
    if kind is None:
        summary.matrix_kind = "unknown"
        summary.notes.append("matrix_kind_undeclared_processing_unknown_diagnostic")
        summary.notes.append("numeric inference is diagnostic only and is not used for official transforms")
    else:
        summary.matrix_kind = kind
        for item in loaded:
            summary.notes.extend(numeric_kind_contradiction(item["matrix"], kind))

    common = _intersect_genes(gene_sets)
    summary.notes.append("aligned_to_gene_intersection_before_identity")

    records: list[ExperimentRecord] = []
    rows: list[np.ndarray] = []
    key_to_vector: dict[str, np.ndarray] = {}
    conflict_table: list[str] = []

    for item in loaded:
        path = item["path"]
        genes = item["genes"]
        col_index = {g: j for j, g in enumerate(genes)}
        indexer = [col_index[g] for g in common]
        aligned = item["matrix"][:, indexer]
        obs_meta = item["obs_meta"]
        obs_ids = item["obs_ids"]
        cell_type_col = pert_col = donor_col = sample_col = dose_col = time_col = None
        if obs_meta is not None:
            cell_type_col = resolve_column(obs_meta.columns, alias_map, "cell_type")
            pert_col = resolve_column(obs_meta.columns, alias_map, "perturbation")
            donor_col = resolve_column(obs_meta.columns, alias_map, "donor")
            sample_col = resolve_column(obs_meta.columns, alias_map, "sample_id")
            dose_col = resolve_column(obs_meta.columns, alias_map, "dose")
            time_col = resolve_column(obs_meta.columns, alias_map, "time")
            if donor_col is not None:
                summary.donor_metadata_present = True
        inferred_ct = default_cell_type
        if cell_type_from_filename and inferred_ct is None:
            inferred_ct = _cell_type_from_name(path.name)
        for i, original_id in enumerate(obs_ids):
            cell_type = inferred_ct or "unknown"
            perturbation = infer_perturbation_from_index(original_id)
            donor = None
            sample_id = None
            dose = None
            time = None
            if sample_id_fallback == "filename_declared":
                sample_id = inferred_ct or path.stem
            if obs_meta is not None:
                if cell_type_col:
                    cell_type = str(obs_meta.iloc[i][cell_type_col])
                if pert_col:
                    perturbation = str(obs_meta.iloc[i][pert_col])
                if donor_col:
                    donor = _optional_meta(obs_meta.iloc[i][donor_col])
                if sample_col:
                    sample_id = str(obs_meta.iloc[i][sample_col])
                if dose_col:
                    dose = _optional_float(obs_meta.iloc[i][dose_col])
                if time_col:
                    time = _optional_float(obs_meta.iloc[i][time_col])
            observation_id = f"{study}|{sample_id}|{original_id}|{path.name}|{i}"
            identity_key = f"{study}|{sample_id}|{original_id}"
            rec = ExperimentRecord(
                observation_id=observation_id,
                study=study,
                species=species,
                cell_type=cell_type,
                perturbation_id=perturbation,
                dose=dose,
                dose_unit="none" if dose is None else "unknown",
                time=time,
                time_unit="none" if time is None else "unknown",
                assay=assay,
                original_obs_id=original_id,
                sample_id=sample_id,
                donor=donor,
                source_file=str(path),
                matrix_kind=summary.matrix_kind,
            )
            rec = rec.__class__(**{**asdict(rec), "condition_id": rec.condition_key()})
            vector = np.asarray(aligned[i], dtype=np.float64)
            if identity_key in key_to_vector:
                prev = key_to_vector[identity_key]
                if prev.shape == vector.shape and np.allclose(prev, vector, equal_nan=False):
                    summary.n_deduplicated += 1
                    continue
                summary.n_conflicts += 1
                summary.conflict_keys.append(identity_key)
                conflict_table.append(identity_key)
                if conflict_policy == "fail_closed":
                    raise SchemaError(f"numeric/metadata conflict for {identity_key}; official import is fail-closed")
                summary.n_dropped += 1
                summary.dropped_reasons["numeric_conflict"] = summary.dropped_reasons.get("numeric_conflict", 0) + 1
                continue
            key_to_vector[identity_key] = vector
            records.append(rec)
            rows.append(vector.reshape(1, -1))

    if not records:
        raise SchemaError("import produced zero observations")
    full = np.vstack(rows)
    genes = common
    for rec_i, rec in enumerate(records):
        records[rec_i] = rec.__class__(**{**asdict(rec), "matrix_kind": summary.matrix_kind})
        validate_experiment_record(asdict(records[rec_i]), where=f"records[{rec_i}]")
    summary.n_kept = len(records)
    summary.gene_order_hash = gene_order_hash(genes)
    cond: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        cond.setdefault(rec.condition_key(), []).append(i)
    if summary.matrix_kind == "unknown":
        summary.notes.append("matrix_kind_unknown_not_eligible_for_official_scoring")
    if not summary.donor_metadata_present:
        summary.notes.append("donor_metadata_missing_recorded_as_null")
    if conflict_table:
        summary.notes.append(f"conflicts={len(conflict_table)}")
    return CanonicalStore(matrix=full, records=records, gene_ids=genes, summary=summary, condition_to_rows=cond)


def _intersect_genes(gene_sets: list[list[str]]) -> list[str]:
    common = set(gene_sets[0])
    for genes in gene_sets[1:]:
        common &= set(genes)
    if not common:
        raise SchemaError("empty gene intersection across imported files")
    return [g for g in gene_sets[0] if g in common]


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
