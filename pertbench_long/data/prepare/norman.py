"""Prepare a Norman 2019 CRISPRa standard input from counts + cell identities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertbench_long.data.parsing import NTC_TOKENS, is_missing, parse_bool
from pertbench_long.errors import ConfigError, SchemaError
from pertbench_long.hashes import sha256_file, sha256_json
from pertbench_long.schemas.conditions import (
    PERTURBATION_CONTROL,
    PERTURBATION_GENETIC_PAIR,
    PERTURBATION_GENETIC_SINGLE,
    make_condition_id,
    parse_component_token,
)


def _read_counts(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    if path.suffix.lower() == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
        if hasattr(matrix, "toarray"):
            matrix = np.asarray(matrix.toarray(), dtype=np.float64)
        else:
            matrix = np.asarray(matrix, dtype=np.float64)
        return matrix, [str(i) for i in adata.obs_names], [str(g) for g in adata.var_names]
    frame = pd.read_csv(path, index_col=0)
    return frame.to_numpy(dtype=np.float64), [str(i) for i in frame.index], [str(c) for c in frame.columns]


def _require_col(frame: pd.DataFrame, names: tuple[str, ...], *, what: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise SchemaError(f"Norman prepare is missing {what}; looked for {list(names)}")


def _guide_lookup(guide_path: Path | None) -> dict[str, dict[str, Any]]:
    if guide_path is None or not guide_path.exists():
        return {}
    frame = pd.read_csv(guide_path)
    name_col = _require_col(frame, ("guide", "guide_identity", "guide_id"), what="guide id")
    gene_col = _require_col(frame, ("gene", "target_gene", "gene_id"), what="guide target gene")
    ntc_col = "is_ntc" if "is_ntc" in frame.columns else None
    out = {}
    for _, row in frame.iterrows():
        gene = str(row[gene_col])
        is_ntc = parse_bool(row[ntc_col], field="is_ntc") if ntc_col else gene.lower() in NTC_TOKENS
        out[str(row[name_col])] = {"gene": gene, "is_ntc": bool(is_ntc)}
    return out


def resolve_norman_identity(raw: str, *, guide_table: dict[str, dict[str, Any]] | None = None) -> tuple[str, tuple[str, ...]]:
    text = str(raw).strip()
    if not text:
        raise SchemaError("empty Norman guide identity")
    if guide_table and text in guide_table:
        item = guide_table[text]
        if item["is_ntc"]:
            return PERTURBATION_CONTROL, (str(item["gene"]),)
        return PERTURBATION_GENETIC_SINGLE, (str(item["gene"]),)
    parts = list(parse_component_token(text))
    resolved = []
    ntc_flags = []
    for part in parts:
        if guide_table and part in guide_table:
            item = guide_table[part]
            resolved.append(str(item["gene"]))
            ntc_flags.append(bool(item["is_ntc"]))
        else:
            resolved.append(part)
            ntc_flags.append(part.lower() in NTC_TOKENS)
    genes = [g for g, ntc in zip(resolved, ntc_flags) if not ntc]
    if not genes:
        return PERTURBATION_CONTROL, (resolved[0] if resolved else "control",)
    if len(genes) == 1:
        return PERTURBATION_GENETIC_SINGLE, (genes[0],)
    if len(genes) == 2:
        if genes[0] == genes[1]:
            raise SchemaError(f"Norman pair cannot be a self-pair: {text}")
        return PERTURBATION_GENETIC_PAIR, tuple(sorted(genes))
    raise SchemaError(f"Norman identity {text!r} has more than two non-NTC genes; multiple guides are not silently reduced")


def prepare_norman(
    source_dir: Path | str,
    dest_dir: Path | str,
    *,
    study: str = "norman2019",
    accession: str = "GSE133344",
) -> dict[str, Any]:
    source = Path(source_dir)
    dest = Path(dest_dir)
    if not source.exists():
        raise ConfigError(f"Norman source directory does not exist: {source}")
    ident_path = source / "cell_identities.csv"
    counts_path = source / "counts.csv" if (source / "counts.csv").exists() else source / "counts.h5ad"
    if not ident_path.exists() or not counts_path.exists():
        raise ConfigError("Norman prepare requires cell_identities.csv and counts.csv|counts.h5ad")
    ident = pd.read_csv(ident_path)
    barcode_col = _require_col(ident, ("cell_barcode", "barcode", "cell"), what="cell barcode")
    ident[barcode_col] = ident[barcode_col].astype(str)
    if ident[barcode_col].duplicated().any():
        raise SchemaError("duplicate barcodes in Norman cell_identities.csv")
    guide_col = _require_col(ident, ("guide_identity", "perturbation", "guide", "perturbation_id"), what="guide identity")
    guide_table = _guide_lookup(source / "guides.csv")
    matrix, obs_ids, genes = _read_counts(counts_path)
    if len(genes) != len(set(genes)):
        raise SchemaError("duplicate gene IDs in Norman counts")
    if len(obs_ids) != len(set(obs_ids)):
        raise SchemaError("duplicate barcodes in Norman counts")
    meta_index = {str(b): i for i, b in enumerate(ident[barcode_col])}
    missing_identity = [oid for oid in obs_ids if oid not in meta_index]
    keep_obs = [oid for oid in obs_ids if oid in meta_index]
    if not keep_obs:
        raise SchemaError("no overlapping barcodes between Norman counts and cell_identities.csv")
    matrix = np.asarray(matrix[[obs_ids.index(oid) for oid in keep_obs]], dtype=np.float64)
    ctx_col = "cell_type" if "cell_type" in ident.columns else None
    sample_col = "sample_id" if "sample_id" in ident.columns else None
    rep_col = "replicate_id" if "replicate_id" in ident.columns else None
    batch_col = "batch_id" if "batch_id" in ident.columns else None
    ctrl_col = "is_control" if "is_control" in ident.columns else None
    obs_rows = []
    for oid in keep_obs:
        row = ident.iloc[meta_index[oid]]
        raw_id = str(row[guide_col])
        kind, components = resolve_norman_identity(raw_id, guide_table=guide_table or None)
        flag = parse_bool(row[ctrl_col], field="is_control") if ctrl_col else None
        if flag is True:
            kind, components = PERTURBATION_CONTROL, components or ("control",)
        context = str(row[ctx_col]) if ctx_col and not is_missing(row[ctx_col]) else "K562"
        cid = make_condition_id(
            study=study,
            context_id=context,
            perturbation_kind=kind,
            perturbation_components=components,
            dose=None,
            dose_unit="none",
            assay="scrna",
        )
        display = "Control" if kind == PERTURBATION_CONTROL else "+".join(components)
        obs_rows.append(
            {
                "cell_type": context,
                "context_id": context,
                "perturbation": display,
                "perturbation_id": display,
                "source_guide_identity": raw_id,
                "is_control": kind == PERTURBATION_CONTROL,
                "sample_id": None if not sample_col or is_missing(row[sample_col]) else str(row[sample_col]),
                "replicate_id": None if not rep_col or is_missing(row[rep_col]) else str(row[rep_col]),
                "batch_id": None if not batch_col or is_missing(row[batch_col]) else str(row[batch_col]),
                "condition_id": cid,
                "perturbation_kind": kind,
                "perturbation_components": "+".join(components),
                "intervention": "CRISPRa",
            }
        )
    obs = pd.DataFrame(obs_rows, index=keep_obs)
    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
    dest.mkdir(parents=True, exist_ok=True)
    import anndata as ad

    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = matrix.copy()
    adata.uns["matrix_kind"] = "counts"
    adata.uns["counts_layer"] = "counts"
    adata.uns["intervention"] = "CRISPRa"
    matrix_path = dest / "matrix.h5ad"
    adata.write_h5ad(matrix_path)
    cond = (
        obs.groupby("condition_id", dropna=False)
        .agg(
            n_cells=("perturbation", "size"),
            context_id=("context_id", "first"),
            perturbation=("perturbation", "first"),
            perturbation_kind=("perturbation_kind", "first"),
            n_replicates=("replicate_id", lambda s: s.dropna().nunique()),
        )
        .reset_index()
    )
    cond.to_parquet(dest / "conditions.parquet", index=False)
    provenance = {
        "dataset_id": study,
        "accession": accession,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source.resolve()),
        "source_files": {
            "cell_identities.csv": {"sha256": sha256_file(ident_path), "sha256_verified_by": "local_compute"},
            counts_path.name: {"sha256": sha256_file(counts_path), "sha256_verified_by": "local_compute"},
        },
        "matrix": {"file": "matrix.h5ad", "layer": "counts", "matrix_kind": "counts"},
        "intervention": "CRISPRa",
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "n_conditions": int(cond.shape[0]),
        "dropped_missing_identity": len(missing_identity),
        "guide_table_used": bool(guide_table),
        "notes": [
            "NTC / non-targeting tokens are not treated as a second intervened gene.",
            "Missing pairs are not imputed.",
            "This is a local prepare record, not a locked GEO file list.",
        ],
    }
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "n_singles": int((cond["perturbation_kind"] == PERTURBATION_GENETIC_SINGLE).sum()),
        "n_pairs": int((cond["perturbation_kind"] == PERTURBATION_GENETIC_PAIR).sum()),
        "n_controls": int((cond["perturbation_kind"] == PERTURBATION_CONTROL).sum()),
        "dropped_missing_identity": missing_identity[:20],
        "provenance_sha256": sha256_json(provenance),
    }
    (dest / "qc_report.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return {"status": "ok", "dest": str(dest), "matrix": str(matrix_path), "qc": qc, "provenance": str(dest / "provenance.json")}
