"""Prepare a Norman 2019 CRISPRa standard input from counts + cell identities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pertbench_long import __version__
from pertbench_long.data.parsing import NTC_TOKENS, is_missing, parse_bool
from pertbench_long.data.prepare.matrix_io import align_barcodes, publish_directory, read_expression, read_mex_bundle, take_rows
from pertbench_long.episodes.evidence import PREPARED_MANIFEST_VERSION, PREPARED_PROVENANCE_VERSION
from pertbench_long.errors import ConfigError, SchemaError
from pertbench_long.hashes import sha256_file, sha256_json
from pertbench_long.schemas.conditions import (
    PERTURBATION_CONTROL,
    PERTURBATION_GENETIC_PAIR,
    PERTURBATION_GENETIC_SINGLE,
    make_condition_id,
    parse_component_token,
)


def _select_counts(source: Path, input_profile: str | None) -> Path:
    csv_path = source / "counts.csv"
    h5_path = source / "counts.h5ad"
    mtx_path = source / "matrix.mtx" if (source / "matrix.mtx").exists() else source / "matrix.mtx.gz"
    if input_profile == "csv_bundle":
        if not csv_path.exists():
            raise ConfigError("input_profile=csv_bundle requires counts.csv")
        return csv_path
    if input_profile == "h5ad_bundle":
        if not h5_path.exists():
            raise ConfigError("input_profile=h5ad_bundle requires counts.h5ad")
        return h5_path
    if input_profile == "mex_bundle":
        if not mtx_path.exists():
            raise ConfigError("input_profile=mex_bundle requires matrix.mtx, barcodes.tsv, and genes.tsv")
        return mtx_path
    present = [p for p in (csv_path, h5_path, mtx_path) if p.exists()]
    if len(present) > 1:
        raise ConfigError("multiple Norman count inputs exist; set input_profile")
    if not present:
        raise ConfigError("Norman prepare requires counts.csv, counts.h5ad, or a Cell Ranger matrix.mtx bundle")
    return present[0]


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
        elif part.lower().startswith(("guide", "grna", "sgrna")):
            raise SchemaError(f"unmapped guide id {part!r} is not a gene; provide guides.csv")
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
    input_profile: str | None = None,
    require_guide_map: bool = False,
    matrix_layer: str | None = None,
    matrix_kind: str | None = None,
) -> dict[str, Any]:
    source = Path(source_dir)
    dest = Path(dest_dir)
    if not source.exists():
        raise ConfigError(f"Norman source directory does not exist: {source}")
    ident_path = source / "cell_identities.csv"
    counts_path = _select_counts(source, input_profile)
    if not ident_path.exists():
        raise ConfigError("Norman prepare requires cell_identities.csv")
    if require_guide_map and not (source / "guides.csv").exists():
        raise ConfigError("this Norman profile requires guides.csv; unresolved guide IDs are not treated as genes")
    ident = pd.read_csv(ident_path)
    barcode_col = _require_col(ident, ("cell_barcode", "barcode", "cell"), what="cell barcode")
    ident[barcode_col] = ident[barcode_col].astype(str)
    if ident[barcode_col].duplicated().any():
        raise SchemaError("duplicate barcodes in Norman cell_identities.csv")
    guide_col = _require_col(ident, ("guide_identity", "perturbation", "guide", "perturbation_id"), what="guide identity")
    guide_table = _guide_lookup(source / "guides.csv")
    mex_files: list[Path] = []
    if counts_path.suffix.lower() == ".mtx" or counts_path.name.startswith("matrix.mtx"):
        matrix, obs_ids, genes, matrix_kind, mex_files = read_mex_bundle(source)
    else:
        declared = "counts" if counts_path.suffix.lower() == ".csv" else matrix_kind
        matrix, obs_ids, genes, matrix_kind = read_expression(counts_path, layer=matrix_layer, declared_kind=declared)
    if len(genes) != len(set(genes)):
        raise SchemaError("duplicate gene IDs in Norman counts")
    if len(obs_ids) != len(set(obs_ids)):
        raise SchemaError("duplicate barcodes in Norman counts")
    meta_index = {str(b): i for i, b in enumerate(ident[barcode_col])}
    keep_obs, rows, missing_identity = align_barcodes(obs_ids, meta_index)
    if not keep_obs:
        raise SchemaError("no overlapping barcodes between Norman counts and cell_identities.csv")
    matrix = take_rows(matrix, rows)
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
    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    import anndata as ad

    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = matrix.copy() if hasattr(matrix, "copy") else matrix
    adata.uns["matrix_kind"] = matrix_kind
    adata.uns["counts_layer"] = "counts"
    adata.uns["intervention"] = "CRISPRa"
    matrix_path = staging / "matrix.h5ad"
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
    cond_path = staging / "conditions.parquet"
    cond.to_parquet(cond_path, index=False)
    provenance = {
        "provenance_version": PREPARED_PROVENANCE_VERSION,
        "dataset_id": study,
        "source_lock_status": "blocked",
        "code_version": __version__,
        "input_profile": input_profile or counts_path.name,
        "accession": accession,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source.resolve()),
        "source_files": {
            path.name: {"sha256": sha256_file(path), "sha256_verified_by": "local_compute", "path": str(path)}
            for path in [ident_path, *mex_files, counts_path, source / "guides.csv"]
            if path.exists()
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
    prov_path = staging / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
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
    return {"status": "ok", "dest": str(dest), "matrix": str(dest / "matrix.h5ad"), "qc": qc, "provenance": str(dest / "provenance.json")}
