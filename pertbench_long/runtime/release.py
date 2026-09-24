"""Release preflight: bound hashes, declared whitelist only, no glob of extra public files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pertbench_long.errors import IntegrityError, InvalidEpisode
from pertbench_long.hashes import gene_order_hash, sha256_file, sha256_json
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def declared_public_evidence_names(public_spec) -> list[str]:
    names: list[str] = []
    for artifact_id in list(public_spec.initial_evidence) + list(public_spec.reference_evidence):
        token = str(artifact_id)
        names.append(token if token.endswith(".h5ad") else f"{token}.h5ad")
    return names


def _expected_hashes(public_spec) -> dict[str, str]:
    meta = dict(public_spec.artifact_metadata or {})
    expected: dict[str, str] = {}
    for group in ("initial", "controls"):
        item = meta.get(group) or {}
        rel = item.get("relative_path")
        digest = item.get("sha256")
        if rel and digest:
            expected[str(rel).replace("\\", "/")] = str(digest)
    if meta.get("gene_universe_sha256"):
        expected[str(public_spec.gene_universe_artifact)] = str(meta["gene_universe_sha256"])
    if meta.get("submission_contract_sha256"):
        expected["submission_contract.json"] = str(meta["submission_contract_sha256"])
    for rel, digest in dict(meta.get("public_files") or {}).items():
        if digest:
            expected[str(rel)] = str(digest)
    return expected


def check_release(public_dir: Path, private_manifest: Path) -> dict[str, Any]:
    """Shared by validate --check-artifacts and run/resume preflight.

    Compares live file bytes to hashes bound in both the public pack and the
    private-embedded public spec. Extra files in public/evidence are rejected.
    """
    public_dir = Path(public_dir).resolve()
    private_manifest = Path(private_manifest).resolve()
    if not (public_dir / "episode.json").exists():
        raise InvalidEpisode("public episode.json is missing")
    if not private_manifest.exists():
        raise InvalidEpisode("private manifest is missing")
    public_spec = validate_public_payload(_load_json(public_dir / "episode.json"))
    private_spec = validate_private_payload(_load_json(private_manifest))
    pub_a = public_spec.to_dict()
    pub_b = private_spec.public.to_dict()
    if sha256_json(pub_a) != sha256_json(pub_b):
        raise IntegrityError("public pack does not match the private-bound public spec")
    if private_spec.episode_id != public_spec.episode_id:
        raise IntegrityError("private episode_id does not match public pack")

    expected = _expected_hashes(public_spec)
    if not expected:
        raise InvalidEpisode("release is missing bound artifact hashes")

    evidence_dir = public_dir / "evidence"
    if not evidence_dir.is_dir():
        raise InvalidEpisode("public/evidence is missing")
    declared = set(declared_public_evidence_names(public_spec))
    present = {p.name for p in evidence_dir.iterdir() if p.is_file()}
    extra = sorted(name for name in present if name.endswith(".h5ad") and name not in declared)
    if extra:
        raise InvalidEpisode(f"undeclared public evidence files are not allowed: {extra}")
    missing = sorted(name for name in declared if name not in present)
    if missing:
        raise InvalidEpisode(f"declared public evidence files are missing: {missing}")

    live: dict[str, str] = {}
    for rel, digest in expected.items():
        path = public_dir / rel
        if not path.exists():
            raise InvalidEpisode(f"declared artifact missing: {rel}")
        live_hash = sha256_file(path)
        live[rel] = live_hash
        if live_hash != digest:
            raise IntegrityError(f"artifact hash mismatch for {rel}")

    genes_path = public_dir / public_spec.gene_universe_artifact
    gene_ids = [line for line in genes_path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    if gene_order_hash(gene_ids) != private_spec.gene_order_hash:
        raise IntegrityError("gene universe order does not match the private spec")
    if gene_ids != list(private_spec.gene_ids):
        raise IntegrityError("gene universe ids do not match the private spec")

    private_root = private_manifest.parent
    labels = Path(private_spec.label_artifact)
    labels_path = labels if labels.is_absolute() else (private_root / labels)
    if not labels_path.exists():
        raise InvalidEpisode("label artifact is missing")
    if sha256_file(labels_path) != private_spec.private_data_hash:
        raise IntegrityError("label artifact hash does not match private_data_hash")

    _reject_treated_in_controls(public_dir, private_spec)

    q_arts = (private_spec.provenance or {}).get("queryable_artifacts") or []
    for item in q_arts:
        exp_id = item.get("experiment_id")
        artifact = item.get("artifact") or {}
        qpath = private_root / "queryable" / f"ev_{exp_id}.h5ad"
        if not qpath.exists():
            raise InvalidEpisode(f"queryable artifact missing for {exp_id}")
        expected_q = artifact.get("sha256")
        if expected_q and sha256_file(qpath) != expected_q:
            raise IntegrityError(f"queryable artifact hash mismatch for {exp_id}")

    return {
        "ok": True,
        "episode_id": public_spec.episode_id,
        "declared_public_evidence": sorted(declared),
        "live_hashes": live,
        "public_spec_hash": sha256_json(pub_a),
        "private_manifest_sha256": sha256_file(private_manifest),
        "label_sha256": sha256_file(labels_path),
        "gene_order_hash": private_spec.gene_order_hash,
        "scoring_track": (private_spec.scoring_config or {}).get("scoring_track", "official"),
        "synthetic": bool(public_spec.synthetic),
        "experimental_budget": int(public_spec.experimental_budget),
    }


MOUNT_POLICY = "declared_public_v1"
PUBLIC_PROTOCOL_FILES = ("episode.json", "submission_contract.json")


def declared_public_relative_paths(public_spec) -> list[str]:
    """Relative paths inside the public pack that an agent workspace may see."""
    rels = ["evidence/" + name for name in declared_public_evidence_names(public_spec)]
    rels.append(str(public_spec.gene_universe_artifact))
    rels.extend(PUBLIC_PROTOCOL_FILES)
    rels.extend(sorted(dict(getattr(public_spec, "artifact_metadata", {}) or {}).get("public_files") or {}))
    cleaned = []
    for rel in rels:
        token = str(rel).replace("\\", "/")
        if token.startswith("/") or ".." in Path(token).parts:
            raise IntegrityError(f"public artifact path escapes the pack: {rel}")
        if token not in cleaned:
            cleaned.append(token)
    return cleaned


def copy_declared_public_inputs(public_dir: Path, workspace: Path, public_spec) -> list[str]:
    """Copy only the declared initial/reference evidence plus public protocol files."""
    import shutil as _shutil

    public_dir = Path(public_dir)
    workspace = Path(workspace)
    (workspace / "evidence").mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in declared_public_evidence_names(public_spec):
        src = public_dir / "evidence" / name
        dest = workspace / "evidence" / name
        _shutil.copyfile(src, dest)
        copied.append(f"evidence/{name}")
    for rel in declared_public_relative_paths(public_spec):
        if rel.startswith("evidence/"):
            continue
        src = (public_dir / rel).resolve()
        if not str(src).startswith(str(public_dir.resolve())):
            raise IntegrityError(f"public artifact escapes the pack: {rel}")
        if src.is_symlink():
            raise IntegrityError(f"public artifact symlink is not allowed: {rel}")
        if src.exists():
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copyfile(src, dest)
            copied.append(rel)
    return copied


def _reject_treated_in_controls(public_dir: Path, private_spec) -> None:
    t_ids = set(getattr(private_spec, "target_condition_ids", None) or [])
    q_ids = set(getattr(private_spec, "queryable_condition_ids", None) or [])
    forbidden = t_ids | q_ids
    if not forbidden:
        return
    ctrl = public_dir / "evidence" / "ev_controls_001.h5ad"
    if not ctrl.exists():
        return
    import anndata as ad

    obs = ad.read_h5ad(ctrl).obs
    if "condition_id" not in obs.columns:
        return
    leaked = set(obs["condition_id"].astype(str)) & forbidden
    if leaked:
        raise IntegrityError(f"control artifact contains Q/T treated conditions: {sorted(leaked)[:8]}")
