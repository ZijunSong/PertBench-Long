"""Official eligibility is a checked artifact chain, not a config flag."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pertbench_long.errors import ConfigError, IntegrityError, UnsupportedProfile
from pertbench_long.hashes import sha256_file, sha256_json

PREPARED_PROVENANCE_VERSION = "prepared_v1"
PREPARED_MANIFEST_VERSION = "prepared_manifest_v1"
CALIBRATION_VERSION = "scoring_calibration_v1"


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ConfigError(f"{path.name} is empty or not an object")
    return payload


def verify_prepared_provenance(data_dir: Path) -> dict[str, Any]:
    root = Path(data_dir)
    prov_path = root / "provenance.json"
    man_path = root / "prepared_manifest.json"
    missing = [name for name in ("provenance.json", "prepared_manifest.json", "matrix.h5ad", "conditions.parquet", "qc_report.json") if not (root / name).exists()]
    if missing:
        raise UnsupportedProfile(f"official evidence missing prepared artifacts: {missing}")
    provenance = _load(prov_path)
    if provenance.get("provenance_version") != PREPARED_PROVENANCE_VERSION:
        raise UnsupportedProfile("provenance_version is missing or unsupported; an empty object is not verified provenance")
    sources = provenance.get("source_files") or {}
    if not sources:
        raise UnsupportedProfile("provenance has no source file hashes")
    manifest = _load(man_path)
    if manifest.get("manifest_version") != PREPARED_MANIFEST_VERSION:
        raise UnsupportedProfile("prepared_manifest version is unsupported")
    if manifest.get("provenance_sha256") != sha256_file(prov_path):
        raise IntegrityError("prepared_manifest provenance hash does not match provenance.json")
    outputs = dict(manifest.get("outputs") or {})
    for name in ("matrix.h5ad", "conditions.parquet", "qc_report.json"):
        if outputs.get(name) != sha256_file(root / name):
            raise IntegrityError(f"prepared output hash mismatch for {name}")
    qc = _load(root / "qc_report.json")
    if not qc.get("status"):
        raise UnsupportedProfile("qc_report is empty")
    return {"ok": True, "provenance": provenance, "prepared_manifest": manifest, "qc": qc}


def verify_calibration(path: Path, *, a_family: float, label_profile: str) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("calibration_version") != CALIBRATION_VERSION:
        raise UnsupportedProfile("scoring calibration artifact is missing calibration_version")
    if payload.get("label_profile") != label_profile:
        raise UnsupportedProfile("scoring calibration label_profile does not match the episode")
    if float(payload.get("a_family")) != float(a_family):
        raise UnsupportedProfile("scoring calibration a_family does not match the build")
    if payload.get("fit_on") not in {"development", "fixture"}:
        raise UnsupportedProfile("scoring calibration must declare fit_on=development or fixture")
    digest = sha256_file(path)
    return {"ok": True, "sha256": digest, "payload": payload, "scale_source": "frozen_calibration_artifact"}


def verify_official_evidence(
    data_dir: Path,
    *,
    calibration_path: Path | None,
    a_family: float,
    label_profile: str,
    split_manifest: Mapping[str, Any] | None,
    declared_scale_hash: str | None,
) -> dict[str, Any]:
    prepared = verify_prepared_provenance(data_dir)
    if calibration_path is None:
        raise UnsupportedProfile("official track requires a scoring calibration file, not a bare hash string")
    calibration = verify_calibration(calibration_path, a_family=a_family, label_profile=label_profile)
    if declared_scale_hash and declared_scale_hash != calibration["sha256"]:
        raise IntegrityError("scoring_scale_hash does not match the calibration file")
    if not split_manifest or not split_manifest.get("partition"):
        raise UnsupportedProfile("official track requires a split manifest with a partition")
    return {
        "ok": True,
        "source_verified": bool(prepared["provenance"].get("source_lock_status") == "locked"),
        "provenance_verified": True,
        "labels_validated": prepared["qc"].get("status") == "ok",
        "split_audited": True,
        "scoring_scale_hash": calibration["sha256"],
        "scale_source": calibration["scale_source"],
        "prepared_manifest_sha256": sha256_json(prepared["prepared_manifest"]),
    }
