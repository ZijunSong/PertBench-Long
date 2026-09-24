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
    if qc.get("status") != "ok":
        raise UnsupportedProfile("qc_report status is not ok")
    return {"ok": True, "provenance": provenance, "prepared_manifest": manifest, "qc": qc}


def _sha256_ok(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def verify_source_files(provenance: Mapping[str, Any]) -> None:
    """A locked status string is not source verification."""
    lock = provenance.get("source_lock")
    if not isinstance(lock, dict) or lock.get("lock_version") != "source_lock_v1":
        raise UnsupportedProfile("source_lock_status alone cannot verify a source; a source_lock_v1 record is required")
    if lock.get("sha256_verified_by") not in {"publisher_checksum", "maintainer_computed_lock"}:
        raise UnsupportedProfile("source lock must say whether the checksum came from the publisher or the maintainer")
    if not lock.get("accession") or not lock.get("release_id"):
        raise UnsupportedProfile("source lock is missing accession or release_id")
    files = dict(provenance.get("source_files") or {})
    if not files:
        raise UnsupportedProfile("provenance has no source file hashes")
    for name, meta in files.items():
        if not isinstance(meta, Mapping) or not _sha256_ok(meta.get("sha256")):
            raise IntegrityError(f"source hash for {name} is missing or not a sha256")
        path = Path(str(meta.get("path") or ""))
        if not path.is_file():
            raise IntegrityError(f"locked source file does not exist: {name}")
        if sha256_file(path) != str(meta["sha256"]):
            raise IntegrityError(f"source hash mismatch for {name}")


def verify_split_manifest(split_manifest: Mapping[str, Any] | None, *, partition: str) -> None:
    if not split_manifest:
        raise UnsupportedProfile("official track requires a split manifest")
    if split_manifest.get("manifest_version") != "split_manifest_v1":
        raise UnsupportedProfile("split manifest version is unsupported")
    if str(split_manifest.get("partition") or "") != str(partition):
        raise UnsupportedProfile(
            f"split manifest partition {split_manifest.get('partition')!r} does not match build partition {partition!r}"
        )
    if split_manifest.get("audit_status") != "ok":
        raise UnsupportedProfile("split manifest audit_status is not ok")


def verify_calibration(
    path: Path,
    *,
    a_family: float,
    label_profile: str,
    protocol: str | None = None,
    effect_unit: str | None = None,
    estimand: str | None = None,
    official: bool = False,
) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("calibration_version") != CALIBRATION_VERSION:
        raise UnsupportedProfile("scoring calibration artifact is missing calibration_version")
    if payload.get("label_profile") != label_profile:
        raise UnsupportedProfile("scoring calibration label_profile does not match the episode")
    if float(payload.get("a_family")) != float(a_family):
        raise UnsupportedProfile("scoring calibration a_family does not match the build")
    if official and payload.get("fit_on") != "development":
        raise UnsupportedProfile("fixture calibration cannot grant official scoring")
    if not official and payload.get("fit_on") not in {"development", "fixture"}:
        raise UnsupportedProfile("scoring calibration must declare fit_on=development or fixture")
    if protocol and payload.get("protocol") not in {None, protocol}:
        raise UnsupportedProfile("scoring calibration protocol does not match the build")
    if effect_unit and payload.get("effect_unit") not in {None, effect_unit}:
        raise UnsupportedProfile("scoring calibration effect_unit does not match the build")
    if estimand and estimand != "auto" and payload.get("estimand") not in {None, estimand}:
        raise UnsupportedProfile("scoring calibration estimand does not match the build")
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
    partition: str | None = None,
    protocol: str | None = None,
    effect_unit: str = "lognorm_cellmean_delta",
    estimand: str | None = None,
) -> dict[str, Any]:
    prepared = verify_prepared_provenance(data_dir)
    if prepared["qc"].get("status") != "ok":
        raise UnsupportedProfile("qc_report status is not ok")
    verify_source_files(prepared["provenance"])
    verify_split_manifest(split_manifest, partition=str(partition or ""))
    if calibration_path is None:
        raise UnsupportedProfile("official track requires a scoring calibration file, not a bare hash string")
    calibration = verify_calibration(
        calibration_path,
        a_family=a_family,
        label_profile=label_profile,
        protocol=protocol,
        effect_unit=effect_unit,
        estimand=estimand,
        official=True,
    )
    if declared_scale_hash and declared_scale_hash != calibration["sha256"]:
        raise IntegrityError("scoring_scale_hash does not match the calibration file")
    return {
        "ok": True,
        "source_verified": True,
        "provenance_verified": True,
        "labels_validated": True,
        "split_audited": True,
        "scoring_scale_hash": calibration["sha256"],
        "scale_source": calibration["scale_source"],
        "prepared_manifest_sha256": sha256_json(prepared["prepared_manifest"]),
    }
