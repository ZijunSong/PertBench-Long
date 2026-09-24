"""Shared acquisition-manifest loading. Paths are user-supplied; /data/ppnm is never implied."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pertbench_long.errors import ConfigError, IntegrityError, SchemaError
from pertbench_long.hashes import sha256_file


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    accession: str
    release_id: str
    license: str
    redistribution: str
    required_files: tuple[str, ...]
    field_map: dict[str, str]
    software: dict[str, Any]
    notes: tuple[str, ...]
    file_hashes: dict[str, str] = field(default_factory=dict)


def load_acquisition_manifest(path: Path | str) -> SourceSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SchemaError("acquisition manifest must be an object")
    required = payload.get("required_files") or []
    hashes = {str(k): str(v) for k, v in dict(payload.get("file_hashes") or {}).items() if v}
    return SourceSpec(
        dataset_id=str(payload.get("dataset_id") or ""),
        accession=str(payload.get("accession") or ""),
        release_id=str(payload.get("release_id") or ""),
        license=str(payload.get("license") or "unknown"),
        redistribution=str(payload.get("redistribution") or "user_must_obtain_source"),
        required_files=tuple(str(x) for x in required),
        field_map=dict(payload.get("field_map") or {}),
        software=dict(payload.get("software") or {}),
        notes=tuple(str(x) for x in (payload.get("notes") or [])),
        file_hashes=hashes,
    )


def resolve_source_dir(data_dir: Path | str | None, *, dataset_id: str) -> Path:
    if data_dir is None:
        raise ConfigError(f"{dataset_id} requires an explicit --data-dir; author paths are not used")
    path = Path(data_dir)
    if not path.exists():
        raise ConfigError(f"{dataset_id} data directory does not exist: {path}")
    return path


def require_files(root: Path, names: tuple[str, ...], *, hashes: Mapping[str, str] | None = None) -> dict[str, Path]:
    found: dict[str, Path] = {}
    missing = []
    for name in names:
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        if hashes and name in hashes:
            digest = sha256_file(path)
            if digest != hashes[name]:
                raise IntegrityError(f"source hash mismatch for {name}")
        found[name] = path
    if missing:
        raise ConfigError(f"missing required source files: {missing}")
    return found
