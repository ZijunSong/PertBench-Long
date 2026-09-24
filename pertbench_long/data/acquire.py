"""Versioned source fetch. Missing URLs are reported, never invented."""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from pertbench_long.data.adapters.base import load_acquisition_manifest
from pertbench_long.data.data_root import raw_dir, resolve_data_root
from pertbench_long.errors import ConfigError, IntegrityError
from pertbench_long.hashes import sha256_file

REASON_MISSING_DIR = "missing_directory"
REASON_MISSING_SOURCE = "missing_source_files"
REASON_HASH_MISMATCH = "hash_mismatch"
REASON_DOWNLOAD_INCOMPLETE = "download_incomplete"
REASON_URL_UNAVAILABLE = "url_unavailable"
REASON_OFFLINE_REUSED = "offline_reused"
REASON_USER_MUST_OBTAIN = "user_must_obtain"
REASON_READY = "ready"


@dataclass
class SourceFile:
    name: str
    role: str
    url: str | None = None
    sha256: str | None = None
    expected_bytes: int | None = None
    format: str = "unknown"
    status: str = "user_must_obtain"
    accession: str | None = None
    author_entry: str | None = None
    note: str = ""
    requirement: str = "required"
    group: str = ""


@dataclass
class FetchReport:
    dataset_id: str
    release_id: str
    dest: Path
    status: str
    reason: str
    files: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "release_id": self.release_id,
            "dest": str(self.dest),
            "status": self.status,
            "reason": self.reason,
            "files": self.files,
            "missing": self.missing,
            "notes": self.notes,
        }


def load_source_files(manifest_path: Path | str) -> tuple[str, str, list[SourceFile], dict[str, Any]]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = load_acquisition_manifest(path)
    items: list[SourceFile] = []
    for raw in payload.get("raw_sources") or payload.get("sources") or []:
        if not isinstance(raw, Mapping):
            raise ConfigError("source entries must be objects")
        url = raw.get("url")
        items.append(
            SourceFile(
                name=str(raw.get("name") or ""),
                role=str(raw.get("role") or "unknown"),
                url=None if url in {None, "", "null"} else str(url),
                sha256=None if not raw.get("sha256") else str(raw["sha256"]),
                expected_bytes=int(raw["bytes"]) if raw.get("bytes") is not None else None,
                format=str(raw.get("format") or "unknown"),
                status=str(raw.get("status") or "user_must_obtain"),
                accession=None if not raw.get("accession") else str(raw["accession"]),
                author_entry=None if not raw.get("author_entry") else str(raw["author_entry"]),
                note=str(raw.get("note") or ""),
                requirement=str(raw.get("requirement") or ("optional" if str(raw.get("status") or "") == "optional" else "required")),
                group=str(raw.get("group") or ""),
            )
        )
    return spec.dataset_id, spec.release_id, items, payload


def _verify_existing(path: Path, source: SourceFile) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": source.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "sha256_verified_by": "local_compute",
    }
    if source.expected_bytes is not None and info["bytes"] != source.expected_bytes:
        raise IntegrityError(f"{REASON_DOWNLOAD_INCOMPLETE}: {source.name} has {info['bytes']} bytes, expected {source.expected_bytes}")
    if source.sha256:
        if info["sha256"] != source.sha256:
            raise IntegrityError(f"{REASON_HASH_MISMATCH}: {source.name}")
        info["sha256_verified_by"] = "declared_manifest"
    else:
        info["note"] = "hash computed locally; not independently declared in the release manifest"
    return info


def atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    tmp.replace(dest)


def atomic_download(url: str, dest: Path, *, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        src = Path(parsed.path if parsed.scheme == "file" else url)
        if not src.exists():
            raise ConfigError(f"{REASON_MISSING_SOURCE}: local source {src} does not exist")
        atomic_copy(src, dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "pertbench-long-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise ConfigError(f"{REASON_URL_UNAVAILABLE}: {exc}") from exc
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    tmp.replace(dest)


def fetch_dataset(
    dataset_id: str,
    *,
    data_root: Path | str | None = None,
    manifest_path: Path | str | None = None,
    offline: bool = False,
    timeout: int = 60,
) -> FetchReport:
    root = resolve_data_root(data_root)
    if manifest_path is None:
        repo = Path(__file__).resolve().parents[2]
        manifest_path = repo / "data" / "acquisition" / f"{dataset_id}_v1.json"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ConfigError(f"acquisition manifest missing: {manifest_path}")
    ds, release_id, sources, _payload = load_source_files(manifest_path)
    dest = raw_dir(root, ds or dataset_id, release_id or "v1")
    dest.mkdir(parents=True, exist_ok=True)
    notes = [
        "URLs are never guessed. A null url means the user must place the author file offline.",
        "Existing files with a different declared hash are not overwritten.",
    ]
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    if not sources:
        notes.append("manifest has no raw_sources; only a prepared-matrix import contract is defined")
    for source in sources:
        if not source.name:
            raise ConfigError("source entry is missing name")
        path = dest / source.name
        if path.exists():
            try:
                info = _verify_existing(path, source)
            except IntegrityError:
                raise
            info["status"] = REASON_OFFLINE_REUSED
            files.append(info)
            continue
        if source.requirement == "optional":
            files.append({"name": source.name, "role": source.role, "status": "optional_absent", "note": source.note})
            continue
        if source.requirement == "one_of":
            files.append({"name": source.name, "role": source.role, "status": "one_of_absent", "group": source.group})
            continue
        if offline or not source.url:
            missing.append(source.name)
            files.append(
                {
                    "name": source.name,
                    "role": source.role,
                    "status": REASON_USER_MUST_OBTAIN,
                    "url": source.url,
                    "accession": source.accession,
                    "author_entry": source.author_entry,
                    "note": source.note or "place this file in the raw directory; no download URL is locked",
                }
            )
            continue
        try:
            atomic_download(source.url, path, timeout=timeout)
            info = _verify_existing(path, source)
            info["status"] = "downloaded"
            files.append(info)
        except IntegrityError:
            if path.exists():
                path.unlink()
            raise
    groups: dict[str, list[SourceFile]] = {}
    for source in sources:
        if source.requirement == "one_of" and source.group:
            groups.setdefault(source.group, []).append(source)
    for group, members in groups.items():
        if not any((dest / member.name).exists() for member in members):
            missing.append(f"one_of:{group}")
    if missing:
        return FetchReport(
            dataset_id=ds or dataset_id,
            release_id=release_id,
            dest=dest,
            status="blocked",
            reason=REASON_USER_MUST_OBTAIN if any(not s.url for s in sources) else REASON_MISSING_SOURCE,
            files=files,
            missing=missing,
            notes=notes,
        )
    return FetchReport(
        dataset_id=ds or dataset_id,
        release_id=release_id,
        dest=dest,
        status="ok",
        reason=REASON_READY,
        files=files,
        notes=notes,
    )
