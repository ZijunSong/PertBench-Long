"""Content hashing helpers. Never log secrets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def gene_order_hash(gene_ids: Sequence[str]) -> str:
    return sha256_text("\n".join(str(g) for g in gene_ids))


def fingerprint_files(paths: Iterable[Path | str]) -> str:
    items = []
    for path in paths:
        p = Path(path)
        items.append({"name": p.name, "sha256": sha256_file(p) if p.exists() else None, "size": p.stat().st_size if p.exists() else 0})
    return sha256_json(items)


def mapping_fingerprint(mapping: Mapping[str, Any]) -> str:
    return sha256_json(dict(mapping))
