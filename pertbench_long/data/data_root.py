"""Resolve DATA_ROOT. Author paths and /data/ppnm are never implied."""

from __future__ import annotations

import os
from pathlib import Path

from pertbench_long.errors import ConfigError

ENV_DATA_ROOT = "PERTBENCH_DATA_ROOT"


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    raise ConfigError("data root is required (--data-root or PERTBENCH_DATA_ROOT)")


def raw_dir(data_root: Path, dataset_id: str, release_id: str) -> Path:
    return Path(data_root) / "raw" / dataset_id / release_id


def prepared_dir(data_root: Path, dataset_id: str, version: str = "v1") -> Path:
    return Path(data_root) / "prepared" / dataset_id / version


def episodes_dir(data_root: Path) -> Path:
    return Path(data_root) / "episodes"
