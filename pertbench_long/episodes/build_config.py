"""Strict YAML schema for real T1/T2/T3 builds. Unknown fields are errors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from pertbench_long.errors import ConfigError

KNOWN_BUILD_FIELDS = frozenset(
    {
        "fixture",
        "output",
        "pbmc_csv_dir",
        "matrix_kind",
        "study",
        "conflict_policy",
        "o_types",
        "q_types",
        "t_types",
        "n_panel",
        "data_release_id",
        "experimental_budget",
        "protocol",
        "label_profile",
        "dataset",
        "adapter",
        "data_dir",
        "data_root",
        "source_manifest",
        "manifest",
        "matrix",
        "episode_id",
        "partition",
        "split_manifest",
        "split_variant",
        "context_ids",
        "source_context",
        "target_contexts",
        "min_candidates",
        "min_targets",
        "min_cells",
        "seed",
        "reference_policy",
        "resource_profile",
        "requested_track",
        "label_estimand",
        "a_family",
        "scoring_scale_hash",
        "scoring_scale",
        "scoring_calibration",
        "input_profile",
        "synthetic",
        "development_episodes",
        "processing_history",
        "scoring_track",
        "notes",
        "config_version",
    }
)

KNOWN_MATRIX_FIELDS = frozenset({"file", "layer", "matrix_kind", "require_verified_provenance"})


ALLOWED_ESTIMANDS = frozenset({"auto", "replicate_equal_weight", "cell_weighted_descriptive", "donor_equal_weight"})
ALLOWED_REFERENCE_POLICIES = frozenset({"matched_vehicle_v1", "matched_ntc_v1", "target_control_available_v1"})
ALLOWED_TRACKS = frozenset({"pilot", "diagnostic", "official"})


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ConfigError(f"unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def validate_build_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(cfg) - KNOWN_BUILD_FIELDS)
    if unknown:
        raise ConfigError(f"unknown build config fields: {unknown}")
    payload = expand_env(dict(cfg))
    matrix = payload.get("matrix")
    if matrix is not None:
        if not isinstance(matrix, Mapping):
            raise ConfigError("matrix must be a mapping")
        extra = sorted(set(matrix) - KNOWN_MATRIX_FIELDS)
        if extra:
            raise ConfigError(f"unknown matrix fields: {extra}")
    estimand = payload.get("label_estimand")
    if estimand is not None and estimand not in ALLOWED_ESTIMANDS:
        raise ConfigError(f"unknown label_estimand {estimand!r}")
    policy = payload.get("reference_policy")
    if policy is not None and policy not in ALLOWED_REFERENCE_POLICIES:
        raise ConfigError(f"unknown reference_policy {policy!r}")
    track = payload.get("requested_track")
    if track is not None and track not in ALLOWED_TRACKS:
        raise ConfigError(f"unknown requested_track {track!r}")
    return payload


def resolve_relative(raw: str | Path | None, *, config_path: Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if config_path is not None:
        return (Path(config_path).parent / path).resolve()
    return path.resolve()
