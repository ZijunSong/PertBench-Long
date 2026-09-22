"""Column alias maps. Agent-facing names never include MOA answer tokens unless the task profile says so."""

from __future__ import annotations

from typing import Mapping

DEFAULT_ALIASES = {
    "Cell.Type": "cell_type",
    "celltype": "cell_type",
    "cell_type": "cell_type",
    "perturbation_status": "perturbation",
    "perturbation": "perturbation",
    "perturbation_id": "perturbation",
    "stim": "perturbation",
    "condition": "perturbation",
    "treatment_time": "time",
    "time": "time",
    "dose_value": "dose",
    "dose": "dose",
    "donor": "donor",
    "donor_id": "donor",
    "sample": "sample_id",
    "sample_id": "sample_id",
    "species": "species",
    "context_id": "context_id",
    "replicate_id": "replicate_id",
    "batch_id": "batch_id",
    "is_control": "is_control",
    "control": "is_control",
    "dose_unit": "dose_unit",
    "time_unit": "time_unit",
}


def resolve_column(columns: Mapping[str, int] | list[str], aliases: Mapping[str, str], canonical: str) -> str | None:
    cols = list(columns) if not isinstance(columns, Mapping) else list(columns.keys())
    inverse = {}
    for src, dst in aliases.items():
        inverse.setdefault(dst, []).append(src)
    candidates = inverse.get(canonical, []) + [canonical]
    for name in candidates:
        if name in cols:
            return name
    return None
