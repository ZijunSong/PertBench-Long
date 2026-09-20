"""P2 temporal interpolation adapter/validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pertbench_long.errors import UnsupportedProfile


def audit_temporal_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    csv = root / "GSM3770930_A549_lognorm_scale_hvg3000.csv"
    annot = root / "GSM3770930_A549_cell_annotate.txt"
    if not csv.exists():
        return {"status": "blocked", "reason": "temporal_csv_missing", "path": str(root)}
    return {
        "status": "blocked",
        "reason": "matrix_kind_lognorm_scale_hvg3000_cannot_use_pbmc_proxy_tau",
        "files": [str(csv), str(annot) if annot.exists() else None],
        "protocol_if_enabled": "temporal_interpolation_v1",
        "proposed_split": {"O": ["0h", "10h"], "Q": ["2h", "8h"], "T": ["4h", "6h"], "B": 1},
        "warning": "This is retrospective interpolation, not future forecasting. 0h is not automatically a matched untreated control.",
        "processing": "filename indicates log-normalization, scaling, and HVG=3000; full-data feature selection cannot be undone.",
    }


def build_temporal_episode(*_a, **_k) -> None:
    raise UnsupportedProfile("temporal official labels are blocked until scale/HVG provenance is resolved")
