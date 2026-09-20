"""P2 MOA adapter/validator. Official benchmark remains blocked until isolation and data gates pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pertbench_long.errors import UnsupportedProfile


def audit_moa_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {"status": "blocked", "reason": "moa_root_missing", "path": str(root)}
    files = list(root.rglob("*_train*.csv")) + list(root.rglob("*_train*.h5ad"))
    names = [p.name for p in files]
    leaky = [n for n in names if "moa" in n.lower() or any(tok in n for tok in ("JAKSTAT", "signaling", "Antioxidant"))]
    return {
        "status": "unsupported",
        "n_files": len(files),
        "filename_leaks_moa_label": bool(leaky),
        "note": "MOA class in filenames must not be Agent-visible if the task is to recover that class. Official MOA track is not enabled in v0.1.",
        "assay": "unverified_single_cell_or_bulk",
        "donor": None,
        "hypothesis_discrimination": "not_enabled",
    }


def build_moa_episode(*_a, **_k) -> None:
    raise UnsupportedProfile("MOA episodes are specified but not enabled: data/unit/leak gates have not passed")
