"""Replay visible information from events.jsonl. Does not claim bitwise LLM reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_events(run_dir: Path) -> Iterator[dict[str, Any]]:
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return
        yield  # pragma: no cover
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def replay_visible_trace(run_dir: Path) -> dict[str, Any]:
    events = list(iter_events(run_dir))
    evidence_versions = sorted({int(e.get("evidence_version", 0)) for e in events})
    purchases = [e for e in events if e.get("action") == "request_experiment"]
    return {
        "n_events": len(events),
        "evidence_versions": evidence_versions,
        "purchases": [
            {
                "experiment_id": e.get("experiment_id") or (e.get("result") or {}).get("experiment_id"),
                "charged": (e.get("result") or {}).get("charged_credits"),
                "evidence_version": e.get("evidence_version"),
            }
            for e in purchases
        ],
        "note": "Replay restores visible artifacts and scores; external LLM trajectories are not bitwise reproducible.",
    }
