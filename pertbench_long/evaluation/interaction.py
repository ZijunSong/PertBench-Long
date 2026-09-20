"""Development diagnostic: does extra evidence change predictions? Not an official score."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_interaction_value_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Interaction value report",
        "",
        "This report is a development diagnostic. It is not an official scientific score.",
        "",
        f"- no-query score: {payload.get('no_query')}",
        f"- random-query score: {payload.get('random_query')}",
        f"- fixed-order score: {payload.get('fixed_order')}",
        f"- adaptive/scripted score: {payload.get('adaptive')}",
        f"- S(B)-S(0): {payload.get('delta')}",
        "",
        payload.get("conclusion", "If extra observations barely change scores, the candidate pool may not support sequential experimental choice."),
        "",
        "Tool-call counts are diagnostic only and are not rewarded.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
