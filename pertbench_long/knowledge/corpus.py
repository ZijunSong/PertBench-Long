"""Frozen corpus retrieval stub. v1 scoring does not require retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CorpusSnapshot:
    name: str
    version: str
    license: str | None
    content_hash: str | None
    filtered_answers: bool
    notes: list[str]


def default_disabled_corpus() -> CorpusSnapshot:
    return CorpusSnapshot(
        name="none",
        version="v1_no_retrieval",
        license=None,
        content_hash=None,
        filtered_answers=True,
        notes=["v1 completes without literature retrieval", "no paper-answer shortcut is provided"],
    )


def search(query: str, corpus: CorpusSnapshot | None = None) -> list[dict[str, Any]]:
    snap = corpus or default_disabled_corpus()
    if snap.name == "none":
        return []
    return []
