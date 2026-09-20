from __future__ import annotations

from pathlib import Path

import pytest

from pertbench_long.episodes.builder import build_synthetic_episode


@pytest.fixture
def synthetic_episode(tmp_path: Path) -> dict:
    return build_synthetic_episode(tmp_path / "episode")
