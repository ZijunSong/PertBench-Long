from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from pertbench_long.cli import main


def test_import_does_not_load_torch():
    import pertbench_long

    src = Path(pertbench_long.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "import anndata" not in src


def test_cli_help_and_list():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    rc = main(["list"])
    assert rc == 0


def test_t19_legacy_pertdiffbench_cli_still_present():
    exe = Path("/data/ppnm/miniconda3/envs/pertdiffbench/bin/pertdiffbench")
    if not exe.exists():
        return
    proc = subprocess.run([str(exe), "list-tasks"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "known_condition" in proc.stdout
