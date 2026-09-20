from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pertbench_long.data.adapter import import_tables
from pertbench_long.data.preprocess import detect_repeated_log1p, to_effect_space


def test_import_dedup_and_conflict(tmp_path: Path):
    genes = ["G1", "G2"]
    a = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=["c1-control", "c2-stimulated"], columns=genes)
    b = pd.DataFrame([[0.1, 0.2], [9.0, 9.0]], index=["c1-control", "c3-stimulated"], columns=genes)
    pa = tmp_path / "task1_train_CD4T_exp.csv"
    pb = tmp_path / "task1_valid_CD4T_exp.csv"
    a.to_csv(pa)
    b.to_csv(pb)
    store = import_tables([pa, pb], study="unit", declared_matrix_kind="log1p")
    assert store.summary.n_deduplicated == 1
    assert store.summary.n_conflicts == 0
    # conflict row
    c = pd.DataFrame([[0.5, 0.6]], index=["c1-control"], columns=genes)
    pc = tmp_path / "task1_valid_CD4T_conflict.csv"
    c.to_csv(pc)
    store2 = import_tables(
        [pa, pc],
        study="unit2",
        declared_matrix_kind="log1p",
        conflict_policy="diagnostic_keep_first",
    )
    assert store2.summary.n_conflicts == 1
    assert store.summary.matrix_kind == "log1p"
    assert store.summary.gene_order_hash
    from pertbench_long.errors import SchemaError
    import pytest

    with pytest.raises(SchemaError):
        import_tables([pa, pc], study="unit3", declared_matrix_kind="log1p", conflict_policy="fail_closed")


def test_repeated_log1p_detected():
    x = np.log1p(np.array([[1.0, 2.0], [0.0, 3.0]]))
    assert detect_repeated_log1p(x, "log1p") is True
    y = to_effect_space(x, "log1p")
    assert np.allclose(x, y)
