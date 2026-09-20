from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pertbench_long.evaluation.labels import build_effect_proxy_labels, build_replicate_de_labels
from pertbench_long.evaluation.scoring import aubc, direction_score_from_brier, make_prediction_frame, multiclass_brier, pearson_or_na, score_predictions
from pertbench_long.evaluation.submission import SubmissionInvalid, validate_predictions
from pertbench_long.errors import LabelBuildError, SubmissionInvalid as SI2
from pertbench_long.evaluation.outcomes import official_aubc, official_direction_score, summarize_runs


def _labels():
    genes = ["G1", "G2", "G3"]
    stim = np.array([[1.0, 0.0, -1.0]])
    ctrl = np.array([[0.0, 0.0, 0.0]])
    return build_effect_proxy_labels(gene_ids=genes, targets={"t_01": (stim, ctrl)})


def test_t12_hand_computed_probabilities():
    labels = _labels()
    genes = ["G1", "G2", "G3"]
    perfect = make_prediction_frame(
        ["t_01"],
        genes,
        effects=np.array([1.0, 0.0, -1.0]),
        probs=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float),
    )
    wrong = make_prediction_frame(
        ["t_01"],
        genes,
        effects=np.array([-1.0, 1.0, 1.0]),
        probs=np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
    )
    uniform = make_prediction_frame(
        ["t_01"],
        genes,
        effects=np.zeros(3),
        probs=np.full((3, 3), 1 / 3),
    )
    s_perfect = score_predictions(perfect, labels.frame, target_ids=["t_01"], gene_ids=genes)
    s_wrong = score_predictions(wrong, labels.frame, target_ids=["t_01"], gene_ids=genes)
    s_uni = score_predictions(uniform, labels.frame, target_ids=["t_01"], gene_ids=genes)
    assert s_perfect["direction_score"] == pytest.approx(1.0)
    assert s_wrong["direction_score"] == pytest.approx(0.0)
    assert s_uni["direction_score"] == pytest.approx(2 / 3)
    corr, reason = pearson_or_na(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert corr is None and reason == "constant_vector"


def test_t11_gene_order_and_illegal_rows():
    labels = _labels()
    genes = ["G1", "G2", "G3"]
    base = make_prediction_frame(
        ["t_01"],
        genes,
        effects=np.array([1.0, 0.0, -1.0]),
        probs=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float),
    )
    shuffled_genes = ["G3", "G1", "G2"]
    shuffled = base.sort_values("gene_id", ascending=False).reset_index(drop=True)
    s1 = score_predictions(base, labels.frame, target_ids=["t_01"], gene_ids=genes)
    s2 = score_predictions(shuffled, labels.frame, target_ids=["t_01"], gene_ids=genes)
    assert s1["direction_score"] == pytest.approx(s2["direction_score"])
    missing = base.iloc[:2]
    with pytest.raises((SubmissionInvalid, SI2)):
        validate_predictions(missing, target_ids=["t_01"], gene_ids=genes)
    dup = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    with pytest.raises((SubmissionInvalid, SI2)):
        validate_predictions(dup, target_ids=["t_01"], gene_ids=genes)
    bad = base.copy()
    bad.loc[0, "p_down"] = 0.5
    with pytest.raises((SubmissionInvalid, SI2)):
        validate_predictions(bad, target_ids=["t_01"], gene_ids=genes)


def test_t13_no_fake_replicate_de():
    with pytest.raises(LabelBuildError):
        build_replicate_de_labels(donors=None, counts_available=False)
    labels = _labels()
    assert "effect_direction_proxy" in labels.frame.columns
    assert "significant" not in labels.frame.columns
    assert any("not significant DE" in n or "not significant" in n for n in labels.notes)


def test_t14_aubc_and_negative_gain():
    assert aubc({0: 0.5}, 0) == pytest.approx(0.5)
    curve = aubc({0: 0.8, 1: 0.6, 2: 0.4}, 2)
    assert curve == pytest.approx((0.5 * 0.8 + 0.6 + 0.5 * 0.4) / 2)
    assert (0.4 - 0.8) < 0


def test_t17_failure_denominators():
    runs = [
        {"run_id": "a", "outcome": "completed", "direction_score": 0.9},
        {"run_id": "b", "outcome": "invalid_submission", "direction_score": 0.1},
        {"run_id": "c", "outcome": "infra_error", "reason": "disk"},
        {"run_id": "d", "outcome": "invalid_episode", "reason": "qc"},
    ]
    summary = summarize_runs(runs)
    assert summary["science_denominator"] == 2
    assert summary["mean_direction_score_including_failures"] == pytest.approx(0.45)
    assert official_direction_score("invalid_submission", 0.9) == 0.0
    assert official_aubc("invalid_submission", 0.9) == 0.0
    assert summary["nanmean_not_used"] is True
