"""Structured numeric scoring. Gene order does not affect the score."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from pertbench_long.evaluation.labels import DIRECTIONS
from pertbench_long.evaluation.submission import validate_predictions

EPS = 1e-12


def _one_hot(labels: Sequence[str]) -> np.ndarray:
    index = {k: i for i, k in enumerate(DIRECTIONS)}
    out = np.zeros((len(labels), 3), dtype=np.float64)
    for row, lab in enumerate(labels):
        out[row, index[str(lab)]] = 1.0
    return out


def multiclass_brier(prob: np.ndarray, one_hot: np.ndarray) -> float:
    return float(np.mean(np.sum((prob - one_hot) ** 2, axis=1)))


def direction_score_from_brier(brier: float) -> float:
    return float(1.0 - brier / 2.0)


def pearson_or_na(x: np.ndarray, y: np.ndarray) -> tuple[Optional[float], Optional[str]]:
    if x.size < 2 or y.size < 2:
        return None, "insufficient_length"
    if np.std(x) < EPS or np.std(y) < EPS:
        return None, "constant_vector"
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return None, "non_finite"
    return float(corr), None


def macro_f1(pred: Sequence[str], truth: Sequence[str]) -> float:
    f1s = []
    for cls in DIRECTIONS:
        tp = sum(p == cls and t == cls for p, t in zip(pred, truth))
        fp = sum(p == cls and t != cls for p, t in zip(pred, truth))
        fn = sum(p != cls and t == cls for p, t in zip(pred, truth))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def class_recall(pred: Sequence[str], truth: Sequence[str]) -> dict[str, float]:
    out = {}
    for cls in DIRECTIONS:
        denom = sum(t == cls for t in truth)
        if denom == 0:
            out[cls] = float("nan")
        else:
            out[cls] = sum(p == cls and t == cls for p, t in zip(pred, truth)) / denom
    return out


def score_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target_ids: Sequence[str],
    gene_ids: Sequence[str],
) -> dict[str, Any]:
    pred = validate_predictions(predictions, target_ids=target_ids, gene_ids=gene_ids)
    lab = labels.copy()
    lab_key = lab["target_id"].astype(str) + "\t" + lab["gene_id"].astype(str)
    if lab_key.duplicated().any():
        raise ValueError("duplicate target×gene labels; conflict is not dropped")
    pred_key = pred["target_id"].astype(str) + "\t" + pred["gene_id"].astype(str)
    lab = lab.assign(_k=lab_key)
    merged = pred.assign(_k=pred_key).merge(lab[["_k", "effect_direction_proxy", "reference_effect"]], on="_k", how="left")
    if merged["effect_direction_proxy"].isna().any():
        raise ValueError("labels missing for required prediction rows")
    per_target = []
    for target in target_ids:
        part = merged[merged["target_id"] == target]
        prob = part.loc[:, ["p_down", "p_neutral", "p_up"]].to_numpy(dtype=np.float64)
        truth = part["effect_direction_proxy"].astype(str).tolist()
        one_hot = _one_hot(truth)
        brier = multiclass_brier(prob, one_hot)
        dscore = direction_score_from_brier(brier)
        hard = [DIRECTIONS[int(i)] for i in np.argmax(prob, axis=1)]
        pearson, pearson_reason = pearson_or_na(
            part["predicted_effect"].to_numpy(dtype=np.float64),
            part["reference_effect"].to_numpy(dtype=np.float64),
        )
        prevalence = {cls: truth.count(cls) / len(truth) for cls in DIRECTIONS}
        per_target.append(
            {
                "target_id": target,
                "n": int(len(part)),
                "multiclass_brier": brier,
                "direction_score": dscore,
                "macro_f1": macro_f1(hard, truth),
                "recall": class_recall(hard, truth),
                "class_prevalence": prevalence,
                "effect_mae": float(np.mean(np.abs(part["predicted_effect"] - part["reference_effect"]))),
                "pearson_delta": pearson,
                "pearson_delta_na_reason": pearson_reason,
            }
        )
    episode_score = float(np.mean([row["direction_score"] for row in per_target])) if per_target else 0.0
    return {
        "direction_score": episode_score,
        "per_target": per_target,
        "n_targets": len(per_target),
        "n_genes": len(gene_ids),
    }


def aubc(scores_by_budget: dict[int, float], budget: int) -> float:
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if budget == 0:
        return float(scores_by_budget[0])
    acc = 0.5 * scores_by_budget[0]
    for b in range(1, budget):
        acc += scores_by_budget[b]
    acc += 0.5 * scores_by_budget[budget]
    return float(acc / budget)


def uniform_probabilities(n: int) -> np.ndarray:
    return np.full((n, 3), 1.0 / 3.0)


def make_prediction_frame(
    target_ids: Sequence[str],
    gene_ids: Sequence[str],
    *,
    effects: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:
    rows = []
    k = 0
    for target in target_ids:
        for gene in gene_ids:
            rows.append(
                {
                    "target_id": target,
                    "gene_id": gene,
                    "predicted_effect": float(effects[k]),
                    "p_down": float(probs[k, 0]),
                    "p_neutral": float(probs[k, 1]),
                    "p_up": float(probs[k, 2]),
                }
            )
            k += 1
    return pd.DataFrame(rows)
