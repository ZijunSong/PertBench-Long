"""Submission validation. Illegal probabilities are not silently renormalized."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pertbench_long.errors import SubmissionInvalid

REQUIRED_COLUMNS = ("target_id", "gene_id", "predicted_effect", "p_down", "p_neutral", "p_up")
PROB_COLS = ("p_down", "p_neutral", "p_up")
PROB_TOL = 1e-6


def load_predictions(path: Path | str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame


def validate_predictions(
    frame: pd.DataFrame,
    *,
    target_ids: Sequence[str],
    gene_ids: Sequence[str],
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SubmissionInvalid(f"missing columns: {missing}")
    if frame.empty:
        raise SubmissionInvalid("prediction table is empty")
    key = frame["target_id"].astype(str) + "\t" + frame["gene_id"].astype(str)
    if key.duplicated().any():
        raise SubmissionInvalid("duplicate target_id × gene_id rows")
    expected = {(t, g) for t in target_ids for g in gene_ids}
    got = set(zip(frame["target_id"].astype(str), frame["gene_id"].astype(str)))
    missing_pairs = expected - got
    extra_pairs = got - expected
    if missing_pairs:
        raise SubmissionInvalid(f"missing {len(missing_pairs)} required target×gene rows")
    if extra_pairs:
        raise SubmissionInvalid(f"{len(extra_pairs)} unknown target×gene rows")
    numeric = frame.loc[:, ["predicted_effect", *PROB_COLS]]
    if numeric.isna().any().any():
        raise SubmissionInvalid("NaN in predicted_effect or probabilities")
    arr = numeric.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise SubmissionInvalid("non-finite values in predicted_effect or probabilities")
    probs = frame.loc[:, PROB_COLS].to_numpy(dtype=np.float64)
    if np.any(probs < 0) or np.any(probs > 1):
        raise SubmissionInvalid("probabilities must be in [0, 1]")
    sums = probs.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > PROB_TOL):
        raise SubmissionInvalid("probability rows must sum to 1 ± 1e-6; no silent renormalization")
    ordered = (
        frame.assign(_t=pd.Categorical(frame["target_id"], list(target_ids)), _g=pd.Categorical(frame["gene_id"], list(gene_ids)))
        .sort_values(["_t", "_g"])
        .drop(columns=["_t", "_g"])
        .reset_index(drop=True)
    )
    return ordered


def validate_claims(payload: dict, *, target_ids: Sequence[str], evidence_ids: Sequence[str]) -> dict:
    if not isinstance(payload, dict):
        raise SubmissionInvalid("claims.json must be an object")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise SubmissionInvalid("claims must be a list")
    visible = set(evidence_ids)
    known_targets = set(target_ids)
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise SubmissionInvalid(f"claims[{i}] must be an object")
        for key in ("claim_id", "scope", "statement", "evidence_ids", "prediction_entries", "limitations"):
            if key not in claim:
                raise SubmissionInvalid(f"claims[{i}].{key} missing")
        for ev in claim["evidence_ids"]:
            if ev not in visible:
                raise SubmissionInvalid(f"claims[{i}] cites non-visible evidence {ev!r}")
        for entry in claim["prediction_entries"]:
            if entry.get("target_id") not in known_targets:
                raise SubmissionInvalid(f"claims[{i}] refers to unknown target")
    return payload
