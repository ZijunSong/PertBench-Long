"""Frozen mapping from scalar effects to three-class probabilities. Sigma is not fit on hidden T."""

from __future__ import annotations

import numpy as np

from pertbench_long.evaluation.labels import TAU_DEFAULT


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def effects_to_probabilities(effects: np.ndarray, *, tau: float = TAU_DEFAULT, sigma: float = 0.15) -> np.ndarray:
    """Uncalibrated logistic-normal threshold mapping. Marked uncalibrated unless sigma was frozen on development."""
    delta = np.asarray(effects, dtype=np.float64).reshape(-1)
    down = (-delta - tau) / sigma
    neu = np.zeros_like(delta)
    up = (delta - tau) / sigma
    logits = np.stack([down, neu, up], axis=1)
    return softmax(logits)
