"""Inference-only predictor adapter. Test outcome paths are forbidden."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from pertbench_long.errors import PredictorRejected


class Predictor(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, visible_evidence: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def predict(self, target_context: Mapping[str, Any], artifact_out: str) -> None:
        ...


def assert_no_test_paths(config: Mapping[str, Any]) -> None:
    forbidden = ("test_h5ad", "test_path", "label_artifact", "private_root", "heldout")
    blob = str(config).lower()
    for key in config:
        if any(tok in str(key).lower() for tok in forbidden):
            raise PredictorRejected(f"predictor config contains forbidden key {key}")
    if "test outcome" in blob:
        raise PredictorRejected("predictor config contains test outcome path")


def assert_checkpoint_provenance(checkpoint: Mapping[str, Any], allowed_observation_hash: str) -> None:
    train_hash = checkpoint.get("train_observation_hash")
    split = checkpoint.get("split_id")
    if not train_hash or not split:
        mode = checkpoint.get("mode", "official")
        if mode != "diagnostic":
            raise PredictorRejected("checkpoint missing train/split provenance; official track rejected")
        return
    leaked = set(checkpoint.get("includes_unpurchased_or_target", []))
    if leaked:
        raise PredictorRejected("checkpoint trained on unpurchased Q or T observations")
    if checkpoint.get("allowed_observation_hash") != allowed_observation_hash:
        raise PredictorRejected("checkpoint evidence hash does not match current visible evidence")


class MeanDeltaPredictor(Predictor):
    name = "mean_delta_visible"

    def __init__(self) -> None:
        self._effects = None

    def fit(self, visible_evidence: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        assert_no_test_paths(config)
        self._effects = visible_evidence.get("gene_effects")

    def predict(self, target_context: Mapping[str, Any], artifact_out: str) -> None:
        if self._effects is None:
            raise PredictorRejected("predictor has not been fit on visible evidence")
        from pathlib import Path
        import json

        Path(artifact_out).write_text(json.dumps({"effects": self._effects, "targets": target_context.get("target_ids")}), encoding="utf-8")
