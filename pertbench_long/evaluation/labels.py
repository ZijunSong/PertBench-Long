"""Proxy labels and (conditional) replicate DE labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pertbench_long.errors import LabelBuildError
from pertbench_long.schemas.types import LABEL_EFFECT_PROXY_V1, LABEL_REPLICATE_DE_V1

TAU_DEFAULT = 0.10
DIRECTIONS = ("down", "neutral", "up")


def direction_from_delta(delta: np.ndarray, tau: float = TAU_DEFAULT) -> np.ndarray:
    out = np.full(delta.shape, "neutral", dtype=object)
    out = np.where(delta < -tau, "down", out)
    out = np.where(delta > tau, "up", out)
    return out


def mean_effect(
    stim: np.ndarray,
    control: np.ndarray,
    *,
    donor_stim: Optional[Sequence[str]] = None,
    donor_control: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, str]:
    if stim.size == 0 or control.size == 0:
        raise LabelBuildError("stim and control matrices must be non-empty")
    if (
        donor_stim is not None
        and donor_control is not None
        and any(d is not None for d in list(donor_stim) + list(donor_control))
    ):
        donors = sorted(set(donor_stim) & set(donor_control) - {None, "None", ""})
        if not donors:
            raise LabelBuildError("donor metadata present but no paired donors across stim/control")
        deltas = []
        for donor in donors:
            s = stim[np.array(donor_stim) == donor]
            c = control[np.array(donor_control) == donor]
            if s.size == 0 or c.size == 0:
                continue
            deltas.append(s.mean(axis=0) - c.mean(axis=0))
        if not deltas:
            raise LabelBuildError("no donor with both stim and control cells")
        return np.mean(np.stack(deltas, axis=0), axis=0), "donor_equal_weight"
    return stim.mean(axis=0) - control.mean(axis=0), "cell_weighted"


@dataclass
class LabelTable:
    frame: pd.DataFrame
    profile: str
    effect_unit: str
    tau: Optional[float]
    aggregation: str
    notes: list[str]

    def to_parquet(self, path) -> None:
        self.frame.to_parquet(path, index=False)


def build_effect_proxy_labels(
    *,
    gene_ids: Sequence[str],
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    tau: float = TAU_DEFAULT,
    donors: Mapping[str, tuple[Optional[Sequence[str]], Optional[Sequence[str]]]] | None = None,
) -> LabelTable:
    rows = []
    notes = [
        "label_name=effect_direction_proxy",
        "neutral means |delta| <= tau, not significant DE and not biological no-change",
        f"tau={tau} log1p units is an engineering threshold frozen from development",
    ]
    aggregation = "cell_weighted"
    for target_id, (stim, control) in targets.items():
        donor_pair = (donors or {}).get(target_id)
        donor_stim = donor_pair[0] if donor_pair else None
        donor_ctrl = donor_pair[1] if donor_pair else None
        if donor_stim is None:
            notes.append("donor_metadata_missing_using_cell_weighted_descriptive_stats")
        delta, aggregation = mean_effect(stim, control, donor_stim=donor_stim, donor_control=donor_ctrl)
        direction = direction_from_delta(delta, tau=tau)
        for gene, dlt, direc in zip(gene_ids, delta, direction):
            rows.append(
                {
                    "target_id": target_id,
                    "gene_id": gene,
                    "reference_effect": float(dlt),
                    "effect_direction_proxy": str(direc),
                    "label_name": "effect_direction_proxy",
                }
            )
    frame = pd.DataFrame(rows)
    return LabelTable(
        frame=frame,
        profile=LABEL_EFFECT_PROXY_V1,
        effect_unit="log1p_mean_diff",
        tau=tau,
        aggregation=aggregation,
        notes=notes,
    )


def build_replicate_de_labels(**kwargs) -> LabelTable:
    donors = kwargs.get("donors")
    n_replicates = kwargs.get("n_replicates_per_group", 0)
    counts_available = kwargs.get("counts_available", False)
    if not donors:
        raise LabelBuildError("replicate_de_v1 refused: no donor/biological-replicate metadata; cells are not replicates")
    if not counts_available:
        raise LabelBuildError("replicate_de_v1 refused: original counts unavailable")
    if n_replicates < 3:
        raise LabelBuildError("replicate_de_v1 refused: configured minimum of 3 biological replicates not met")
    raise LabelBuildError("replicate_de_v1 not enabled for this dataset: design/confounding gate failed")
