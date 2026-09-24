"""Proxy labels and (conditional) replicate DE labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pertbench_long.errors import LabelBuildError
from pertbench_long.schemas.types import LABEL_EFFECT_PROXY_V1, LABEL_LOGNORM_CELLMEAN_DELTA_V1, LABEL_REPLICATE_DE_V1

TAU_DEFAULT = 0.10
DIRECTIONS = ("down", "neutral", "up")


def _require_finite(array: np.ndarray, where: str) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    if data.size == 0:
        raise LabelBuildError(f"{where} is empty")
    if not np.all(np.isfinite(data)):
        raise LabelBuildError(f"{where} contains NaN/Inf; invalid values are not labeled as neutral")
    return data


def direction_from_delta(delta: np.ndarray, tau: float = TAU_DEFAULT) -> np.ndarray:
    data = _require_finite(delta, "effect delta")
    out = np.full(data.shape, "neutral", dtype=object)
    out = np.where(data < -tau, "down", out)
    out = np.where(data > tau, "up", out)
    return out


def _paired_groups(stim_groups: Sequence[str], control_groups: Sequence[str]) -> list[str]:
    return sorted(set(stim_groups) & set(control_groups) - {None, "None", ""})


def _group_equal_delta(
    stim: np.ndarray,
    control: np.ndarray,
    stim_groups: Sequence[str],
    control_groups: Sequence[str],
    *,
    label: str,
) -> tuple[np.ndarray, str]:
    groups = _paired_groups(stim_groups, control_groups)
    if not groups:
        raise LabelBuildError(f"{label} metadata present but no paired groups across stim/control")
    deltas = []
    for group in groups:
        s = stim[np.array(stim_groups) == group]
        c = control[np.array(control_groups) == group]
        if s.size == 0 or c.size == 0:
            continue
        deltas.append(s.mean(axis=0) - c.mean(axis=0))
    if not deltas:
        raise LabelBuildError(f"no {label} with both stim and control cells")
    return np.mean(np.stack(deltas, axis=0), axis=0), f"{label}_equal_weight"


def mean_effect(
    stim: np.ndarray,
    control: np.ndarray,
    *,
    donor_stim: Optional[Sequence[str]] = None,
    donor_control: Optional[Sequence[str]] = None,
    group_stim: Optional[Sequence[str]] = None,
    group_control: Optional[Sequence[str]] = None,
    estimand: str = "auto",
) -> tuple[np.ndarray, str]:
    stim = _require_finite(stim, "stim matrix")
    control = _require_finite(control, "control matrix")
    if stim.size == 0 or control.size == 0:
        raise LabelBuildError("stim and control matrices must be non-empty")
    if stim.shape[1] != control.shape[1]:
        raise LabelBuildError("stim and control gene dimensions do not match")
    has_groups = group_stim is not None and group_control is not None and any(
        g not in {None, "None", ""} for g in list(group_stim) + list(group_control)
    )
    has_donors = donor_stim is not None and donor_control is not None and any(
        d is not None for d in list(donor_stim) + list(donor_control)
    )
    if estimand == "cell_weighted_descriptive":
        return stim.mean(axis=0) - control.mean(axis=0), "cell_weighted_descriptive"
    if estimand == "replicate_equal_weight" or (estimand == "auto" and has_groups):
        if not has_groups:
            raise LabelBuildError("replicate_equal_weight refused: replicate_id metadata is missing")
        return _group_equal_delta(stim, control, group_stim, group_control, label="replicate")
    if estimand == "donor_equal_weight" or (estimand == "auto" and has_donors):
        if not has_donors:
            raise LabelBuildError("donor metadata present but incomplete")
        return _group_equal_delta(stim, control, donor_stim, donor_control, label="donor")
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
            if not np.isfinite(dlt):
                raise LabelBuildError("reference_effect is not finite")
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
    if frame.empty:
        raise LabelBuildError("label table is empty")
    key = frame["target_id"].astype(str) + "\t" + frame["gene_id"].astype(str)
    if key.duplicated().any():
        raise LabelBuildError("duplicate target_id × gene_id labels")
    if not np.all(np.isfinite(frame["reference_effect"].to_numpy(dtype=np.float64))):
        raise LabelBuildError("label table contains non-finite reference_effect")
    return LabelTable(
        frame=frame,
        profile=LABEL_EFFECT_PROXY_V1,
        effect_unit="log1p_mean_diff",
        tau=tau,
        aggregation=aggregation,
        notes=notes,
    )


def build_lognorm_cellmean_delta_labels(
    *,
    gene_ids: Sequence[str],
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    donors: Mapping[str, tuple[Optional[Sequence[str]], Optional[Sequence[str]]]] | None = None,
    replicates: Mapping[str, tuple[Optional[Sequence[str]], Optional[Sequence[str]]]] | None = None,
    estimand: str = "auto",
    interaction_partners: Mapping[str, tuple[np.ndarray | None, np.ndarray | None]] | None = None,
) -> LabelTable:
    """Continuous treated-minus-matched-control cell-mean delta on an already-transformed matrix.

    The caller must place the store into log-normalized space using the full gene universe
    *before* panel crop. This function does not apply a second log1p.
    """
    rows = []
    notes = [
        "label_name=lognorm_cellmean_delta",
        "delta is mean(z_treated)-mean(z_matched_control) after lognorm_cellmean_delta_v1",
        "this is not log2 fold-change and not a significance call",
    ]
    aggregations: dict[str, str] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for target_id, (stim, control) in targets.items():
        donor_pair = (donors or {}).get(target_id)
        donor_stim = donor_pair[0] if donor_pair else None
        donor_ctrl = donor_pair[1] if donor_pair else None
        rep_pair = (replicates or {}).get(target_id)
        group_stim = rep_pair[0] if rep_pair else None
        group_ctrl = rep_pair[1] if rep_pair else None
        if group_stim is None or not any(g not in {None, "None", ""} for g in (group_stim or [])):
            notes.append(f"{target_id}:sample_level_descriptive_effect_cells_are_not_biological_replicates")
        target_estimand = estimand
        if estimand == "auto" and (group_stim is None or not any(g not in {None, "None", ""} for g in (group_stim or []))):
            target_estimand = "cell_weighted_descriptive"
        delta, aggregation = mean_effect(
            stim,
            control,
            donor_stim=donor_stim,
            donor_control=donor_ctrl,
            group_stim=group_stim,
            group_control=group_ctrl,
            estimand=target_estimand,
        )
        aggregations[target_id] = aggregation
        n_rep = len({g for g in (group_stim or []) if g not in {None, "None", ""}})
        n_rep_ctrl = len({g for g in (group_ctrl or []) if g not in {None, "None", ""}})
        coverage[target_id] = {
            "n_stim_cells": int(np.asarray(stim).shape[0]),
            "n_control_cells": int(np.asarray(control).shape[0]),
            "n_stim_replicates": n_rep,
            "n_control_replicates": n_rep_ctrl,
            "aggregation": aggregation,
        }
        residual = None
        residual_reason = None
        partners = (interaction_partners or {}).get(target_id)
        if partners is not None:
            delta_a, delta_b = partners
            if delta_a is None or delta_b is None:
                residual_reason = "missing_single_gene_reference"
            else:
                residual = delta - np.asarray(delta_a, dtype=np.float64) - np.asarray(delta_b, dtype=np.float64)
        for j, (gene, dlt) in enumerate(zip(gene_ids, delta)):
            if not np.isfinite(dlt):
                raise LabelBuildError("reference_effect is not finite")
            item = {
                "target_id": target_id,
                "gene_id": gene,
                "reference_effect": float(dlt),
                "label_name": "lognorm_cellmean_delta",
                "aggregation": aggregation,
            }
            if residual is None:
                item["interaction_residual"] = np.nan
                item["interaction_residual_reason"] = residual_reason or "not_requested"
            else:
                item["interaction_residual"] = float(residual[j]) if np.isfinite(residual[j]) else np.nan
                item["interaction_residual_reason"] = None if np.isfinite(residual[j]) else "non_finite"
            rows.append(item)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise LabelBuildError("label table is empty")
    key = frame["target_id"].astype(str) + "\t" + frame["gene_id"].astype(str)
    if key.duplicated().any():
        raise LabelBuildError("duplicate target_id × gene_id labels")
    unique_agg = sorted(set(aggregations.values()))
    notes.append("per_target_aggregation=" + ",".join(f"{k}:{v}" for k, v in sorted(aggregations.items())))
    notes.append("coverage=" + str(coverage))
    return LabelTable(
        frame=frame,
        profile=LABEL_LOGNORM_CELLMEAN_DELTA_V1,
        effect_unit="lognorm_cellmean_delta",
        tau=None,
        aggregation=",".join(unique_agg) if unique_agg else "cell_weighted_descriptive",
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
