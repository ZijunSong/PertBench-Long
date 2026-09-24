"""Non-LLM baselines sharing the same O/Q/T, budget, tools, and scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from pertbench_long.evaluation.scoring import make_prediction_frame
from pertbench_long.runtime.broker import Broker
from pertbench_long.tools.helpers import (
    control_similarity,
    evidence_paths_from_spec,
    load_public_control_mapping,
    mean_delta_from_effects,
    visible_effect_table,
)


def _gene_ids(broker: Broker) -> list[str]:
    path = broker.public_root / broker.public_spec.gene_universe_artifact
    return path.read_text(encoding="utf-8").strip().splitlines()[1:]


def _targets(broker: Broker) -> list[str]:
    return [t.target_id for t in broker.public_spec.targets]


def _write_pred(broker: Broker, effects, probs) -> tuple[str, str]:
    frame = make_prediction_frame(_targets(broker), _gene_ids(broker), effects=effects, probs=probs)
    pred_path = broker.workspace / "outputs" / "predictions.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(pred_path, index=False)
    claims = {
        "schema_version": "1.0",
        "claims": [
            {
                "claim_id": "c1",
                "scope": {"targets": _targets(broker)},
                "statement": "Baseline effect transfer from currently unlocked evidence only.",
                "evidence_ids": list(broker.public_spec.initial_evidence) + list(broker.public_spec.reference_evidence),
                "prediction_entries": [{"target_id": t, "gene_id": _gene_ids(broker)[0]} for t in _targets(broker)],
                "limitations": ["Response prediction does not identify a unique mechanism."],
                "alternative_explanations": ["Shared program vs condition-specific regulation."],
            }
        ],
    }
    claims_path = broker.workspace / "outputs" / "claims.json"
    claims_path.write_text(__import__("json").dumps(claims, indent=2), encoding="utf-8")
    return "outputs/predictions.parquet", "outputs/claims.json"


def _visible_paths(broker: Broker) -> tuple[list[Path], Path]:
    return evidence_paths_from_spec(broker.public_root, broker.workspace, broker.public_spec)


def _control_mapping(broker: Broker) -> dict[str, str] | None:
    return load_public_control_mapping(broker.public_root)


def estimate(broker: Broker, sigma: float = 0.15):
    evidence, control = _visible_paths(broker)
    table = visible_effect_table(evidence, control, _gene_ids(broker), control_mapping=_control_mapping(broker))
    return mean_delta_from_effects(table, _targets(broker), _gene_ids(broker), sigma=sigma)


def no_change_neutral_onehot(broker: Broker, **_kwargs) -> None:
    n = len(_targets(broker)) * len(_gene_ids(broker))
    import numpy as np

    effects = np.zeros(n)
    probs = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


def no_change_dev_prior(broker: Broker, prior=(1 / 3, 1 / 3, 1 / 3), **_kwargs) -> None:
    n = len(_targets(broker)) * len(_gene_ids(broker))
    import numpy as np

    effects = np.zeros(n)
    probs = np.tile(np.array(prior, dtype=float), (n, 1))
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


def mean_delta_no_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    effects, probs = estimate(broker, sigma=sigma)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


def _query_then_update(broker: Broker, experiment_ids: Sequence[str], sigma: float = 0.15) -> None:
    effects, probs = estimate(broker, sigma=sigma)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    for i, exp_id in enumerate(experiment_ids, start=1):
        budget = broker.get_budget()
        if budget["remaining_credits"] <= 0:
            break
        broker.request_experiment(exp_id, request_id=f"req_{i:04d}")
        effects, probs = estimate(broker, sigma=sigma)
        p, c = _write_pred(broker, effects, probs)
        broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="budget_exhausted" if broker.get_budget()["remaining_credits"] == 0 else "early_stop")


def random_query(broker: Broker, *, seed: int = 0, sigma: float = 0.15, **_kwargs) -> None:
    import random

    catalog = [c["experiment_id"] for c in broker.list_experiments()]
    rng = random.Random(seed)
    rng.shuffle(catalog)
    chosen = catalog[: broker.public_spec.experimental_budget]
    _query_then_update(broker, chosen, sigma=sigma)


def fixed_order_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    catalog = [c["experiment_id"] for c in broker.list_experiments()]
    chosen = catalog[: broker.public_spec.experimental_budget]
    _query_then_update(broker, chosen, sigma=sigma)


def control_similarity_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    _, control = _visible_paths(broker)
    targets = [t.cell_type for t in broker.public_spec.targets]
    catalog = broker.list_experiments()
    order = control_similarity(control, targets, [c["cell_type"] for c in catalog])
    id_by_ct = {c["cell_type"]: c["experiment_id"] for c in catalog}
    chosen = [id_by_ct[ct] for ct in order if ct in id_by_ct][: broker.public_spec.experimental_budget]
    _query_then_update(broker, chosen, sigma=sigma)


BASELINE_REGISTRY = {
    "no_change": no_change_neutral_onehot,
    "no_change_neutral_onehot": no_change_neutral_onehot,
    "no_change_dev_prior": no_change_dev_prior,
    "mean_delta_no_query": mean_delta_no_query,
    "random_query": random_query,
    "fixed_order_query": fixed_order_query,
    "control_similarity_query": control_similarity_query,
}


def pbmc_baselines() -> dict[str, Any]:
    return {k: BASELINE_REGISTRY[k] for k in ("no_change", "mean_delta_no_query", "random_query", "fixed_order_query", "control_similarity_query")}


def _target_meta(broker: Broker):
    return list(broker.public_spec.targets)


def nearest_dose_no_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    import numpy as np

    evidence, control = _visible_paths(broker)
    table = visible_effect_table(evidence, control, _gene_ids(broker), control_mapping=_control_mapping(broker))
    genes = _gene_ids(broker)
    effects = []
    for target in _target_meta(broker):
        same = table[table["perturbation"].astype(str) == str(target.perturbation)] if not table.empty and "perturbation" in table.columns else table.iloc[0:0]
        if same.empty:
            gene_mean = table.groupby("gene_id")["effect"].mean() if not table.empty else {}
            effects.extend(float(gene_mean.get(g, 0.0)) if hasattr(gene_mean, "get") else 0.0 for g in genes)
            continue
        if "dose" in same.columns and target.dose is not None:
            same = same.copy()
            same["_dist"] = (same["dose"].astype(float) - float(target.dose)).abs()
            best_cid = same.sort_values("_dist").iloc[0]["condition_id"]
            block = same[same["condition_id"] == best_cid]
        else:
            block = same
        gene_mean = block.groupby("gene_id")["effect"].mean()
        effects.extend(float(gene_mean.get(g, 0.0)) for g in genes)
    effects = np.asarray(effects, dtype=np.float64)
    from pertbench_long.baselines.mapping import effects_to_probabilities

    p, c = _write_pred(broker, effects, effects_to_probabilities(effects, sigma=sigma))
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


def log_dose_linear_no_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    import numpy as np

    evidence, control = _visible_paths(broker)
    table = visible_effect_table(evidence, control, _gene_ids(broker), control_mapping=_control_mapping(broker))
    genes = _gene_ids(broker)
    effects = []
    for target in _target_meta(broker):
        same = table[table["perturbation"].astype(str) == str(target.perturbation)] if not table.empty and "perturbation" in table.columns else table.iloc[0:0]
        if same.empty or target.dose in {None, 0}:
            gene_mean = table.groupby("gene_id")["effect"].mean() if not table.empty else {}
            effects.extend(float(gene_mean.get(g, 0.0)) if hasattr(gene_mean, "get") else 0.0 for g in genes)
            continue
        by_dose = same.groupby(["dose", "gene_id"])["effect"].mean().unstack("gene_id")
        xs = np.log10(np.clip(np.asarray(by_dose.index, dtype=np.float64), 1e-12, None))
        x_t = float(np.log10(max(float(target.dose), 1e-12)))
        pred = []
        for gene in genes:
            if gene not in by_dose.columns or len(xs) < 2:
                pred.append(0.0)
                continue
            ys = by_dose[gene].to_numpy(dtype=np.float64)
            slope, intercept = np.polyfit(xs, ys, 1)
            pred.append(float(slope * x_t + intercept))
        effects.extend(pred)
    effects = np.asarray(effects, dtype=np.float64)
    from pertbench_long.baselines.mapping import effects_to_probabilities

    p, c = _write_pred(broker, effects, effects_to_probabilities(effects, sigma=sigma))
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


def single_gene_additivity_no_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    import numpy as np

    evidence, control = _visible_paths(broker)
    table = visible_effect_table(evidence, control, _gene_ids(broker), control_mapping=_control_mapping(broker))
    genes = _gene_ids(broker)
    effects = []
    for target in _target_meta(broker):
        comps = list(target.perturbation_components or [])
        if len(comps) != 2 or table.empty:
            gene_mean = table.groupby("gene_id")["effect"].mean() if not table.empty else {}
            effects.extend(float(gene_mean.get(g, 0.0)) if hasattr(gene_mean, "get") else 0.0 for g in genes)
            continue
        parts = []
        for gene_name in comps:
            block = table[table["perturbation"].astype(str) == str(gene_name)] if "perturbation" in table.columns else table.iloc[0:0]
            if block.empty:
                parts = []
                break
            parts.append(block.groupby("gene_id")["effect"].mean())
        if len(parts) != 2:
            gene_mean = table.groupby("gene_id")["effect"].mean()
            effects.extend(float(gene_mean.get(g, 0.0)) for g in genes)
        else:
            effects.extend(float(parts[0].get(g, 0.0) + parts[1].get(g, 0.0)) for g in genes)
    effects = np.asarray(effects, dtype=np.float64)
    from pertbench_long.baselines.mapping import effects_to_probabilities

    p, c = _write_pred(broker, effects, effects_to_probabilities(effects, sigma=sigma))
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


def context_transfer_no_query(broker: Broker, sigma: float = 0.15, **_kwargs) -> None:
    import numpy as np

    evidence, control = _visible_paths(broker)
    table = visible_effect_table(evidence, control, _gene_ids(broker), control_mapping=_control_mapping(broker))
    genes = _gene_ids(broker)
    effects = []
    for target in _target_meta(broker):
        same = table[table["perturbation"].astype(str) == str(target.perturbation)] if not table.empty and "perturbation" in table.columns else table.iloc[0:0]
        if same.empty:
            gene_mean = table.groupby("gene_id")["effect"].mean() if not table.empty else {}
            effects.extend(float(gene_mean.get(g, 0.0)) if hasattr(gene_mean, "get") else 0.0 for g in genes)
        else:
            gene_mean = same.groupby("gene_id")["effect"].mean()
            effects.extend(float(gene_mean.get(g, 0.0)) for g in genes)
    effects = np.asarray(effects, dtype=np.float64)
    from pertbench_long.baselines.mapping import effects_to_probabilities

    p, c = _write_pred(broker, effects, effects_to_probabilities(effects, sigma=sigma))
    broker.save_snapshot(p, c)
    broker.submit(p, c, stop_reason="no_query")


BASELINE_REGISTRY.update(
    {
        "nearest_dose_no_query": nearest_dose_no_query,
        "log_dose_linear_no_query": log_dose_linear_no_query,
        "single_gene_additivity_no_query": single_gene_additivity_no_query,
        "context_transfer_no_query": context_transfer_no_query,
    }
)


def chemical_dose_baselines() -> dict[str, Any]:
    return {k: BASELINE_REGISTRY[k] for k in ("no_change", "mean_delta_no_query", "nearest_dose_no_query", "log_dose_linear_no_query", "random_query", "fixed_order_query")}


def genetic_pair_baselines() -> dict[str, Any]:
    return {k: BASELINE_REGISTRY[k] for k in ("no_change", "mean_delta_no_query", "single_gene_additivity_no_query", "random_query", "fixed_order_query")}


def context_campaign_baselines() -> dict[str, Any]:
    return {k: BASELINE_REGISTRY[k] for k in ("no_change", "mean_delta_no_query", "context_transfer_no_query", "random_query", "fixed_order_query")}


def run_baseline(name: str, broker: Broker, **kwargs: Any) -> None:
    if name not in BASELINE_REGISTRY:
        raise KeyError(name)
    BASELINE_REGISTRY[name](broker, **kwargs)
