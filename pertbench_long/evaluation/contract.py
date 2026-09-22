"""Versioned public submission contract. Models must not need host source to submit."""

from __future__ import annotations

from typing import Any

from pertbench_long.evaluation.submission import REQUIRED_COLUMNS

SUBMISSION_CONTRACT_VERSION = "submission_contract_v1"
CONTRACT_FILENAME = "submission_contract.json"


def public_submission_contract(*, effect_unit: str = "log1p_mean_diff", require_direction: bool = True) -> dict[str, Any]:
    columns = list(REQUIRED_COLUMNS) if require_direction else ["target_id", "gene_id", "predicted_effect"]
    return {
        "schema_version": SUBMISSION_CONTRACT_VERSION,
        "effect_unit": effect_unit,
        "predictions": {
            "relative_path": "outputs/predictions.parquet",
            "container_path": "/workspace/outputs/predictions.parquet",
            "required_columns": columns,
            "coverage": "exactly one row for every public target_id × gene_id from genes_v1.tsv",
            "predicted_effect": f"finite float in canonical unit {effect_unit}",
            "probabilities": {
                "columns": ["p_down", "p_neutral", "p_up"],
                "range": [0.0, 1.0],
                "sum_tolerance": 1e-6,
                "silent_renormalization": False,
            },
            "rejected": ["NaN", "Inf", "duplicate target_id×gene_id", "missing rows", "unknown ids"],
        },
        "claims": {
            "relative_path": "outputs/claims.json",
            "container_path": "/workspace/outputs/claims.json",
            "required_claim_keys": [
                "claim_id",
                "scope",
                "statement",
                "evidence_ids",
                "prediction_entries",
                "limitations",
            ],
            "minimum_example": {
                "schema_version": "1.0",
                "claims": [
                    {
                        "claim_id": "c1",
                        "scope": {"targets": ["t_01"]},
                        "statement": "public example claim; replace with the model's actual statement",
                        "evidence_ids": ["ev_initial_001"],
                        "prediction_entries": [{"target_id": "t_01", "gene_id": "<gene_id from genes_v1.tsv>"}],
                        "limitations": ["example"],
                    }
                ],
            },
        },
        "snapshots": {
            "before_purchase": "call save_prediction_snapshot for the current evidence version before request_experiment",
            "submit": "submit freezes the final trusted snapshot; a text claim of being done is not a receipt",
        },
        "paths": {
            "accepted": ["outputs/predictions.parquet", "/workspace/outputs/predictions.parquet"],
            "rejected": ["host absolute paths", "../private", "symlinks that leave the workspace"],
        },
    }


def contract_prompt_block(contract: dict[str, Any]) -> str:
    preds = contract["predictions"]
    cols = ", ".join(preds["required_columns"])
    return (
        f"Submission contract {contract['schema_version']} is also at {CONTRACT_FILENAME}.\n"
        f"Write parquet columns [{cols}] covering every target×gene. "
        f"Probabilities in [0,1] must sum to 1 ± 1e-6; they are not renormalized. "
        f"Effect unit: {contract['effect_unit']}. "
        f"claims.json must include claim_id, scope, statement, evidence_ids, prediction_entries, limitations. "
        f"Use relative paths such as outputs/predictions.parquet or the matching /workspace/... path."
    )
