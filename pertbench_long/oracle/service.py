"""Trusted query oracle. Agent may pass only public experiment IDs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from pertbench_long.errors import BudgetExceeded, IdempotencyConflict, InvalidState, UnavailableExperiment
from pertbench_long.hashes import sha256_file, sha256_json
from pertbench_long.oracle.ledger import Ledger
from pertbench_long.schemas.types import PrivateEpisodeSpec
from pertbench_long.schemas.validate import validate_private_payload


class Oracle:
    def __init__(self, private_spec: PrivateEpisodeSpec, *, ledger: Ledger, private_root: Path, public_root: Path) -> None:
        self.spec = private_spec
        self.ledger = ledger
        self.private_root = Path(private_root)
        self.public_root = Path(public_root)
        self._q_ids = {c.experiment_id for c in private_spec.public.candidate_experiments}
        self._t_ids = {t.target_id for t in private_spec.public.targets}

    @classmethod
    def from_manifest(cls, manifest_path: Path, *, ledger_path: Path, public_root: Path) -> "Oracle":
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        spec = validate_private_payload(payload)
        ledger = Ledger(ledger_path)
        return cls(spec, ledger=ledger, private_root=Path(manifest_path).parent, public_root=Path(public_root))

    def list_experiments(self) -> list[dict[str, Any]]:
        return [
            {
                "experiment_id": c.experiment_id,
                "cell_type": c.cell_type,
                "perturbation": c.perturbation,
                "cost": c.cost,
            }
            for c in self.spec.public.candidate_experiments
        ]

    def get_budget(self, run_id: str) -> dict[str, Any]:
        snap = self.ledger.snapshot(run_id)
        return {
            "status": "ok",
            "budget": snap.budget,
            "charged_credits": snap.charged,
            "remaining_credits": snap.remaining,
            "unique_purchases": snap.unique_purchases,
            "cost_unit": "credit",
        }

    def _artifact_for(self, experiment_id: str) -> dict[str, Any]:
        q_arts = self.spec.provenance.get("queryable_artifacts") or []
        for item in q_arts:
            if item["experiment_id"] == experiment_id:
                src = self.private_root / "queryable" / f"ev_{experiment_id}.h5ad"
                if not src.exists():
                    raise UnavailableExperiment("experiment result is not available")
                return item["artifact"] | {"source_path": str(src)}
        raise UnavailableExperiment("experiment result is not available")

    def request_experiment(
        self,
        *,
        run_id: str,
        experiment_id: str,
        request_id: str,
        evidence_dir: Path,
    ) -> dict[str, Any]:
        payload_hash = sha256_json({"experiment_id": experiment_id})
        if experiment_id not in self._q_ids:
            # Do not reveal whether the ID matches a hidden target.
            raise UnavailableExperiment("experiment result is not available")
        snap = self.ledger.snapshot(run_id)
        if snap.status != "RUNNING":
            raise InvalidState("purchases are not allowed in the current run state")
        meta = self._artifact_for(experiment_id)
        src = Path(meta["source_path"])
        dest_dir = Path(evidence_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"ev_{experiment_id}.h5ad"
        tmp = dest.with_suffix(".h5ad.tmp")
        if not dest.exists():
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
        artifact = {
            "artifact_id": f"ev_{experiment_id}",
            "relative_path": f"evidence/ev_{experiment_id}.h5ad",
            "sha256": sha256_file(dest),
        }
        result = self.ledger.commit_purchase(
            run_id=run_id,
            request_id=request_id,
            experiment_id=experiment_id,
            payload_hash=payload_hash,
            price=1,
            artifact=artifact,
            already_owned=False,
        )
        if result.get("error") == "IDEMPOTENCY_CONFLICT":
            raise IdempotencyConflict("request_id was reused with a different payload")
        if result.get("error") == "BUDGET_EXCEEDED":
            raise BudgetExceeded("experimental budget would be exceeded")
        if result.get("error") == "INVALID_STATE":
            raise InvalidState("purchases are not allowed in the current run state")
        return {
            "status": "ok",
            "experiment_id": experiment_id,
            "request_id": request_id,
            "charged_credits": result["charged"],
            "remaining_credits": result["remaining"],
            "evidence_version": result["evidence_version"],
            "artifact": result["artifact"],
        }
