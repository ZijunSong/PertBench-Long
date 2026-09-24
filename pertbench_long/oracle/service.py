"""Trusted query oracle. Agent may pass only public experiment IDs.

Reveal order: authorize in the ledger first, materialize into a private staging
directory that the Agent cannot read, then deliver into the visible registry.
Unauthorized results never appear as files, bytes, or registry entries.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from pertbench_long.errors import BudgetExceeded, IdempotencyConflict, IntegrityError, InvalidState, UnavailableExperiment
from pertbench_long.hashes import sha256_file, sha256_json
from pertbench_long.oracle.ledger import (
    PHASE_AUTHORIZED,
    PHASE_DELIVERED,
    PHASE_MATERIALIZED,
    Ledger,
)
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
        self.staging_root = self.private_root / "_oracle_staging"
        self.staging_root.mkdir(parents=True, exist_ok=True)

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
                "dose": c.dose,
                "dose_unit": c.dose_unit,
                "context_id": c.context_id or c.cell_type,
                "condition_id": c.condition_id,
                "perturbation_kind": c.perturbation_kind,
                "time": c.time,
                "time_unit": c.time_unit,
                "perturbation_components": list(c.perturbation_components),
                "control_group_id": c.control_group_id,
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

    def _source_path(self, experiment_id: str) -> Path:
        src = self.private_root / "queryable" / f"ev_{experiment_id}.h5ad"
        if not src.exists():
            raise UnavailableExperiment("experiment result is not available")
        return src

    def _artifact_meta(self, experiment_id: str) -> dict[str, Any]:
        q_arts = self.spec.provenance.get("queryable_artifacts") or []
        for item in q_arts:
            if item["experiment_id"] == experiment_id:
                return item["artifact"]
        raise UnavailableExperiment("experiment result is not available")

    def _raise_ledger_error(self, result: dict[str, Any]) -> None:
        if result.get("error") == "IDEMPOTENCY_CONFLICT":
            raise IdempotencyConflict("request_id was reused with a different payload")
        if result.get("error") == "BUDGET_EXCEEDED":
            raise BudgetExceeded("experimental budget would be exceeded")
        if result.get("error") == "FAILED_REQUEST":
            raise InvalidState(result.get("message") or "request_id previously failed; use a new request_id")
        if result.get("error") == "INVALID_STATE":
            raise InvalidState(result.get("message") or "purchases are not allowed in the current run state")

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
        if snap.episode_id != self.spec.episode_id:
            raise InvalidState("run is bound to a different episode")
        if snap.status != "RUNNING":
            raise InvalidState("purchases are not allowed in the current run state")

        src = self._source_path(experiment_id)
        source_sha = sha256_file(src)
        staging_name = f"{run_id}__{request_id}__{uuid.uuid4().hex[:8]}"
        auth = self.ledger.authorize(
            run_id=run_id,
            episode_id=self.spec.episode_id,
            request_id=request_id,
            experiment_id=experiment_id,
            payload_hash=payload_hash,
            price=1,
            source_sha256=source_sha,
            staging_name=staging_name,
        )
        if auth.get("error"):
            self._raise_ledger_error(auth)
            raise InvalidState("purchase was rejected")

        # Replay uses the original experiment_id and staging, never the conflicting payload.
        exp_id = auth.get("experiment_id") or experiment_id
        src = self._source_path(exp_id)
        live_hash = sha256_file(src)
        expected_hash = auth.get("source_sha256") or live_hash
        if live_hash != expected_hash:
            raise InvalidState("trusted source artifact hash changed")
        owned = self.ledger.get_purchase(run_id, exp_id)
        if owned is None and not auth.get("replay"):
            # authorize should have created the purchase; missing row means refund already happened
            raise InvalidState("purchase is not authorized")
        if auth.get("purchase_phase") == "FAILED":
            raise InvalidState("request_id previously failed; use a new request_id")

        staging_dir = self.staging_root / (auth.get("staging_name") or staging_name)
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / f"ev_{exp_id}.h5ad"
        file_visible = False
        try:
            if not staged.exists():
                tmp = staging_dir / f".{uuid.uuid4().hex}.tmp"
                shutil.copyfile(src, tmp)
                if sha256_file(tmp) != live_hash:
                    tmp.unlink(missing_ok=True)
                    raise InvalidState("staging copy hash mismatch")
                tmp.replace(staged)
            elif sha256_file(staged) != live_hash:
                raise InvalidState("existing staging copy hash mismatch")
            artifact = {
                "artifact_id": f"ev_{exp_id}",
                "relative_path": f"evidence/ev_{exp_id}.h5ad",
                "sha256": sha256_file(staged),
                "bytes": staged.stat().st_size,
            }
            stored = self.ledger.get_request(run_id, request_id)
            phase = stored["purchase_phase"] if stored else PHASE_AUTHORIZED
            if phase == PHASE_AUTHORIZED:
                self.ledger.mark_materialized(run_id, request_id, artifact)
            evidence_dir = Path(evidence_dir)
            evidence_dir.mkdir(parents=True, exist_ok=True)
            dest = evidence_dir / f"ev_{exp_id}.h5ad"
            # Never trust an Agent-writable file of the same name.
            dest_tmp = evidence_dir / f".{request_id}.{uuid.uuid4().hex}.deliver"
            shutil.copyfile(staged, dest_tmp)
            if sha256_file(dest_tmp) != artifact["sha256"]:
                dest_tmp.unlink(missing_ok=True)
                raise InvalidState("delivery copy hash mismatch")
            dest_tmp.replace(dest)
            file_visible = dest.exists()
            try:
                delivered = self.ledger.mark_delivered(run_id, request_id)
            except Exception as exc:
                raise IntegrityError(
                    "evidence file was revealed but delivery was not recorded; the run must stop"
                ) from exc
        except IntegrityError:
            raise
        except Exception:
            stored = self.ledger.get_request(run_id, request_id)
            if stored and stored["purchase_phase"] == PHASE_AUTHORIZED and not file_visible:
                self.ledger.fail_and_refund(run_id, request_id)
            elif file_visible:
                raise IntegrityError(
                    "evidence file was revealed but delivery was not recorded; the run must stop"
                ) from None
            raise

        snap = self.ledger.snapshot(run_id)
        return {
            "status": "ok",
            "experiment_id": exp_id,
            "request_id": request_id,
            "charged_credits": 0 if auth.get("replay") else auth["charged"],
            "remaining_credits": snap.remaining,
            "evidence_version": int(snap.unique_purchases),
            "artifact": delivered.get("artifact") or artifact,
            "purchase_phase": PHASE_DELIVERED,
            "replay": bool(auth.get("replay")),
        }
