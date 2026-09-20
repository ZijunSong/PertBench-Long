"""Trusted broker: artifact path/size/hash checks. No pickle/joblib loads."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from pertbench_long.errors import InvalidState, PathGuardError, SubmissionInvalid
from pertbench_long.evaluation.submission import validate_claims, validate_predictions
from pertbench_long.hashes import sha256_file
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.sandbox import PathGuard, command_allowed
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.types import PublicEpisodeSpec
from pertbench_long.schemas.validate import parse_resource_profile

SAFE_READ_SUFFIX = {".h5ad", ".parquet", ".tsv", ".csv", ".json", ".txt", ".npz"}


class Broker:
    def __init__(
        self,
        *,
        public_spec: PublicEpisodeSpec,
        oracle: Oracle,
        workspace: Path,
        public_root: Path,
        private_root: Path,
        state: StateMachine,
        run_id: str,
        event_log: Path,
        resource_limits: dict[str, Any] | None = None,
    ) -> None:
        self.public_spec = public_spec
        self.oracle = oracle
        self.workspace = Path(workspace)
        self.public_root = Path(public_root)
        self.private_root = Path(private_root)
        self.state = state
        self.run_id = run_id
        self.event_log = Path(event_log)
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "evidence").mkdir(exist_ok=True)
        (self.workspace / "outputs").mkdir(exist_ok=True)
        self.guard = PathGuard(
            [self.workspace, self.public_root],
            denied=[self.private_root, Path("/var/run/docker.sock")],
        )
        self.evidence_version = 0
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.last_snapshot_version: Optional[int] = None
        self.tool_calls = 0
        self.limits = resource_limits or {}
        profile = parse_resource_profile(
            public_spec.resource_profile if isinstance(public_spec.resource_profile, str) else public_spec.resource_profile
        )
        self.max_tool_calls = int(self.limits.get("max_external_tool_calls", profile.max_external_tool_calls))
        self.tool_timeout_s = int(self.limits.get("tool_timeout_s", profile.tool_timeout_s))
        self.max_output_bytes = int(self.limits.get("max_output_bytes", profile.max_output_bytes))
        self.trusted_submission: Optional[dict[str, Any]] = None
        self.carry_forward_missing = False

    def log(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), "run_id": self.run_id, "evidence_version": self.evidence_version, **event}
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def _count_tool(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise InvalidState("external tool call budget exceeded")

    def list_evidence(self) -> list[dict[str, Any]]:
        self._count_tool()
        items = []
        for path in sorted((self.workspace / "evidence").glob("*")):
            if path.is_file():
                items.append({"path": str(path.relative_to(self.workspace)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        for path in sorted((self.public_root / "evidence").glob("*")):
            if path.is_file():
                items.append({"path": f"public/{path.relative_to(self.public_root)}", "sha256": sha256_file(path), "bytes": path.stat().st_size})
        self.log({"action": "list_evidence", "n": len(items)})
        return items

    def read_artifact(self, relative: str) -> Path:
        self._count_tool()
        path = self.guard.check(self._locate(relative))
        if path.suffix.lower() not in SAFE_READ_SUFFIX:
            raise PathGuardError("artifact format is not allowed")
        if path.stat().st_size > self.max_output_bytes:
            raise PathGuardError("artifact exceeds size limit")
        if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
            raise PathGuardError("pickle/joblib artifacts are not allowed")
        self.log({"action": "read_artifact", "path": relative, "sha256": sha256_file(path)})
        return path

    def _locate(self, relative: str) -> Path:
        rel = Path(relative)
        candidates = [self.workspace / rel, self.public_root / rel]
        for cand in candidates:
            if cand.exists():
                return cand
        return self.workspace / rel

    def list_experiments(self) -> list[dict[str, Any]]:
        self._count_tool()
        return self.oracle.list_experiments()

    def get_budget(self) -> dict[str, Any]:
        return self.oracle.get_budget(self.run_id)

    def request_experiment(self, experiment_id: str, request_id: str) -> dict[str, Any]:
        self.state.require_running("request_experiment")
        existing = self.oracle.ledger.get_request(self.run_id, request_id)
        owned = self.oracle.ledger.get_purchase(self.run_id, experiment_id)
        if self.last_snapshot_version != self.evidence_version and existing is None and owned is None:
            raise InvalidState("a frozen prediction snapshot for the current evidence version is required before purchase")
        self._count_tool()
        result = self.oracle.request_experiment(
            run_id=self.run_id,
            experiment_id=experiment_id,
            request_id=request_id,
            evidence_dir=self.workspace / "evidence",
        )
        if result.get("charged_credits", 0) > 0:
            self.evidence_version = int(result["evidence_version"])
        self.log({"action": "request_experiment", "experiment_id": experiment_id, "request_id": request_id, "result": result})
        return result

    def save_snapshot(self, prediction_path: str, claims_path: str | None = None) -> dict[str, Any]:
        self.state.require_running("save_snapshot")
        self._count_tool()
        pred = self.guard.check(self._locate(prediction_path) if Path(prediction_path).is_absolute() else (self.workspace / prediction_path))
        import pandas as pd

        frame = pd.read_parquet(pred)
        target_ids = [t.target_id for t in self.public_spec.targets]
        gene_ids = (self.public_root / self.public_spec.gene_universe_artifact).read_text(encoding="utf-8").strip().splitlines()[1:]
        validate_predictions(frame, target_ids=target_ids, gene_ids=gene_ids)
        snap_dir = self.workspace / "submissions" / f"snap_v{self.evidence_version}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        dest = snap_dir / "predictions.parquet"
        shutil.copyfile(pred, dest)
        payload = {
            "evidence_version": self.evidence_version,
            "prediction_sha256": sha256_file(dest),
            "bytes": dest.stat().st_size,
            "path": str(dest),
        }
        if claims_path:
            claim_src = Path(claims_path)
            cpath = self.guard.check(claim_src if claim_src.is_absolute() else (self.workspace / claims_path))
            claims = json.loads(cpath.read_text(encoding="utf-8"))
            visible = [item["path"] for item in self.list_evidence()]
            # list_evidence increments tool count; acceptable.
            validate_claims(claims, target_ids=target_ids, evidence_ids=list(self.public_spec.initial_evidence) + list(self.public_spec.reference_evidence))
            cdest = snap_dir / "claims.json"
            shutil.copyfile(cpath, cdest)
            payload["claims_sha256"] = sha256_file(cdest)
        self.snapshots[self.evidence_version] = payload
        self.last_snapshot_version = self.evidence_version
        self.log({"action": "save_snapshot", **payload})
        return {"status": "ok", "format": "valid", "evidence_version": self.evidence_version}

    def submit(self, prediction_path: str, claims_path: str, stop_reason: str = "submitted") -> dict[str, Any]:
        self.state.require_running("submit")
        result = self.save_snapshot(prediction_path, claims_path)
        snap = self.snapshots[self.evidence_version]
        trusted = self.workspace.parent / "trusted_submissions" / self.run_id
        trusted.mkdir(parents=True, exist_ok=True)
        dest = trusted / "predictions.parquet"
        shutil.copyfile(snap["path"], dest)
        claims_src = Path(snap["path"]).parent / "claims.json"
        if claims_src.exists():
            shutil.copyfile(claims_src, trusted / "claims.json")
        self.trusted_submission = {
            "prediction_sha256": sha256_file(dest),
            "bytes": dest.stat().st_size,
            "path": str(dest),
            "stop_reason": stop_reason,
            "evidence_version": self.evidence_version,
        }
        self.oracle.ledger.set_status(self.run_id, "SUBMITTED")
        self.state.transition(RunState.SUBMITTED)
        self.log({"action": "submit", **self.trusted_submission})
        return {"status": "ok", "receipt": {"bytes": self.trusted_submission["bytes"], "sha256": self.trusted_submission["prediction_sha256"]}}

    def run_python(self, code: str, execute: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        self.state.require_running("python")
        self._count_tool()
        started = time.time()
        result = execute(code)
        self.log({"action": "python", "elapsed_s": time.time() - started, "ok": result.get("ok")})
        return result

    def run_shell(self, command: str, execute: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        self.state.require_running("shell")
        self._count_tool()
        command_allowed(command)
        started = time.time()
        result = execute(command)
        self.log({"action": "shell", "elapsed_s": time.time() - started, "ok": result.get("ok")})
        return result
