"""Trusted broker: artifact registry, frozen snapshots, and oracle purchases.

Model-generated code never receives this object. Tools go through ToolRouter JSON.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from pertbench_long.errors import IntegrityError, InvalidState, PathGuardError, SubmissionInvalid, ToolObservationError
from pertbench_long.evaluation.submission import validate_claims, validate_predictions
from pertbench_long.hashes import sha256_file
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.executor import PythonExecutor
from pertbench_long.runtime.sandbox import PathGuard
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.types import PublicEpisodeSpec
from pertbench_long.schemas.validate import parse_resource_profile

SAFE_READ_SUFFIX = {".h5ad", ".parquet", ".tsv", ".csv", ".json", ".txt", ".npz"}


@dataclass
class ArtifactRecord:
    artifact_id: str
    logical_path: str
    container_path: str
    sha256: str
    bytes: int
    kind: str
    visibility: str
    evidence_version: int

    def to_public(self) -> dict[str, Any]:
        return asdict(self)


class SnapshotIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {"snapshots": [], "closed_versions": [], "final": None}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def latest_for_version(self, version: int) -> dict[str, Any] | None:
        items = [s for s in self.data["snapshots"] if int(s["evidence_version"]) == int(version)]
        return items[-1] if items else None


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
        trusted_snapshot_dir: Path | None = None,
        runtime_budget: RuntimeBudget | None = None,
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
        self.trusted_snapshot_dir = Path(trusted_snapshot_dir) if trusted_snapshot_dir else (self.workspace.parent / "trusted_snapshots")
        self.trusted_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_index = SnapshotIndex(self.trusted_snapshot_dir / "index.json")
        self.registry: dict[str, ArtifactRecord] = {}
        self.runtime_budget = runtime_budget
        self.progress_path = self.workspace.parent / "run_progress.json"
        self.run_identity = None
        self._bootstrap_registry()
        # Resume must not overwrite a durable checkpoint with a fresh zeroed budget.
        if not self.progress_path.exists():
            self.persist_progress()

    def _bootstrap_registry(self) -> None:
        for artifact_id in list(self.public_spec.initial_evidence) + list(self.public_spec.reference_evidence):
            name = f"{artifact_id}.h5ad" if not artifact_id.endswith(".h5ad") else artifact_id
            ws = self.workspace / "evidence" / name
            pub = self.public_root / "evidence" / name
            src = ws if ws.exists() else pub
            if not src.exists():
                continue
            rec = ArtifactRecord(
                artifact_id=artifact_id.replace(".h5ad", ""),
                logical_path=f"evidence/{name}",
                container_path=f"/workspace/evidence/{name}",
                sha256=sha256_file(src),
                bytes=src.stat().st_size,
                kind="evidence",
                visibility="visible",
                evidence_version=0,
            )
            self.registry[rec.artifact_id] = rec
        gene_name = Path(self.public_spec.gene_universe_artifact).name
        gene_src = self.workspace / gene_name
        if not gene_src.exists():
            gene_src = self.public_root / self.public_spec.gene_universe_artifact
        if gene_src.exists():
            self.registry["gene_universe"] = ArtifactRecord(
                artifact_id="gene_universe",
                logical_path=gene_name,
                container_path=f"/workspace/{gene_name}",
                sha256=sha256_file(gene_src),
                bytes=gene_src.stat().st_size,
                kind="protocol",
                visibility="visible",
                evidence_version=0,
            )
        contract_src = self.workspace / "submission_contract.json"
        if not contract_src.exists():
            contract_src = self.public_root / "submission_contract.json"
        if contract_src.exists():
            self.registry["submission_contract"] = ArtifactRecord(
                artifact_id="submission_contract",
                logical_path="submission_contract.json",
                container_path="/workspace/submission_contract.json",
                sha256=sha256_file(contract_src),
                bytes=contract_src.stat().st_size,
                kind="protocol",
                visibility="visible",
                evidence_version=0,
            )

    def restore_from_ledger(self) -> None:
        """Rebuild visible registry and evidence_version from the durable ledger and snapshots."""
        snap = self.oracle.ledger.snapshot(self.run_id)
        if snap.status == "SUBMITTED":
            raise InvalidState("cannot resume a submitted run")
        for purchase in self.oracle.ledger.list_purchases(self.run_id):
            artifact = json.loads(purchase.get("artifact_json") or "{}")
            aid = str(artifact.get("artifact_id") or f"ev_{purchase['experiment_id']}")
            name = f"{aid}.h5ad" if not aid.endswith(".h5ad") else aid
            dest = self.workspace / "evidence" / name
            if not dest.exists():
                raise InvalidState(f"purchased artifact {aid} is missing from the workspace and cannot be restored")
            rec = ArtifactRecord(
                artifact_id=aid.replace(".h5ad", ""),
                logical_path=f"evidence/{name}",
                container_path=f"/workspace/evidence/{name}",
                sha256=str(artifact.get("sha256") or sha256_file(dest)),
                bytes=int(artifact.get("bytes") or dest.stat().st_size),
                kind="evidence",
                visibility="visible",
                evidence_version=int(snap.unique_purchases),
            )
            if rec.sha256 and sha256_file(dest) != rec.sha256:
                raise InvalidState(f"purchased artifact {aid} hash does not match the ledger")
            self.registry[rec.artifact_id] = rec
        self.evidence_version = int(snap.unique_purchases)
        closed = self.snapshot_index.data.get("closed_versions") or []
        if self.evidence_version and not closed:
            self.snapshot_index.data["closed_versions"] = list(range(self.evidence_version))
            self.snapshot_index.save()
        progress = self.load_progress()
        if progress:
            self.tool_calls = int((progress.get("runtime_budget") or {}).get("tool_calls") or progress.get("tool_calls") or 0)
            if self.runtime_budget is not None:
                remaining = (progress.get("runtime_budget") or {}).get("remaining_s")
                self.runtime_budget.restore(progress.get("runtime_budget") or {}, remaining_s=remaining)
                self.tool_calls = self.runtime_budget.tool_calls
        self._reconcile_visible_purchases()
        self.persist_progress()

    def load_progress(self) -> dict[str, Any] | None:
        if not self.progress_path.exists():
            return None
        return json.loads(self.progress_path.read_text(encoding="utf-8"))

    def persist_progress(self) -> None:
        payload = {
            "run_id": self.run_id,
            "evidence_version": self.evidence_version,
            "closed_versions": list(self.snapshot_index.data.get("closed_versions") or []),
            "tool_calls": self.tool_calls,
            "runtime_budget": self.runtime_budget.snapshot() if self.runtime_budget is not None else None,
            "visible_evidence_ids": self.visible_evidence_ids(),
            "run_identity": getattr(self, "run_identity", None),
        }
        tmp = self.progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.progress_path)

    def _close_and_advance(self, new_version: int) -> None:
        new_version = int(new_version)
        if new_version <= self.evidence_version:
            return
        closed = self.snapshot_index.data.setdefault("closed_versions", [])
        if self.evidence_version not in closed:
            closed.append(self.evidence_version)
            self.snapshot_index.save()
        self.evidence_version = new_version

    def _reconcile_visible_purchases(self) -> None:
        """Ledger unique_purchases is the source of truth after any reveal."""
        snap = self.oracle.ledger.snapshot(self.run_id)
        for purchase in self.oracle.ledger.list_purchases(self.run_id):
            artifact = json.loads(purchase.get("artifact_json") or "{}")
            aid = str(artifact.get("artifact_id") or f"ev_{purchase['experiment_id']}")
            name = f"{aid}.h5ad" if not aid.endswith(".h5ad") else aid
            dest = self.workspace / "evidence" / name
            if not dest.exists():
                continue
            rec = ArtifactRecord(
                artifact_id=aid.replace(".h5ad", ""),
                logical_path=f"evidence/{name}",
                container_path=f"/workspace/evidence/{name}",
                sha256=str(artifact.get("sha256") or sha256_file(dest)),
                bytes=int(artifact.get("bytes") or dest.stat().st_size),
                kind="evidence",
                visibility="visible",
                evidence_version=int(snap.unique_purchases),
            )
            self.registry[rec.artifact_id] = rec
        self._close_and_advance(int(snap.unique_purchases))
        self.persist_progress()

    def visible_evidence_ids(self) -> list[str]:
        return [k for k, v in self.registry.items() if v.kind == "evidence" and v.visibility == "visible"]

    def log(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), "run_id": self.run_id, "evidence_version": self.evidence_version, **event}
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def _count_tool(self) -> None:
        if self.runtime_budget is not None:
            self.runtime_budget.count_tool()
            self.tool_calls = self.runtime_budget.tool_calls
            self.persist_progress()
            return
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise InvalidState("external tool call budget exceeded")
        self.persist_progress()

    def list_evidence(self) -> list[dict[str, Any]]:
        self._count_tool()
        items = [rec.to_public() for rec in sorted(self.registry.values(), key=lambda r: r.artifact_id) if rec.visibility == "visible"]
        self.log({"action": "list_evidence", "n": len(items), "artifact_ids": [i["artifact_id"] for i in items]})
        return items

    def inspect_artifact(self, artifact_id_or_path: str) -> dict[str, Any]:
        self._count_tool()
        rec = self._resolve_record(artifact_id_or_path)
        path = self._host_path(rec)
        preview: dict[str, Any] = {"artifact": rec.to_public()}
        suffix = path.suffix.lower()
        try:
            if suffix == ".h5ad":
                import anndata as ad

                adata = ad.read_h5ad(path, backed=None)
                preview.update(
                    {
                        "shape": [int(adata.n_obs), int(adata.n_vars)],
                        "obs_columns": [str(c) for c in adata.obs.columns][:32],
                        "gene_ids_head": [str(g) for g in list(adata.var_names)[:12]],
                        "obs_head": adata.obs.head(5).astype(str).to_dict(orient="records"),
                    }
                )
            elif suffix == ".parquet":
                import pandas as pd

                frame = pd.read_parquet(path)
                preview.update({"shape": list(frame.shape), "columns": [str(c) for c in frame.columns], "head": frame.head(5).astype(str).to_dict(orient="records")})
            elif suffix in {".tsv", ".csv", ".txt", ".json"}:
                text = path.read_text(encoding="utf-8")[:2000]
                preview["text_preview"] = text
        except Exception as exc:
            preview["preview_error"] = type(exc).__name__
        self.log({"action": "inspect_artifact", "artifact_id": rec.artifact_id})
        return preview

    def _resolve_record(self, token: str) -> ArtifactRecord:
        if token in self.registry:
            return self.registry[token]
        for rec in self.registry.values():
            if rec.logical_path == token or rec.container_path == token or rec.logical_path.endswith(token):
                return rec
        # Accept evidence/ev_x.h5ad after purchase even if caller uses public/ prefix mistakenly
        cleaned = token.replace("public/", "")
        for rec in self.registry.values():
            if rec.logical_path == cleaned:
                return rec
        raise ToolObservationError(f"artifact {token!r} is not visible")

    def map_agent_path(self, raw: str) -> Path:
        """Map container /workspace paths and relative paths onto the host workspace, then PathGuard."""
        text = str(raw).replace("\\", "/").strip()
        if not text:
            raise PathGuardError("path is outside the allowed workspace")
        if text == "/workspace" or text.startswith("/workspace/"):
            rel = text[len("/workspace") :].lstrip("/")
            candidate = self.workspace / rel if rel else self.workspace
            return self.guard.check(candidate)
        if text.startswith("/"):
            raise PathGuardError("host absolute paths are not allowed")
        if text.startswith("~") or ".." in Path(text).parts:
            # still allow PathGuard to reject traversal after join, but fail closed on ~
            if text.startswith("~"):
                raise PathGuardError("path is outside the allowed workspace")
        return self.guard.check(self.workspace / text)

    def _host_path(self, rec: ArtifactRecord) -> Path:
        if rec.kind in {"protocol", "analysis"}:
            return self.map_agent_path(rec.logical_path)
        name = Path(rec.logical_path).name
        ws = self.workspace / "evidence" / name
        if ws.exists():
            return self.guard.check(ws)
        pub = self.public_root / "evidence" / name
        return self.guard.check(pub)

    def read_artifact(self, relative: str) -> Path:
        self._count_tool()
        try:
            rec = self._resolve_record(relative)
            path = self._host_path(rec)
        except ToolObservationError:
            self.map_agent_path(relative)
            raise PathGuardError("artifact is not in the visible registry") from None
        if path.suffix.lower() not in SAFE_READ_SUFFIX:
            raise PathGuardError("artifact format is not allowed")
        if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
            raise PathGuardError("pickle/joblib artifacts are not allowed")
        self.log({"action": "read_artifact", "path": rec.logical_path, "sha256": rec.sha256})
        return path

    def list_experiments(self) -> list[dict[str, Any]]:
        self._count_tool()
        return self.oracle.list_experiments()

    def get_budget(self) -> dict[str, Any]:
        self._count_tool()
        payload = self.oracle.get_budget(self.run_id)
        if self.runtime_budget is not None:
            payload = {**payload, "runtime": self.runtime_budget.snapshot()}
        return payload

    def last_snapshot_version(self) -> Optional[int]:
        latest = self.snapshot_index.latest_for_version(self.evidence_version)
        return None if latest is None else int(latest["evidence_version"])

    def request_experiment(self, experiment_id: str, request_id: str) -> dict[str, Any]:
        self.state.require_running("request_experiment")
        existing = self.oracle.ledger.get_request(self.run_id, request_id)
        owned = self.oracle.ledger.get_purchase(self.run_id, experiment_id)
        if self.last_snapshot_version() != self.evidence_version and existing is None and owned is None:
            raise InvalidState("a frozen prediction snapshot for the current evidence version is required before purchase")
        self._count_tool()
        try:
            result = self.oracle.request_experiment(
                run_id=self.run_id,
                experiment_id=experiment_id,
                request_id=request_id,
                evidence_dir=self.workspace / "evidence",
            )
        except Exception:
            self._reconcile_visible_purchases()
            dest = self.workspace / "evidence" / f"ev_{experiment_id}.h5ad"
            if dest.exists() or self.oracle.ledger.get_purchase(self.run_id, experiment_id):
                if dest.exists():
                    raise IntegrityError(
                        "purchase evidence is visible but the ledger/broker versions were not coordinated; the run must stop"
                    ) from None
            raise
        self._reconcile_visible_purchases()
        if result.get("status") == "ok":
            artifact = result["artifact"]
            aid = artifact["artifact_id"]
            dest = self.workspace / "evidence" / f"{aid}.h5ad"
            rec = ArtifactRecord(
                artifact_id=aid,
                logical_path=f"evidence/{aid}.h5ad",
                container_path=f"/workspace/evidence/{aid}.h5ad",
                sha256=artifact["sha256"],
                bytes=int(artifact.get("bytes") or dest.stat().st_size),
                kind="evidence",
                visibility="visible",
                evidence_version=int(result["evidence_version"]),
            )
            self.registry[aid] = rec
            result = {**result, "evidence_version": self.evidence_version}
        self.log({"action": "request_experiment", "experiment_id": experiment_id, "request_id": request_id, "result": result})
        self.persist_progress()
        return result

    def _gene_ids(self) -> list[str]:
        path = self.public_root / self.public_spec.gene_universe_artifact
        return path.read_text(encoding="utf-8").strip().splitlines()[1:]

    def save_snapshot(self, prediction_path: str, claims_path: str | None = None) -> dict[str, Any]:
        self.state.require_running("save_snapshot")
        self._count_tool()
        if self.evidence_version in self.snapshot_index.data.get("closed_versions", []):
            raise InvalidState("this evidence version is closed; later snapshots cannot backfill it")
        pred = self.map_agent_path(prediction_path)
        import pandas as pd

        staging = self.trusted_snapshot_dir / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
        staged_pred = staging / f"pred_{self.evidence_version}_{time.time_ns()}.parquet"
        shutil.copyfile(pred, staged_pred)
        frame = pd.read_parquet(staged_pred)
        target_ids = [t.target_id for t in self.public_spec.targets]
        gene_ids = self._gene_ids()
        validate_predictions(frame, target_ids=target_ids, gene_ids=gene_ids)
        pred_hash = sha256_file(staged_pred)
        claims_hash = None
        staged_claims = None
        if claims_path:
            cpath = self.map_agent_path(claims_path)
            staged_claims = staging / f"claims_{self.evidence_version}_{time.time_ns()}.json"
            shutil.copyfile(cpath, staged_claims)
            claims = json.loads(staged_claims.read_text(encoding="utf-8"))
            validate_claims(claims, target_ids=target_ids, gene_ids=gene_ids, evidence_ids=self.visible_evidence_ids())
            claims_hash = sha256_file(staged_claims)
        existing = [s for s in self.snapshot_index.data["snapshots"] if int(s["evidence_version"]) == self.evidence_version]
        rev = len(existing)
        dest_dir = self.trusted_snapshot_dir / f"v{self.evidence_version}" / f"rev{rev}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "predictions.parquet"
        shutil.copyfile(staged_pred, dest)
        if sha256_file(dest) != pred_hash:
            raise InvalidState("snapshot hash changed during freeze")
        payload = {
            "snapshot_id": f"snap_v{self.evidence_version}_rev{rev}",
            "evidence_version": self.evidence_version,
            "purchase_count": self.evidence_version,
            "prediction_sha256": pred_hash,
            "claims_sha256": claims_hash,
            "bytes": dest.stat().st_size,
            "path": str(dest),
            "seq": len(self.snapshot_index.data["snapshots"]),
            "ts": time.time(),
        }
        if staged_claims is not None:
            cdest = dest_dir / "claims.json"
            shutil.copyfile(staged_claims, cdest)
        self.snapshot_index.data["snapshots"].append(payload)
        self.snapshot_index.save()
        self.log({"action": "save_snapshot", **payload})
        return {"status": "ok", "format": "valid", "evidence_version": self.evidence_version, "snapshot_id": payload["snapshot_id"]}

    def submit(self, prediction_path: str, claims_path: str, stop_reason: str = "submitted") -> dict[str, Any]:
        self.state.require_running("submit")
        result = self.save_snapshot(prediction_path, claims_path)
        latest = self.snapshot_index.latest_for_version(self.evidence_version)
        if latest is None:
            raise SubmissionInvalid("final snapshot is missing")
        trusted = self.workspace.parent / "trusted_submissions" / self.run_id
        trusted.mkdir(parents=True, exist_ok=True)
        dest = trusted / "predictions.parquet"
        shutil.copyfile(latest["path"], dest)
        if sha256_file(dest) != latest["prediction_sha256"]:
            raise SubmissionInvalid("final snapshot hash mismatch")
        claims_src = Path(latest["path"]).parent / "claims.json"
        if claims_src.exists():
            shutil.copyfile(claims_src, trusted / "claims.json")
        self.trusted_submission = {
            "prediction_sha256": sha256_file(dest),
            "bytes": dest.stat().st_size,
            "path": str(dest),
            "stop_reason": stop_reason,
            "evidence_version": self.evidence_version,
            "snapshot_id": latest["snapshot_id"],
        }
        self.snapshot_index.data["final"] = self.trusted_submission
        self.snapshot_index.save()
        self.oracle.ledger.set_status(self.run_id, "SUBMITTED")
        self.state.transition(RunState.SUBMITTED)
        self.log({"action": "submit", **self.trusted_submission})
        return {"status": "ok", "receipt": {"bytes": self.trusted_submission["bytes"], "sha256": self.trusted_submission["prediction_sha256"]}}

    def _locate(self, relative: str) -> Path:
        rec = None
        try:
            rec = self._resolve_record(relative)
        except ToolObservationError:
            rec = None
        if rec is not None:
            return self._host_path(rec)
        rel = Path(relative)
        candidates = [self.workspace / rel, self.public_root / rel, self.workspace / "evidence" / rel.name]
        for cand in candidates:
            if cand.exists():
                return cand
        return self.workspace / rel

    def run_python(self, code: str, executor: PythonExecutor) -> dict[str, Any]:
        self.state.require_running("python")
        self._count_tool()
        started = time.time()
        timeout_s = self.tool_timeout_s
        if self.runtime_budget is not None:
            self.runtime_budget.check(next_tool=False)
            timeout_s = self.runtime_budget.tool_timeout_s(self.tool_timeout_s)
        result = executor.run_python(code, timeout_s=timeout_s, cwd=self.workspace)
        self.log(
            {
                "action": "python",
                "elapsed_s": time.time() - started,
                "ok": result.get("ok"),
                "exit_code": result.get("exit_code"),
                "code_sha256": __import__("hashlib").sha256(code.encode("utf-8")).hexdigest(),
                "stdout_tail": (result.get("stdout") or "")[-500:],
                "stderr_tail": (result.get("stderr") or "")[-500:],
            }
        )
        return result

    def run_shell(self, command: str, executor: PythonExecutor) -> dict[str, Any]:
        self.state.require_running("shell")
        self._count_tool()
        result = executor.run_shell(command, timeout_s=self.tool_timeout_s, cwd=self.workspace)
        self.log({"action": "shell", "ok": result.get("ok"), "command": command})
        return result
