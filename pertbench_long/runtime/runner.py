"""Trusted runner: oracle and scoring stay outside the agent process."""

from __future__ import annotations

import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from pertbench_long.agents.llm import LLMAdapter, policy_allows_purchase
from pertbench_long.agents.scripted import ScriptedMockAgent
from pertbench_long.baselines.engine import run_baseline
from pertbench_long.errors import (
    AgentIncomplete,
    ConfigError,
    IntegrityError,
    InvalidEpisode,
    IsolationUnavailable,
    PertBenchLongError,
    TransportError,
)
from pertbench_long.evaluation.outcomes import SCIENCE_FAILURES, json_safe, official_aubc, official_direction_score
from pertbench_long.evaluation.scoring import aubc, score_continuous_predictions, score_predictions
from pertbench_long.hashes import sha256_file, sha256_json, sha256_text
from pertbench_long.oracle.ledger import Ledger
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.config import first_defined, run_identity
from pertbench_long.runtime.executor import make_executor
from pertbench_long.runtime.release import check_release, copy_declared_public_inputs
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.runtime.tools import ToolRouter
from pertbench_long.schemas.validate import parse_resource_profile, validate_private_payload, validate_public_payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def git_status_fingerprint(repo: Path) -> dict[str, str]:
    info = {"commit": "unknown", "dirty_diff_hash": None, "repo": str(repo)}
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL)
        info["commit"] = commit
        info["dirty_diff_hash"] = sha256_text(diff.decode("utf-8", errors="replace"))
    except Exception:
        pass
    return info


def _label_path(spec, manifest_path: Path) -> Path:
    raw = Path(spec.label_artifact)
    if raw.is_absolute():
        return raw
    return Path(manifest_path).parent / raw


def bind_public_private(public_spec, private_spec) -> None:
    pub_a = public_spec.to_dict()
    pub_b = private_spec.public.to_dict()
    if sha256_json(pub_a) != sha256_json(pub_b):
        raise IntegrityError("public pack does not match the private-bound public spec")
    if private_spec.episode_id != public_spec.episode_id:
        raise IntegrityError("private episode_id does not match public pack")


AGENT_REGISTRY = {
    "scripted_mock": "available",
    "llm": "available",
    "llm_adaptive_query": "available",
    "llm_no_query": "available",
    "no_change": "available",
    "no_change_neutral_onehot": "available",
    "no_change_dev_prior": "available",
    "mean_delta_no_query": "available",
    "random_query": "available",
    "fixed_order_query": "available",
    "control_similarity_query": "available",
}


class EpisodeRunner:
    def __init__(
        self,
        *,
        public_dir: Path,
        private_manifest: Path,
        run_root: Path,
        mode: str = "local_trusted_debug",
        agent: str = "scripted_mock",
        agent_kwargs: dict[str, Any] | None = None,
        resource_limits: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        self.public_dir = Path(public_dir).resolve()
        self.private_manifest = Path(private_manifest).resolve()
        self.run_root = Path(run_root).resolve()
        self.mode = mode
        self.agent_name = agent
        self.agent_kwargs = agent_kwargs or {}
        self.resource_limits = resource_limits or {}
        self.resolved_config = resolved_config or {}
        self.resume_run_id = resume_run_id
        self.public_spec = validate_public_payload(_load_json(self.public_dir / "episode.json"))
        self.private_spec = validate_private_payload(_load_json(self.private_manifest))
        bind_public_private(self.public_spec, self.private_spec)
        self.isolation_backend = "debug_untrusted"
        self.isolation_error = None
        self.executor = None
        runtime = dict(self.resolved_config.get("runtime") or {})
        if mode == "isolated_eval":
            try:
                image = runtime.get("analysis_image") or self.resource_limits.get("analysis_image")
                profile = parse_resource_profile(
                    self.public_spec.resource_profile if isinstance(self.public_spec.resource_profile, str) else "cpu_pilot_v1"
                )
                attestation = dict(runtime.get("isolation_attestation") or {})
                self.executor = make_executor(
                    mode,
                    image=image,
                    memory_gib=int(first_defined(runtime.get("memory_gib"), profile.memory_gib, default=8)),
                    cpu=int(first_defined(runtime.get("cpu"), profile.cpu, default=2)),
                    required_digest=attestation.get("digest"),
                    workspace_gib=runtime.get("workspace_gib"),
                )
                self.isolation_backend = self.executor.backend
            except IsolationUnavailable as exc:
                self.isolation_error = exc
        else:
            self.executor = make_executor("local_trusted_debug")

    def _prepare(self) -> dict[str, Any]:
        release = check_release(self.public_dir, self.private_manifest)
        cap = (self.resolved_config.get("evaluation") or {}).get("experimental_budget_cap")
        if cap is not None and int(cap) != int(self.public_spec.experimental_budget):
            raise ConfigError(
                "evaluation.experimental_budget_cap differs from the episode budget; "
                "refusing to silently rewrite the scoring curve"
            )
        run_id = self.resume_run_id or f"run_{uuid.uuid4().hex[:12]}"
        run_dir = self.run_root / run_id
        workspace = run_dir / "workspace"
        original_manifest = None
        if self.resume_run_id:
            if not run_dir.exists():
                raise ConfigError(f"resume run {run_id} does not exist")
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                raise ConfigError("resume run is missing run_manifest.json")
            original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            orig_binding = original_manifest.get("binding") or {}
            for key, value in (
                ("episode_id", self.public_spec.episode_id),
                ("split_fingerprint", self.public_spec.public_split_fingerprint),
                ("scoring_fingerprint", self.public_spec.public_scoring_fingerprint),
                ("private_data_hash", self.private_spec.private_data_hash),
                ("gene_order_hash", self.private_spec.gene_order_hash),
                ("public_spec_hash", sha256_json(self.public_spec.to_dict())),
            ):
                if orig_binding.get(key) not in {None, value} and orig_binding.get(key) != value:
                    raise IntegrityError(f"resume binding mismatch for {key}")
            orig_cfg = original_manifest.get("resolved_config") or {}
            current_id = run_identity(self.resolved_config, agent=self.agent_name, mode=self.mode)
            original_id = original_manifest.get("run_identity") or run_identity(orig_cfg, agent=str(orig_cfg.get("agent") or self.agent_name), mode=str(orig_cfg.get("mode") or self.mode))
            if sha256_json(current_id) != sha256_json(original_id):
                raise IntegrityError("resume cannot change model, seed, or runtime limits; start a new run")
            progress_path = run_dir / "run_progress.json"
            if not progress_path.exists():
                raise ConfigError("resume requires run_progress.json; start a new run instead of inventing a fresh budget")
            if self.agent_name in {"llm", "llm_adaptive_query", "llm_no_query"} and not (run_dir / "agent_messages.json").exists():
                raise ConfigError("resume of an LLM run requires agent_messages.json; start a new run")
            term_path = run_dir / "termination.json"
            if term_path.exists():
                term = json.loads(term_path.read_text(encoding="utf-8"))
                if term.get("outcome") in {"completed"} or term.get("state") == "SUBMITTED":
                    raise ConfigError("run already submitted; resume refused")
            ledger_probe = Ledger(run_dir / "budget_ledger.sqlite")
            try:
                snap = ledger_probe.snapshot(run_id)
            except KeyError as exc:
                raise ConfigError("resume ledger is missing the original run") from exc
            if snap.status == "SUBMITTED":
                raise ConfigError("run already submitted; resume refused")
        else:
            workspace.mkdir(parents=True)
            (workspace / "evidence").mkdir(exist_ok=True)
            (workspace / "outputs").mkdir(exist_ok=True)
            copy_declared_public_inputs(self.public_dir, workspace, self.public_spec)
        ledger = Ledger(run_dir / "budget_ledger.sqlite")
        ledger.init_run(run_id, self.public_spec.episode_id, self.public_spec.experimental_budget)
        oracle = Oracle(self.private_spec, ledger=ledger, private_root=self.private_manifest.parent, public_root=self.public_dir)
        state = StateMachine()
        state.transition(RunState.RUNNING)
        events = run_dir / "events.jsonl"
        profile = parse_resource_profile(self.public_spec.resource_profile if isinstance(self.public_spec.resource_profile, str) else "cpu_pilot_v1")
        timeout_s = int(
            first_defined(
                (self.resolved_config.get("runtime") or {}).get("episode_timeout_s"),
                self.resource_limits.get("episode_timeout_s"),
                profile.episode_timeout_s,
                default=300,
            )
        )
        deadline = time.monotonic() + timeout_s
        token_cap = first_defined(
            (self.resolved_config.get("runtime") or {}).get("max_generation_tokens"),
            (self.resolved_config.get("model") or {}).get("max_generation_tokens") if isinstance(self.resolved_config.get("model"), dict) else None,
        )
        total_cap = (self.resolved_config.get("runtime") or {}).get("max_total_tokens")
        if self.resume_run_id:
            saved = json.loads((run_dir / "run_progress.json").read_text(encoding="utf-8"))
            remaining = (saved.get("runtime_budget") or {}).get("remaining_s")
            if remaining is not None:
                deadline = time.monotonic() + max(0.0, float(remaining))
        budget = RuntimeBudget(
            deadline_monotonic=deadline,
            max_generation_tokens=int(token_cap) if token_cap is not None else None,
            max_total_tokens=int(total_cap) if total_cap is not None else None,
            max_tool_calls=int(
                first_defined(self.resource_limits.get("max_external_tool_calls"), profile.max_external_tool_calls, default=200)
            ),
            max_observation_chars=int((self.resolved_config.get("runtime") or {}).get("max_observation_chars") or 8000),
            strict_token_cap=bool((self.resolved_config.get("runtime") or {}).get("strict_token_cap")),
        )
        broker = Broker(
            public_spec=self.public_spec,
            oracle=oracle,
            workspace=workspace,
            public_root=self.public_dir,
            private_root=self.private_manifest.parent,
            state=state,
            run_id=run_id,
            event_log=events,
            resource_limits=self.resource_limits,
            trusted_snapshot_dir=run_dir / "trusted_snapshots",
            runtime_budget=budget,
        )
        if self.resume_run_id:
            broker.restore_from_ledger()
        labels = _label_path(self.private_spec, self.private_manifest)
        binding = {
            "episode_id": self.public_spec.episode_id,
            "split_fingerprint": self.public_spec.public_split_fingerprint,
            "scoring_fingerprint": self.public_spec.public_scoring_fingerprint,
            "private_data_hash": self.private_spec.private_data_hash,
            "label_sha256": sha256_file(labels) if labels.exists() else None,
            "gene_order_hash": self.private_spec.gene_order_hash,
            "public_spec_hash": sha256_json(self.public_spec.to_dict()),
            "private_manifest_sha256": sha256_file(self.private_manifest),
            "scoring_track": (self.private_spec.scoring_config or {}).get("scoring_track", "official"),
            "release_check": {k: release[k] for k in ("declared_public_evidence", "public_spec_hash", "label_sha256") if k in release},
        }
        disk_quota_enforced = bool(getattr(self.executor, "disk_quota_enforced", False))
        isolation_qualified = bool(
            self.mode == "isolated_eval"
            and self.executor
            and getattr(self.executor, "isolation_qualified", False)
            and self.isolation_error is None
            and (disk_quota_enforced or not (self.resolved_config.get("runtime") or {}).get("workspace_gib"))
        )
        identity = run_identity(self.resolved_config, agent=self.agent_name, mode=self.mode)
        scoring_track = (self.private_spec.scoring_config or {}).get("scoring_track", "official")
        official_eligible = bool(isolation_qualified and scoring_track == "official" and not self.public_spec.synthetic)
        manifest = {
            "run_id": run_id,
            "episode_id": self.public_spec.episode_id,
            "mode": self.mode,
            "isolation_backend": self.isolation_backend,
            "isolation_qualified": isolation_qualified,
            "isolation_note": "qualified only when the pinned image digest matches a recorded isolation attestation. debug is never qualified. copying a digest string is not an acceptance test.",
            "image_digest": getattr(self.executor, "resolved_digest", None),
            "run_image_reference": getattr(self.executor, "run_reference", None),
            "disk_quota_enforced": disk_quota_enforced,
            "run_identity": identity,
            "official_eligible": official_eligible,
            "agent": self.agent_name,
            "synthetic": bool(self.public_spec.synthetic),
            "schema_hash": sha256_json(self.public_spec.to_dict()),
            "split_fingerprint": self.public_spec.public_split_fingerprint,
            "scoring_fingerprint": self.public_spec.public_scoring_fingerprint,
            "code": git_status_fingerprint(Path(__file__).resolve().parents[2]),
            "resource_limits": self.resource_limits or {"profile": self.public_spec.resource_profile},
            "resolved_config": {k: v for k, v in (self.resolved_config or {}).items() if k != "api_key"},
            "binding": binding,
            "started_utc": (original_manifest or {}).get("started_utc") or time.time(),
            "model": (self.resolved_config.get("model") or self.agent_kwargs),
            "experimental_budget": self.public_spec.experimental_budget,
        }
        broker.run_identity = identity
        broker.persist_progress()
        if self.resume_run_id:
            resume_log = run_dir / "resume_attempts.jsonl"
            with resume_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": time.time(), "binding": binding, "evidence_version": broker.evidence_version}, default=str) + "\n")
        else:
            write_run_manifest(run_dir / "run_manifest.json", manifest)
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "broker": broker,
            "oracle": oracle,
            "state": state,
            "workspace": workspace,
            "manifest": original_manifest or manifest,
            "deadline": deadline,
            "runtime_budget": budget,
        }

    def _execute_agent(self, broker: Broker) -> None:
        if self.executor is None:
            raise IsolationUnavailable("executor was not created")
        allow_purchase = policy_allows_purchase(self.agent_name) and self.agent_kwargs.get("allow_purchase", True)
        if self.agent_name == "llm_no_query":
            allow_purchase = False
        router = ToolRouter(broker, self.executor, allow_purchase=allow_purchase)
        name = self.agent_name
        if name == "scripted_mock":
            ScriptedMockAgent(policy=self.agent_kwargs.get("policy", "two_query")).run(router)
            return
        if name in {"no_change", "no_change_neutral_onehot", "no_change_dev_prior", "mean_delta_no_query", "random_query", "fixed_order_query", "control_similarity_query"}:
            kwargs = dict(self.agent_kwargs)
            kwargs.pop("policy", None)
            run_baseline(name, broker, **kwargs)
            return
        if name in {"llm", "llm_adaptive_query", "llm_no_query"}:
            cfg = dict(self.agent_kwargs)
            cfg.setdefault("policy", name)
            cfg["allow_purchase"] = allow_purchase
            runtime = dict(self.resolved_config.get("runtime") or {})
            cfg["runtime"] = {**runtime, **dict(cfg.get("runtime") or {})}
            if "max_agent_steps" not in cfg:
                cfg["max_agent_steps"] = first_defined(runtime.get("max_agent_steps"), 60)
            cfg["deadline_monotonic"] = getattr(broker, "_deadline", None)
            if broker.runtime_budget is not None:
                cfg["runtime_budget"] = broker.runtime_budget
                cfg["deadline_monotonic"] = broker.runtime_budget.deadline
            if self.resolved_config.get("seed") is not None:
                cfg["seed"] = self.resolved_config.get("seed")
            LLMAdapter(cfg).run(router)
            return
        raise ConfigError(f"unknown agent {name}")

    def run(self) -> dict[str, Any]:
        try:
            ctx = self._prepare()
        except InvalidEpisode as exc:
            return {
                "run_dir": str(self.run_root),
                "run_id": None,
                "outcome": "invalid_episode",
                "reason": exc.message,
                "state": "INFRA_ERROR",
                "trusted_submission": None,
            }
        except IntegrityError as exc:
            return {
                "run_dir": str(self.run_root),
                "run_id": None,
                "outcome": "integrity_failure",
                "reason": exc.message,
                "state": "INFRA_ERROR",
                "trusted_submission": None,
            }
        except PertBenchLongError as exc:
            return {
                "run_dir": str(self.run_root),
                "run_id": None,
                "outcome": "infra_error" if exc.error_code != "INTEGRITY_FAILURE" else "integrity_failure",
                "reason": exc.error_code,
                "state": "INFRA_ERROR",
                "trusted_submission": None,
            }
        broker: Broker = ctx["broker"]
        run_dir: Path = ctx["run_dir"]
        deadline = ctx.get("deadline")
        outcome = "invalid_submission"
        reason = "no_submit"
        try:
            if self.isolation_error is not None:
                raise self.isolation_error
            if broker.runtime_budget is not None:
                broker.runtime_budget.check()
            self._execute_agent(broker)
            if broker.runtime_budget is not None:
                broker.runtime_budget.check()
            elif deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("episode_timeout")
            if broker.state.state == RunState.SUBMITTED and broker.trusted_submission:
                outcome = "completed"
                reason = None
            else:
                outcome = "invalid_submission"
                reason = "no_submit"
                if broker.state.state == RunState.RUNNING:
                    broker.state.transition(RunState.TERMINATED)
                    broker.oracle.ledger.set_status(broker.run_id, "TERMINATED")
        except IsolationUnavailable as exc:
            outcome = "infra_error"
            reason = str(exc)
            try:
                broker.state.transition(RunState.INFRA_ERROR)
            except Exception:
                pass
        except TransportError as exc:
            outcome = "infra_error"
            reason = f"{exc.error_code}:{exc.message}"
            broker.log({"action": "transport_error", "error": exc.message, "details": {k: v for k, v in (exc.details or {}).items() if k != "authorization"}})
            try:
                broker.state.transition(RunState.INFRA_ERROR)
            except Exception:
                pass
        except AgentIncomplete:
            outcome = "invalid_submission"
            reason = "no_submit"
            if broker.state.state == RunState.RUNNING:
                broker.state.transition(RunState.TERMINATED)
        except TimeoutError as exc:
            outcome = "agent_timeout" if str(exc) in {"episode_timeout", "token_budget"} else "agent_timeout"
            reason = str(exc) or "episode_timeout"
            if broker.state.state == RunState.RUNNING:
                broker.state.transition(RunState.TERMINATED)
        except ConfigError as exc:
            outcome = "infra_error"
            reason = exc.error_code
            try:
                broker.state.transition(RunState.INFRA_ERROR)
            except Exception:
                pass
        except PertBenchLongError as exc:
            if broker.state.state == RunState.RUNNING:
                broker.state.transition(RunState.TERMINATED)
            if exc.error_code == "SUBMISSION_INVALID":
                outcome = "invalid_submission"
            elif exc.error_code == "INTEGRITY_FAILURE":
                outcome = "integrity_failure"
            elif exc.error_code in {"INVALID_EPISODE"}:
                outcome = "invalid_episode"
            elif exc.error_code in {"TRANSPORT_ERROR", "CONFIG_INVALID", "ISOLATION_UNAVAILABLE"}:
                outcome = "infra_error"
            else:
                outcome = "agent_tool_failure"
            reason = exc.error_code
        except Exception as exc:
            outcome = "infra_error"
            reason = type(exc).__name__
            broker.log({"action": "infra_error", "error": type(exc).__name__, "detail": traceback.format_exc(limit=6)})
            try:
                broker.state.transition(RunState.INFRA_ERROR)
            except Exception:
                pass
        if outcome == "completed" and broker.state.state != RunState.SUBMITTED:
            outcome = "invalid_submission"
            reason = "state_not_submitted"
        termination = {
            "outcome": outcome,
            "reason": reason,
            "state": broker.state.state.value,
            "evidence_version": broker.evidence_version,
            "trusted_submission": broker.trusted_submission,
            "isolation_qualified": bool(ctx["manifest"].get("isolation_qualified")),
            "official_eligible": bool(ctx["manifest"].get("official_eligible")),
            "runtime_budget": broker.runtime_budget.snapshot() if broker.runtime_budget is not None else None,
            "score_status": "not_scored",
        }
        (run_dir / "termination.json").write_text(json.dumps(termination, indent=2, default=str), encoding="utf-8")
        return {"run_dir": str(run_dir), "run_id": ctx["run_id"], **termination}


def _score_frame(pred, labels, spec, *, target_ids, gene_ids):
    scoring = spec.scoring_config or {}
    if spec.public.label_profile == "lognorm_cellmean_delta_v1" or scoring.get("scoring_profile") == "continuous_effect_aubc_v1":
        group = {t.target_id: (t.context_id or t.cell_type) for t in spec.public.targets} if spec.public.protocol == "context_campaign_acquisition_v1" else None
        return score_continuous_predictions(
            pred,
            labels,
            target_ids=target_ids,
            gene_ids=gene_ids,
            a_family=float(scoring.get("a_family") or 0.5),
            group_by_context=group,
            require_direction=False,
        )
    return score_predictions(pred, labels, target_ids=target_ids, gene_ids=gene_ids)


def score_run(run_dir: Path, private_manifest: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    termination = _load_json(run_dir / "termination.json")
    run_manifest = _load_json(run_dir / "run_manifest.json")
    spec = validate_private_payload(_load_json(private_manifest))
    binding = run_manifest.get("binding") or {}
    try:
        if spec.episode_id != run_manifest.get("episode_id"):
            raise IntegrityError("score manifest episode_id does not match the run")
        if spec.private_data_hash != binding.get("private_data_hash"):
            raise IntegrityError("private_data_hash does not match the bound run")
        if spec.public.public_split_fingerprint != binding.get("split_fingerprint"):
            raise IntegrityError("split fingerprint does not match the bound run")
        if spec.public.public_scoring_fingerprint != binding.get("scoring_fingerprint"):
            raise IntegrityError("scoring fingerprint does not match the bound run")
        if spec.gene_order_hash != binding.get("gene_order_hash"):
            raise IntegrityError("gene_order_hash does not match the bound run")
        labels_path = _label_path(spec, private_manifest)
        if binding.get("label_sha256") and sha256_file(labels_path) != binding.get("label_sha256"):
            raise IntegrityError("label artifact hash does not match the bound run")
        if sha256_file(private_manifest) != binding.get("private_manifest_sha256"):
            raise IntegrityError("private manifest file hash does not match the bound run")
    except IntegrityError as exc:
        payload = {
            "outcome": "integrity_failure",
            "score_status": "not_scored",
            "reason": exc.message,
            "direction_score": None,
            "AUBC": None,
            "synthetic": spec.public.synthetic,
            "label_profile": spec.public.label_profile,
        }
        (run_dir / "scores.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return payload

    import pandas as pd

    labels = pd.read_parquet(_label_path(spec, private_manifest))
    gene_ids = spec.gene_ids
    target_ids = [t.target_id for t in spec.public.targets]
    index_path = run_dir / "trusted_snapshots" / "index.json"
    if not index_path.exists():
        exec_outcome = termination.get("outcome", "invalid_submission")
        payload = {
            "outcome": "integrity_failure" if exec_outcome == "completed" else exec_outcome,
            "score_status": "not_scored" if exec_outcome not in SCIENCE_FAILURES else "scored",
            "reason": "trusted snapshot index missing",
            "direction_score": official_direction_score(exec_outcome, 0.0 if exec_outcome in SCIENCE_FAILURES else None),
            "AUBC": official_aubc(exec_outcome, 0.0 if exec_outcome in SCIENCE_FAILURES else None),
            "synthetic": spec.public.synthetic,
            "label_profile": spec.public.label_profile,
        }
        (run_dir / "scores.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return payload

    index = json.loads(index_path.read_text(encoding="utf-8"))
    snapshots = {}
    for item in index.get("snapshots") or []:
        version = int(item["evidence_version"])
        path = Path(item["path"])
        if not path.exists() or sha256_file(path) != item["prediction_sha256"]:
            payload = {
                "outcome": "integrity_failure",
                "score_status": "not_scored",
                "reason": f"trusted snapshot {item.get('snapshot_id')} hash mismatch",
                "direction_score": None,
                "AUBC": None,
                "synthetic": spec.public.synthetic,
                "label_profile": spec.public.label_profile,
            }
            (run_dir / "scores.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
            return payload
        pred = pd.read_parquet(path)
        snapshots[version] = _score_frame(pred, labels, spec, target_ids=target_ids, gene_ids=gene_ids)
    budget = spec.public.experimental_budget
    scores_by_b: dict[int, float] = {}
    last = None
    missing_update = []
    for b in range(0, budget + 1):
        if b in snapshots:
            last = snapshots[b]["direction_score"]
            scores_by_b[b] = last
        elif last is not None:
            scores_by_b[b] = last
            missing_update.append(b)
        else:
            missing_update.append(b)

    exec_outcome = termination.get("outcome", "invalid_submission")
    outcome = exec_outcome
    final_score = None
    trusted_meta = termination.get("trusted_submission") or index.get("final")
    if exec_outcome == "completed":
        if not trusted_meta:
            outcome = "invalid_submission"
        else:
            trusted = Path(trusted_meta["path"])
            if not trusted.exists() or sha256_file(trusted) != trusted_meta["prediction_sha256"]:
                outcome = "integrity_failure"
            else:
                pred = pd.read_parquet(trusted)
                final = _score_frame(pred, labels, spec, target_ids=target_ids, gene_ids=gene_ids)
                final_score = final["direction_score"]
                scores_by_b[min(budget, int(termination.get("evidence_version", 0)))] = final_score
    else:
        final_score = None

    if outcome == "integrity_failure":
        payload = {
            "outcome": outcome,
            "execution_outcome": exec_outcome,
            "score_status": "not_scored",
            "direction_score": None,
            "AUBC": None,
            "synthetic": spec.public.synthetic,
            "label_profile": spec.public.label_profile,
        }
        (run_dir / "scores.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return payload

    if outcome != "completed":
        direction = official_direction_score(outcome, 0.0 if outcome in SCIENCE_FAILURES else None)
        curve_aubc = official_aubc(outcome, 0.0 if outcome in SCIENCE_FAILURES else None)
        score_status = "scored" if outcome in SCIENCE_FAILURES else "not_scored"
        payload = {
            "outcome": outcome,
            "execution_outcome": exec_outcome,
            "score_status": score_status,
            "direction_score": direction,
            "scores_by_budget": scores_by_b,
            "S0": scores_by_b.get(0),
            "S1": scores_by_b.get(1),
            "S2": scores_by_b.get(2),
            "scores_curve": {str(i): scores_by_b.get(i) for i in range(0, budget + 1)},
            "delta_S": None,
            "AUBC": curve_aubc,
            "missing_update": missing_update,
            "synthetic": spec.public.synthetic,
            "label_profile": spec.public.label_profile,
            "scoring_track": (spec.scoring_config or {}).get("scoring_track", "official"),
            "reason": termination.get("reason"),
        }
        (run_dir / "scores.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return payload

    if 0 not in scores_by_b and final_score is not None:
        scores_by_b[0] = snapshots.get(0, {}).get("direction_score") if 0 in snapshots else final_score
    for b in range(0, budget + 1):
        if b not in scores_by_b and final_score is not None:
            scores_by_b[b] = final_score
    direction = official_direction_score(outcome, final_score)
    curve_aubc = aubc(scores_by_b, budget) if all(i in scores_by_b for i in range(0, budget + 1)) else None
    payload = {
        "outcome": outcome,
        "execution_outcome": exec_outcome,
        "score_status": "scored",
        "direction_score": direction,
        "scores_by_budget": scores_by_b,
        "S0": scores_by_b.get(0),
        "S1": scores_by_b.get(1),
        "S2": scores_by_b.get(2),
        "scores_curve": {str(i): scores_by_b.get(i) for i in range(0, budget + 1)},
        "delta_S": (scores_by_b.get(budget, 0.0) - scores_by_b.get(0, 0.0)) if outcome == "completed" else None,
        "AUBC": curve_aubc,
        "missing_update": missing_update,
        "snapshots": {str(k): v["direction_score"] for k, v in snapshots.items()},
        "synthetic": spec.public.synthetic,
        "label_profile": spec.public.label_profile,
        "scoring_track": (spec.scoring_config or {}).get("scoring_track", "official"),
    }
    (run_dir / "scores.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    updated = dict(termination)
    updated["score_status"] = "scored"
    updated["scored_outcome"] = outcome
    (run_dir / "termination.json").write_text(json.dumps(updated, indent=2, default=str), encoding="utf-8")
    return payload
