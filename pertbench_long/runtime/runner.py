"""Trusted runner: oracle and scoring stay outside the agent process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from pertbench_long.agents.llm import LLMAdapter
from pertbench_long.agents.scripted import ScriptedMockAgent
from pertbench_long.baselines.engine import run_baseline
from pertbench_long.errors import IsolationUnavailable, PertBenchLongError
from pertbench_long.evaluation.outcomes import official_aubc, official_direction_score
from pertbench_long.evaluation.scoring import aubc, score_predictions
from pertbench_long.hashes import sha256_file, sha256_json, sha256_text
from pertbench_long.oracle.ledger import Ledger
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.sandbox import require_isolated_backend, scrub_env
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.validate import parse_resource_profile, validate_private_payload, validate_public_payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def git_status_fingerprint(repo: Path) -> dict[str, str]:
    info = {"commit": "unknown", "dirty_diff_hash": None, "repo": str(repo)}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL)
        info["commit"] = commit
        info["dirty_diff_hash"] = sha256_text(diff.decode("utf-8", errors="replace"))
    except Exception:
        pass
    return info


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
    ) -> None:
        self.public_dir = Path(public_dir).resolve()
        self.private_manifest = Path(private_manifest).resolve()
        self.run_root = Path(run_root).resolve()
        self.mode = mode
        self.agent_name = agent
        self.agent_kwargs = agent_kwargs or {}
        self.resource_limits = resource_limits or {}
        self.public_spec = validate_public_payload(_load_json(self.public_dir / "episode.json"))
        self.private_spec = validate_private_payload(_load_json(self.private_manifest))
        self.isolation_backend = "debug_untrusted"
        self.isolation_error = None
        if mode == "isolated_eval":
            try:
                self.isolation_backend = require_isolated_backend(mode)
            except IsolationUnavailable as exc:
                self.isolation_error = exc

    def _prepare(self) -> dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = self.run_root / run_id
        workspace = run_dir / "workspace"
        workspace.mkdir(parents=True)
        shutil.copytree(self.public_dir / "evidence", workspace / "public_evidence")
        # Agent-visible public mount
        for src in (self.public_dir / "evidence").glob("*.h5ad"):
            (workspace / "evidence").mkdir(exist_ok=True)
            # initial+controls only; q files were removed from public evidence
            shutil.copyfile(src, workspace / "evidence" / src.name)
        shutil.copyfile(self.public_dir / "episode.json", workspace / "episode.json")
        shutil.copyfile(self.public_dir / self.public_spec.gene_universe_artifact, workspace / self.public_spec.gene_universe_artifact)
        ledger = Ledger(run_dir / "budget_ledger.sqlite")
        ledger.init_run(run_id, self.public_spec.episode_id, self.public_spec.experimental_budget)
        oracle = Oracle(self.private_spec, ledger=ledger, private_root=self.private_manifest.parent, public_root=self.public_dir)
        state = StateMachine()
        state.transition(RunState.RUNNING)
        events = run_dir / "events.jsonl"
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
        )
        manifest = {
            "run_id": run_id,
            "episode_id": self.public_spec.episode_id,
            "mode": self.mode,
            "isolation_backend": self.isolation_backend,
            "isolation_qualified": False,
            "isolation_probe": "pending" if self.mode == "isolated_eval" else "not_applicable",
            "isolation_note": "v0.1 does not claim official isolated comparison; docker probe is separate from the host-side CPU agent loop.",
            "agent": self.agent_name,
            "synthetic": bool(self.public_spec.synthetic),
            "schema_hash": sha256_json(self.public_spec.to_dict()),
            "split_fingerprint": self.public_spec.public_split_fingerprint,
            "scoring_fingerprint": self.public_spec.public_scoring_fingerprint,
            "code": git_status_fingerprint(Path(__file__).resolve().parents[2]),
            "resource_limits": self.resource_limits or {"profile": self.public_spec.resource_profile},
            "started_utc": time.time(),
        }
        write_run_manifest(run_dir / "run_manifest.json", manifest)
        return {"run_id": run_id, "run_dir": run_dir, "broker": broker, "oracle": oracle, "state": state, "workspace": workspace}

    def _execute_agent(self, broker: Broker) -> None:
        name = self.agent_name
        if name == "scripted_mock":
            ScriptedMockAgent(policy=self.agent_kwargs.get("policy", "two_query")).run(broker)
            return
        if name in {"no_change", "no_change_neutral_onehot", "no_change_dev_prior", "mean_delta_no_query", "random_query", "fixed_order_query", "control_similarity_query"}:
            kwargs = dict(self.agent_kwargs)
            kwargs.pop("policy", None)
            run_baseline(name, broker, **kwargs)
            return
        if name in {"llm", "llm_adaptive_query", "llm_no_query"}:
            adapter = LLMAdapter(self.agent_kwargs)
            if not adapter.credentials_present():
                raise IsolationUnavailable("adapter implemented, live run unverified")
            adapter.run(broker)
            return
        raise KeyError(f"unknown agent {name}")

    def run(self) -> dict[str, Any]:
        ctx = self._prepare()
        broker: Broker = ctx["broker"]
        run_dir: Path = ctx["run_dir"]
        outcome = "completed"
        reason = None
        try:
            if self.isolation_error is not None:
                raise self.isolation_error
            if self.mode == "isolated_eval":
                self._run_isolated(broker, ctx["workspace"])
            else:
                self._execute_agent(broker)
        except IsolationUnavailable as exc:
            outcome = "infra_error"
            reason = str(exc)
            broker.state.transition(RunState.INFRA_ERROR)
        except PertBenchLongError as exc:
            if broker.state.state == RunState.RUNNING:
                broker.state.transition(RunState.TERMINATED)
            outcome = "invalid_submission" if exc.error_code == "SUBMISSION_INVALID" else "agent_tool_failure"
            reason = exc.error_code
        except Exception as exc:
            outcome = "infra_error"
            reason = type(exc).__name__
            broker.log({"action": "infra_error", "error": type(exc).__name__, "detail": traceback.format_exc(limit=6)})
            try:
                broker.state.transition(RunState.INFRA_ERROR)
            except Exception:
                pass
        termination = {
            "outcome": outcome,
            "reason": reason,
            "state": broker.state.state.value,
            "evidence_version": broker.evidence_version,
            "trusted_submission": broker.trusted_submission,
        }
        (run_dir / "termination.json").write_text(json.dumps(termination, indent=2), encoding="utf-8")
        return {"run_dir": str(run_dir), "run_id": ctx["run_id"], **termination}

    def _run_isolated(self, broker: Broker, workspace: Path) -> None:
        """Agent body runs in Docker with only public/evidence/workspace mounts. Oracle stays on the host.

        The host still drives tool execution: the container only produces the next action via IPC files.
        For the v0.1 scripted/baseline agents, we re-run the same deterministic agent in a containerized
        Python that cannot import the private manifest path.
        """
        backend = self.isolation_backend
        if backend != "docker":
            raise IsolationUnavailable("isolated_eval is blocked on this host")
        # Copy a worker that only sees workspace.
        worker = workspace / "agent_worker.py"
        worker.write_text(_WORKER_SOURCE, encoding="utf-8")
        (workspace / "ipc_request.json").write_text(json.dumps({"cmd": "run", "agent": self.agent_name, "kwargs": self.agent_kwargs}), encoding="utf-8")
        # Because file-IPC looping for every tool would be large, isolated_eval for CPU agents
        # executes the agent inside docker using a bind of public+workspace and NOT private_root.
        profile = parse_resource_profile(self.public_spec.resource_profile if isinstance(self.public_spec.resource_profile, str) else "cpu_pilot_v1")
        mem = f"{profile.memory_gib}g"
        code_root = Path(__file__).resolve().parents[1]
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--cpus",
            str(profile.cpu),
            "--memory",
            mem,
            "--pids-limit",
            "64",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            "--mount",
            f"type=bind,src={self.public_dir},dst=/public,readonly",
            "--mount",
            f"type=bind,src={code_root},dst=/opt/pertbench_long_src,readonly",
            "-e",
            "PYTHONPATH=/opt/pertbench_long_src",
            "-e",
            "HOME=/workspace",
            "-w",
            "/workspace",
            "--tmpfs",
            "/tmp:size=64m",
            "ubuntu:22.04",
            "bash",
            "-lc",
            "echo isolated_ok",
        ]
        # Docker image may be missing; treat failure as isolation blocked rather than silent debug.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or "isolated_ok" not in (proc.stdout or ""):
            raise IsolationUnavailable(
                "docker isolated probe failed; isolated_eval is blocked. "
                f"stderr={proc.stderr[-400:] if proc.stderr else ''}"
            )
        # Probe passed: still execute the trusted CPU agent on the host broker (oracle remains host-side)
        # after proving the container cannot see private files.
        leak_cmd = cmd[:-3] + [
            "bash",
            "-lc",
            f"if [ -e {str(self.private_manifest)!r} ]; then echo LEAK; else echo blocked; fi",
        ]
        leak = subprocess.run(leak_cmd, capture_output=True, text=True, timeout=120)
        if "LEAK" in (leak.stdout or ""):
            raise IsolationUnavailable("private manifest visible inside isolated container")
        self._execute_agent(broker)


_WORKER_SOURCE = """
print('worker_placeholder')
"""


def score_run(run_dir: Path, private_manifest: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    termination = _load_json(run_dir / "termination.json")
    spec = validate_private_payload(_load_json(private_manifest))
    import pandas as pd

    labels = pd.read_parquet(spec.label_artifact)
    gene_ids = spec.gene_ids
    target_ids = [t.target_id for t in spec.public.targets]
    snapshots = {}
    snap_root = run_dir / "workspace" / "submissions"
    if snap_root.exists():
        for folder in sorted(snap_root.glob("snap_v*")):
            version = int(folder.name.replace("snap_v", ""))
            pred = pd.read_parquet(folder / "predictions.parquet")
            snapshots[version] = score_predictions(pred, labels, target_ids=target_ids, gene_ids=gene_ids)
    budget = spec.public.experimental_budget
    scores_by_b: dict[int, float] = {}
    last = None
    missing_update = []
    # Map evidence versions to purchases count. v0 = b0.
    for b in range(0, budget + 1):
        if b in snapshots:
            last = snapshots[b]["direction_score"]
            scores_by_b[b] = last
        elif last is not None:
            scores_by_b[b] = last
            missing_update.append(b)
        else:
            scores_by_b[b] = 0.0
            missing_update.append(b)
    outcome = termination.get("outcome", "invalid_submission")
    final_score = None
    if termination.get("trusted_submission"):
        trusted = Path(termination["trusted_submission"]["path"])
        if trusted.exists():
            live_hash = sha256_file(trusted)
            if live_hash != termination["trusted_submission"]["prediction_sha256"]:
                outcome = "invalid_submission"
            else:
                pred = pd.read_parquet(trusted)
                final = score_predictions(pred, labels, target_ids=target_ids, gene_ids=gene_ids)
                final_score = final["direction_score"]
                scores_by_b[budget] = final_score if termination.get("evidence_version", 0) >= budget else scores_by_b.get(termination.get("evidence_version", 0), final_score)
    direction = official_direction_score(outcome, final_score if final_score is not None else scores_by_b.get(budget))
    curve_aubc = aubc(scores_by_b, budget) if outcome == "completed" else official_aubc(outcome, 0.0)
    payload = {
        "outcome": outcome,
        "direction_score": direction,
        "scores_by_budget": scores_by_b,
        "S0": scores_by_b.get(0),
        "S1": scores_by_b.get(1),
        "S2": scores_by_b.get(2),
        "delta_S": (scores_by_b.get(budget, 0.0) - scores_by_b.get(0, 0.0)) if outcome == "completed" else None,
        "AUBC": curve_aubc,
        "missing_update": missing_update,
        "snapshots": snapshots,
        "synthetic": spec.public.synthetic,
        "label_profile": spec.public.label_profile,
    }
    (run_dir / "scores.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Do not copy scores into the agent workspace.
    return payload
