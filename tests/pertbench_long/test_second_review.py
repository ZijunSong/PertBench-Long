"""Regressions for the second readiness review (B01–B11) against b7cfeac."""

from __future__ import annotations

import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import yaml

from pertbench_long.agents.client import ModelClient
from pertbench_long.agents.llm import LLMAdapter
from pertbench_long.cli import main
from pertbench_long.errors import InvalidState, PathGuardError
from pertbench_long.evaluation.outcomes import summarize_runs
from pertbench_long.oracle.ledger import PHASE_FAILED, Ledger
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.executor import isolation_is_qualified
from pertbench_long.runtime.release import check_release
from pertbench_long.runtime.runner import EpisodeRunner, score_run
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload


def _oracle(episode: dict, tmp_path: Path) -> tuple[Oracle, Broker]:
    public_dir = Path(episode["public_dir"])
    manifest = Path(episode["private_manifest"])
    spec = validate_private_payload(json.loads(manifest.read_text()))
    public = validate_public_payload(json.loads((public_dir / "episode.json").read_text()))
    run_id = "run_test"
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.init_run(run_id, public.episode_id, public.experimental_budget)
    oracle = Oracle(spec, ledger=ledger, private_root=manifest.parent, public_root=public_dir)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    (workspace / "evidence").mkdir()
    (workspace / "outputs").mkdir()
    for src in (public_dir / "evidence").iterdir():
        if src.suffix == ".h5ad":
            (workspace / "evidence" / src.name).write_bytes(src.read_bytes())
    for name in ("genes_v1.tsv", "episode.json", "submission_contract.json"):
        src = public_dir / name
        if src.exists():
            (workspace / name).write_bytes(src.read_bytes())
    state = StateMachine(RunState.RUNNING)
    broker = Broker(
        public_spec=public,
        oracle=oracle,
        workspace=workspace,
        public_root=public_dir,
        private_root=manifest.parent,
        state=state,
        run_id=run_id,
        event_log=tmp_path / "events.jsonl",
        trusted_snapshot_dir=tmp_path / "trusted_snapshots",
    )
    return oracle, broker


def _write_pred(broker: Broker) -> tuple[str, str]:
    from pertbench_long.baselines.engine import estimate, _write_pred as write_pred

    effects, probs = estimate(broker)
    return write_pred(broker, effects, probs)


class _ScriptedLLMHandler(BaseHTTPRequestHandler):
    script: list[dict] = []
    captured: list[dict] = []

    def log_message(self, format, *args):
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(n).decode("utf-8"))
        type(self).captured.append(payload)
        step = min(len(type(self).captured) - 1, len(type(self).script) - 1)
        body = json.dumps(type(self).script[step]).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _tool_response(name: str, arguments: dict, call_id: str = "call") -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


def test_b01_failed_request_same_id_does_not_free_reveal(synthetic_episode, tmp_path, monkeypatch):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    real = shutil.copyfile
    state = {"n": 0}

    def boom(src, dst, *args, **kwargs):
        state["n"] += 1
        if state["n"] <= 1:
            raise OSError("injected staging failure")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr("pertbench_long.oracle.service.shutil.copyfile", boom)
    with pytest.raises(OSError):
        broker.request_experiment("q_01", "rid-fail")
    snap = oracle.get_budget(broker.run_id)
    assert snap["charged_credits"] == 0
    assert snap["unique_purchases"] == 0
    assert not (broker.workspace / "evidence" / "ev_q_01.h5ad").exists()
    req = oracle.ledger.get_request(broker.run_id, "rid-fail")
    assert req["purchase_phase"] == PHASE_FAILED
    oracle.ledger.fail_and_refund(broker.run_id, "rid-fail")
    oracle.ledger.fail_and_refund(broker.run_id, "rid-fail")
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 0
    monkeypatch.setattr("pertbench_long.oracle.service.shutil.copyfile", real)
    with pytest.raises(InvalidState):
        broker.request_experiment("q_01", "rid-fail")
    assert not (broker.workspace / "evidence" / "ev_q_01.h5ad").exists()
    ok = broker.request_experiment("q_01", "rid-new")
    assert ok["status"] == "ok"
    assert ok["charged_credits"] == 1
    assert (broker.workspace / "evidence" / "ev_q_01.h5ad").exists()
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 1


def test_b02_resume_restores_purchase_and_refuses_submitted(synthetic_episode, tmp_path):
    resolved = {
        "mode": "local_trusted_debug",
        "agent": "scripted_mock",
        "public_dir": synthetic_episode["public_dir"],
        "private_manifest": synthetic_episode["private_manifest"],
        "seed": 0,
    }
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config=resolved,
    )
    ctx = runner._prepare()
    broker = ctx["broker"]
    started = json.loads((ctx["run_dir"] / "run_manifest.json").read_text())["started_utc"]
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    broker.request_experiment("q_01", "one")
    assert broker.evidence_version == 1
    resume = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config=resolved,
        resume_run_id=ctx["run_id"],
    )
    ctx2 = resume._prepare()
    assert ctx2["broker"].evidence_version == 1
    assert "ev_q_01" in ctx2["broker"].visible_evidence_ids()
    assert json.loads((ctx["run_dir"] / "run_manifest.json").read_text())["started_utc"] == started
    assert (ctx["run_dir"] / "resume_attempts.jsonl").exists()
    done = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config=resolved,
    )
    finished = done.run()
    assert finished["outcome"] == "completed"
    with pytest.raises(Exception):
        EpisodeRunner(
            public_dir=Path(synthetic_episode["public_dir"]),
            private_manifest=Path(synthetic_episode["private_manifest"]),
            run_root=Path(finished["run_dir"]).parent,
            mode="local_trusted_debug",
            agent="scripted_mock",
            resolved_config=resolved,
            resume_run_id=finished["run_id"],
        )._prepare()


def test_b03_isolation_qualified_requires_matching_digest():
    assert isolation_is_qualified(mode="isolated_eval", backend="docker", resolved_digest=None, attestation_digest="sha256:abc") is False
    assert isolation_is_qualified(mode="local_trusted_debug", backend="docker", resolved_digest="sha256:abc", attestation_digest="sha256:abc") is False
    assert isolation_is_qualified(
        mode="isolated_eval",
        backend="docker",
        resolved_digest="sha256:" + "a" * 64,
        attestation_digest="sha256:" + "a" * 64,
    )
    from pertbench_long.runtime.executor import DockerPythonExecutor, DebugPythonExecutor

    assert getattr(DockerPythonExecutor, "isolation_qualified", False) is not True
    debug = DebugPythonExecutor()
    assert debug.isolation_qualified is False


def test_b04_workspace_virtual_path_is_accepted(synthetic_episode, tmp_path):
    _, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    mapped = broker.map_agent_path("/workspace/outputs/predictions.parquet")
    assert mapped == (broker.workspace / "outputs" / "predictions.parquet").resolve()
    broker.save_snapshot("/workspace/outputs/predictions.parquet", "/workspace/outputs/claims.json")
    with pytest.raises(PathGuardError):
        broker.map_agent_path(str(Path(synthetic_episode["private_manifest"])))
    with pytest.raises(PathGuardError):
        broker.map_agent_path("/etc/passwd")


def test_b05_deadline_stops_get_budget(synthetic_episode, tmp_path):
    _, broker = _oracle(synthetic_episode, tmp_path)
    broker.runtime_budget = RuntimeBudget(deadline_monotonic=time.monotonic() - 0.01, max_tool_calls=100)
    with pytest.raises(TimeoutError, match="episode_timeout"):
        broker.get_budget()


def test_b06_seed_and_max_steps_reach_client_and_loop(tmp_path):
    captured = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            captured.append(payload)
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "no tool"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        client = ModelClient({"model": {"model": "m", "base_url": f"http://127.0.0.1:{port}/v1", "auth_mode": "none", "max_retries": 0}, "seed": 7})
        assert client.seed == 7
        client.generate([{"role": "user", "content": "x"}])
        assert captured[0].get("seed") == 7
        adapter = LLMAdapter({"model": {"model": "m", "base_url": f"http://127.0.0.1:{port}/v1", "auth_mode": "none", "max_retries": 0}, "max_agent_steps": 2, "runtime": {"max_agent_steps": 2}})
        assert adapter.config.get("max_agent_steps") == 2
    finally:
        server.shutdown()


def test_b07_http_401_is_infra_not_science_zero(synthetic_episode, tmp_path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            self.send_response(401)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        runner = EpisodeRunner(
            public_dir=Path(synthetic_episode["public_dir"]),
            private_manifest=Path(synthetic_episode["private_manifest"]),
            run_root=tmp_path / "runs",
            mode="local_trusted_debug",
            agent="llm_no_query",
            agent_kwargs={
                "model": {
                    "backend": "openai_compatible_chat",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "model": "mock",
                    "auth_mode": "none",
                    "max_retries": 0,
                }
            },
            resolved_config={"mode": "local_trusted_debug", "runtime": {"max_agent_steps": 2, "episode_timeout_s": 30}, "seed": 1},
        )
        result = runner.run()
        assert result["outcome"] == "infra_error"
        scores = score_run(Path(result["run_dir"]), Path(synthetic_episode["private_manifest"]))
        assert scores["direction_score"] is None
        assert scores["AUBC"] is None
        assert scores["score_status"] == "not_scored"
    finally:
        server.shutdown()


def test_b08_malformed_native_args_still_return_tool_result(synthetic_episode, tmp_path):
    genes = (Path(synthetic_episode["public_dir"]) / "genes_v1.tsv").read_text().splitlines()[1:]
    code = (
        "import json, pandas as pd\nfrom pathlib import Path\n"
        f"genes={genes!r}\n"
        "targets=['t_01','t_02']\n"
        "rows=[{'target_id':t,'gene_id':g,'predicted_effect':0.0,'p_down':0.0,'p_neutral':1.0,'p_up':0.0} for t in targets for g in genes]\n"
        "Path('outputs').mkdir(exist_ok=True)\n"
        "pd.DataFrame(rows).to_parquet('outputs/predictions.parquet', index=False)\n"
        "Path('outputs/claims.json').write_text(json.dumps({'schema_version':'1.0','claims':[{'claim_id':'c1','scope':{'targets':targets},'statement':'neutral','evidence_ids':['ev_initial_001','ev_controls_001'],'prediction_entries':[{'target_id':'t_01','gene_id':genes[0]}],'limitations':['mock']}]}))\n"
    )
    _ScriptedLLMHandler.script = [
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "get_budget", "arguments": "[1,2,3]"}}],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        _tool_response("run_python", {"code": code}, "c1"),
        _tool_response("save_prediction_snapshot", {"prediction_path": "/workspace/outputs/predictions.parquet", "claims_path": "/workspace/outputs/claims.json"}, "c2"),
        _tool_response("submit", {"prediction_path": "outputs/predictions.parquet", "claims_path": "outputs/claims.json"}, "c3"),
    ]
    _ScriptedLLMHandler.captured = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLMHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    cfg = {
        "mode": "local_trusted_debug",
        "allow_unqualified": True,
        "agent": "llm_no_query",
        "public_dir": synthetic_episode["public_dir"],
        "private_manifest": synthetic_episode["private_manifest"],
        "output": str(tmp_path / "llmrun"),
        "model": {
            "backend": "openai_compatible_chat",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "model": "mock-model",
            "auth_mode": "none",
            "action_format": "native_tools",
            "max_retries": 0,
            "request_timeout_s": 10,
        },
        "runtime": {"max_agent_steps": 8, "episode_timeout_s": 60},
    }
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        rc = main(["run", "--config", str(path)])
        assert rc == 0
        bodies = _ScriptedLLMHandler.captured
        assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "bad" for item in bodies for msg in item.get("messages") or [])
    finally:
        server.shutdown()
        server.server_close()


def test_b09_extra_and_tampered_public_artifacts_rejected(synthetic_episode, tmp_path):
    public = Path(synthetic_episode["public_dir"])
    private = Path(synthetic_episode["private_manifest"])
    extra = public / "evidence" / "ev_q_03.h5ad"
    shutil.copyfile(private.parent / "queryable" / "ev_q_03.h5ad", extra)
    with pytest.raises(Exception):
        check_release(public, private)
    rc = main(["validate", "--manifest", str(private), "--check-artifacts"])
    assert rc != 0
    extra.unlink()
    import anndata as ad

    path = public / "evidence" / "ev_initial_001.h5ad"
    adata = ad.read_h5ad(path)
    adata.X = np.zeros_like(adata.X)
    adata.write_h5ad(path)
    with pytest.raises(Exception):
        check_release(public, private)
    runner = EpisodeRunner(
        public_dir=public,
        private_manifest=private,
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        resolved_config={"mode": "local_trusted_debug"},
    )
    result = runner.run()
    assert result["outcome"] in {"invalid_episode", "integrity_failure"}
    assert result.get("run_id") is None


def test_b10_suite_does_not_pool_incompatible_tracks():
    rows = [
        {"run_id": "a", "outcome": "completed", "direction_score": 0.8, "score_status": "scored", "model": "m", "policy": "p", "seed": 0, "scoring_track": "official", "synthetic": False, "budget": 2, "isolation_qualified": False, "episode_id": "e1"},
        {"run_id": "b", "outcome": "completed", "direction_score": 0.2, "score_status": "scored", "model": "m", "policy": "p", "seed": 0, "scoring_track": "diagnostic", "synthetic": True, "budget": 2, "isolation_qualified": False, "episode_id": "e2"},
        {"run_id": "c", "outcome": "infra_error", "direction_score": None, "score_status": "not_scored", "reason": "HTTP 401", "model": "m", "policy": "p", "seed": 1, "scoring_track": "official", "synthetic": False, "budget": 2, "isolation_qualified": False, "episode_id": "e1"},
    ]
    summary = summarize_runs(rows, group_by=["model", "policy", "seed", "scoring_track", "synthetic", "budget", "isolation_qualified", "episode_id"])
    assert len(summary["groups"]) == 3
    official = [g for g in summary["groups"] if g["key"]["scoring_track"] == "official" and g["key"]["seed"] == 0][0]
    diagnostic = [g for g in summary["groups"] if g["key"]["scoring_track"] == "diagnostic"][0]
    assert official["mean_direction_score_including_failures"] == 0.8
    assert diagnostic["mean_direction_score_including_failures"] == 0.2
    assert summary["infra_errors"]
    assert summary["science_denominator"] == 2


@pytest.mark.skip(reason="Docker analysis image digest and boundary tests are unverified on this host")
def test_b11_docker_isolation_live_unverified():
    raise AssertionError("should be skipped")


@pytest.mark.skip(reason="live local/API model eval is unverified without user credentials and a running endpoint")
def test_b11_live_model_unverified():
    raise AssertionError("should be skipped")
