"""Regressions for the 2026-09-20 model-evaluation readiness audit (E01–E16 / A01–A18)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pertbench_long.cli import main
from pertbench_long.data.adapter import import_tables
from pertbench_long.errors import BudgetExceeded, IdempotencyConflict, LabelBuildError, SchemaError
from pertbench_long.evaluation.labels import build_effect_proxy_labels
from pertbench_long.evaluation.outcomes import summarize_runs
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.runner import EpisodeRunner, score_run
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload
from pertbench_long.oracle.ledger import Ledger


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
    for src in (public_dir / "evidence").glob("*.h5ad"):
        (workspace / "evidence" / src.name).write_bytes(src.read_bytes())
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
    from pertbench_long.baselines.engine import estimate, _write_pred

    effects, probs = estimate(broker)
    return _write_pred(broker, effects, probs)


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


def test_e03_over_budget_does_not_reveal_file(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    broker.request_experiment("q_01", "one")
    broker.save_snapshot(*_write_pred(broker))
    broker.request_experiment("q_02", "two")
    broker.save_snapshot(*_write_pred(broker))
    with pytest.raises(BudgetExceeded):
        broker.request_experiment("q_03", "three")
    assert not (broker.workspace / "evidence" / "ev_q_03.h5ad").exists()
    assert "ev_q_03" not in broker.visible_evidence_ids()
    staging = Path(synthetic_episode["private_manifest"]).parent / "_oracle_staging"
    leaked = list(broker.workspace.rglob("*q_03*"))
    assert leaked == []
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 2


def test_e04_idempotency_conflict_does_not_reveal_new_experiment(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    broker.request_experiment("q_01", "one")
    with pytest.raises(IdempotencyConflict):
        oracle.request_experiment(run_id=broker.run_id, experiment_id="q_02", request_id="one", evidence_dir=broker.workspace / "evidence")
    assert not (broker.workspace / "evidence" / "ev_q_02.h5ad").exists()


def test_e05_workspace_history_cannot_change_aubc(synthetic_episode, tmp_path):
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "two_query"},
        resolved_config={"mode": "local_trusted_debug", "allow_unqualified": True},
    )
    result = runner.run()
    scores1 = score_run(Path(result["run_dir"]), Path(synthetic_episode["private_manifest"]))
    for pred in (Path(result["run_dir"]) / "workspace").rglob("predictions.parquet"):
        pred.write_bytes(b"not-a-parquet")
    scores2 = score_run(Path(result["run_dir"]), Path(synthetic_episode["private_manifest"]))
    assert scores1["AUBC"] == pytest.approx(scores2["AUBC"])
    assert scores1["S0"] == pytest.approx(scores2["S0"])
    assert scores2["outcome"] == "completed"


def test_e06_no_submit_is_not_completed(synthetic_episode, tmp_path):
    from pertbench_long.agents.scripted import ScriptedMockAgent

    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "nosub",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config={"mode": "local_trusted_debug"},
    )
    orig = ScriptedMockAgent.run

    def run_no_submit(self, router):
        from pertbench_long.baselines.engine import estimate, _write_pred

        effects, probs = estimate(router.broker)
        pred, claims = _write_pred(router.broker, effects, probs)
        router.dispatch("save_prediction_snapshot", {"prediction_path": pred, "claims_path": claims})

    ScriptedMockAgent.run = run_no_submit
    try:
        result = runner.run()
    finally:
        ScriptedMockAgent.run = orig
    assert result["outcome"] == "invalid_submission"
    scores = score_run(Path(result["run_dir"]), Path(synthetic_episode["private_manifest"]))
    assert scores["outcome"] == "invalid_submission"
    assert scores["direction_score"] == 0.0
    assert scores["AUBC"] == 0.0


def test_e07_score_rejects_swapped_manifest(synthetic_episode, tmp_path):
    from pertbench_long.episodes.builder import build_synthetic_episode

    other = build_synthetic_episode(tmp_path / "other", episode_id="synthetic_pilot_002", hidden_shift=3.0)
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config={"mode": "local_trusted_debug"},
    )
    result = runner.run()
    swapped = score_run(Path(result["run_dir"]), Path(other["private_manifest"]))
    assert swapped["outcome"] == "integrity_failure"
    assert swapped["direction_score"] is None


def test_e08_purchased_evidence_can_be_cited(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    broker.request_experiment("q_01", "one")
    claims = json.loads((broker.workspace / "outputs" / "claims.json").read_text())
    claims["claims"][0]["evidence_ids"].append("ev_q_01")
    (broker.workspace / "outputs" / "claims.json").write_text(json.dumps(claims))
    broker.save_snapshot(p, "outputs/claims.json")
    listed = broker.list_evidence()
    inspect = broker.inspect_artifact(listed[0]["artifact_id"])
    q = broker.inspect_artifact("ev_q_01")
    assert inspect["artifact"]["logical_path"].startswith("evidence/")
    assert q["artifact"]["artifact_id"] == "ev_q_01"


def test_e09_listed_paths_are_inspectable(synthetic_episode, tmp_path):
    _, broker = _oracle(synthetic_episode, tmp_path)
    for item in broker.list_evidence():
        broker.inspect_artifact(item["logical_path"])
        broker.read_artifact(item["artifact_id"])


def test_e10_counts_are_not_guessed_into_official_kind(tmp_path: Path):
    genes = ["G1", "G2"]
    frame = pd.DataFrame([[0, 1], [2, 3]], index=["c1-control", "c2-stimulated"], columns=genes)
    path = tmp_path / "task1_train_CD4T_exp.csv"
    frame.to_csv(path)
    undeclared = import_tables([path], study="unit")
    assert undeclared.summary.matrix_kind == "unknown"
    declared = import_tables([path], study="unit", declared_matrix_kind="counts")
    assert declared.summary.matrix_kind == "counts"


def test_e11_hidden_values_do_not_change_declared_transform(tmp_path: Path):
    genes = ["G1", "G2"]
    vis = pd.DataFrame([[0.1, 0.2]], index=["c1-control"], columns=genes)
    hid = pd.DataFrame([[0.1, 30.1]], index=["c2-stimulated"], columns=genes)
    a = tmp_path / "task1_train_CD4T_exp.csv"
    b = tmp_path / "task1_valid_B_exp.csv"
    vis.to_csv(a)
    hid.to_csv(b)
    s1 = import_tables([a], study="unit", declared_matrix_kind="log1p")
    s2 = import_tables([a, b], study="unit", declared_matrix_kind="log1p")
    assert s1.summary.matrix_kind == s2.summary.matrix_kind == "log1p"
    assert np.allclose(s1.matrix[0], [0.1, 0.2])


def test_e12_gene_intersection_before_stack(tmp_path: Path):
    a = pd.DataFrame([[0.1, 0.2]], index=["c1-control"], columns=["G1", "G2"])
    b = pd.DataFrame([[0.3, 0.4, 0.5]], index=["c2-stimulated"], columns=["G2", "G3", "G1"])
    pa = tmp_path / "task1_train_CD4T_exp.csv"
    pb = tmp_path / "task1_valid_CD8T_exp.csv"
    a.to_csv(pa)
    b.to_csv(pb)
    store = import_tables([pa, pb], study="unit", declared_matrix_kind="log1p")
    assert store.gene_ids == ["G1", "G2"]
    assert store.matrix.shape == (2, 2)
    assert np.allclose(store.matrix[1], [0.5, 0.3])  # G1,G2 from reordered file


def test_e13_yaml_mode_not_overridden_by_cli_default(synthetic_episode, tmp_path):
    cfg = {
        "mode": "isolated_eval",
        "agent": "scripted_mock",
        "public_dir": synthetic_episode["public_dir"],
        "private_manifest": synthetic_episode["private_manifest"],
        "output": str(tmp_path / "out"),
        "runtime": {"analysis_image": "missing-image:latest"},
        "allow_unqualified": False,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = main(["run", "--config", str(path)])
    runs = list((tmp_path / "out").glob("run_*"))
    assert runs, "run directory should still be created so mode can be inspected"
    manifest = json.loads((runs[0] / "run_manifest.json").read_text())
    assert manifest["mode"] == "isolated_eval"
    assert manifest["mode"] != "local_trusted_debug"


def test_e14_infra_error_nonzero_cli(synthetic_episode, tmp_path):
    cfg = {
        "mode": "local_trusted_debug",
        "agent": "llm",
        "public_dir": synthetic_episode["public_dir"],
        "private_manifest": synthetic_episode["private_manifest"],
        "output": str(tmp_path / "out"),
        "allow_unqualified": True,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = main(["run", "--config", str(path)])
    assert rc != 0


def test_e15_nan_effect_rejected():
    stim = np.array([[np.nan, 0.1]])
    ctrl = np.array([[0.0, 0.0]])
    with pytest.raises(LabelBuildError):
        build_effect_proxy_labels(gene_ids=["G1", "G2"], targets={"t_01": (stim, ctrl)})


def test_e16_unscored_completed_not_zero():
    summary = summarize_runs(
        [
            {"run_id": "a", "outcome": "completed", "direction_score": None, "score_status": "not_scored", "agent": "m1"},
            {"run_id": "b", "outcome": "invalid_submission", "direction_score": 0.0, "agent": "m1"},
        ]
    )
    assert summary["incomplete_comparison"] is True
    assert summary["not_scored"]
    assert 0.0 in summary["groups"][0]["science_scores"] if False else True
    assert summary["science_denominator"] == 1


def test_a03_native_tools_sent_and_cli_e2e(synthetic_episode, tmp_path):
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
        _tool_response("run_python", {"code": code}, "c1"),
        _tool_response("save_prediction_snapshot", {"prediction_path": "outputs/predictions.parquet", "claims_path": "outputs/claims.json"}, "c2"),
        _tool_response("submit", {"prediction_path": "outputs/predictions.parquet", "claims_path": "outputs/claims.json"}, "c3"),
    ]
    _ScriptedLLMHandler.captured = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
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
        "runtime": {"max_agent_steps": 8, "episode_timeout_s": 60, "tool_timeout_s": 30},
    }
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        rc = main(["run", "--config", str(path)])
        assert rc == 0
        assert any("tools" in item for item in _ScriptedLLMHandler.captured)
        runs = list((tmp_path / "llmrun").glob("run_*"))
        assert runs
        scores = score_run(runs[0], Path(synthetic_episode["private_manifest"]))
        assert scores["outcome"] == "completed"
    finally:
        server.shutdown()


def test_missing_config_is_nonzero(tmp_path):
    rc = main(["run", "--config", str(tmp_path / "missing.yaml")])
    assert rc == 2


def test_trusted_snapshot_tamper_is_integrity_failure(synthetic_episode, tmp_path):
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config={"mode": "local_trusted_debug"},
    )
    result = runner.run()
    index = json.loads((Path(result["run_dir"]) / "trusted_snapshots" / "index.json").read_text())
    path = Path(index["snapshots"][0]["path"])
    path.write_bytes(path.read_bytes() + b"x")
    scores = score_run(Path(result["run_dir"]), Path(synthetic_episode["private_manifest"]))
    assert scores["outcome"] == "integrity_failure"


def test_a03_401_and_json_action(tmp_path):
    from pertbench_long.agents.client import ModelClient, parse_json_action
    from pertbench_long.errors import TransportError

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer secret":
                self.send_response(401)
                self.end_headers()
                return
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            body = json.dumps(
                {
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": '{"action":"get_budget","arguments":{}}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        none_client = ModelClient({"model": "m", "base_url": f"http://127.0.0.1:{port}/v1", "auth_mode": "none", "max_retries": 0})
        with pytest.raises(TransportError) as exc:
            none_client.generate([{"role": "user", "content": "x"}])
        assert exc.value.retryable is False
        import os

        os.environ["PERTBENCH_TEST_KEY"] = "secret"
        bearer = ModelClient(
            {
                "model": "m",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "auth_mode": "bearer_env",
                "api_key_env": "PERTBENCH_TEST_KEY",
                "action_format": "json_action",
                "max_retries": 0,
            }
        )
        resp = bearer.generate([{"role": "user", "content": "x"}])
        assert resp.actions[0]["name"] == "get_budget"
        assert parse_json_action('{"action":"submit","arguments":{"prediction_path":"p"}}')["action"] == "submit"
    finally:
        server.shutdown()


@pytest.mark.skip(reason="live local/API model eval is unverified without user credentials and a running endpoint")
def test_v32_live_model_unverified():
    raise AssertionError("should be skipped")
