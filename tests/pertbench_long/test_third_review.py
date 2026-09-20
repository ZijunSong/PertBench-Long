"""Regressions for the third review (C01–C05) against ab9b880."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from pertbench_long.cli import main
from pertbench_long.errors import IntegrityError, InvalidState
from pertbench_long.oracle.ledger import Ledger
from pertbench_long.runtime.budget import RuntimeBudget
from pertbench_long.runtime.executor import DebugPythonExecutor, _bounded_communicate
from pertbench_long.runtime.runner import EpisodeRunner
from pertbench_long.runtime.tools import ToolRouter
from tests.pertbench_long.test_second_review import _oracle, _tool_response, _write_pred


def test_c01_visible_file_advances_version_after_deliver_fault(synthetic_episode, tmp_path, monkeypatch):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    real = Ledger.mark_delivered

    def boom(self, *args, **kwargs):
        raise OSError("injected mark_delivered failure")

    monkeypatch.setattr(Ledger, "mark_delivered", boom)
    with pytest.raises(IntegrityError):
        broker.request_experiment("q_01", "rid-c01")
    assert (broker.workspace / "evidence" / "ev_q_01.h5ad").exists()
    assert broker.evidence_version == 1
    assert 0 in broker.snapshot_index.data.get("closed_versions", [])
    later = broker.save_snapshot(*_write_pred(broker))
    assert later["snapshot_id"].startswith("snap_v1_")
    v0 = [s for s in broker.snapshot_index.data["snapshots"] if int(s["evidence_version"]) == 0]
    assert [s["snapshot_id"] for s in v0] == ["snap_v0_rev0"]
    monkeypatch.setattr(Ledger, "mark_delivered", real)
    retry = broker.request_experiment("q_01", "rid-c01")
    assert retry["status"] == "ok"
    assert retry["charged_credits"] == 0
    assert broker.evidence_version == 1
    again = broker.save_snapshot(*_write_pred(broker))
    assert again["snapshot_id"].startswith("snap_v1_")
    v0 = [s for s in broker.snapshot_index.data["snapshots"] if int(s["evidence_version"]) == 0]
    assert [s["snapshot_id"] for s in v0] == ["snap_v0_rev0"]
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 1


def test_c01_integrity_is_not_tool_observation(synthetic_episode, tmp_path, monkeypatch):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    p, c = _write_pred(broker)
    broker.save_snapshot(p, c)
    router = ToolRouter(broker, DebugPythonExecutor(), allow_purchase=True)

    def boom(self, *args, **kwargs):
        raise OSError("injected mark_delivered failure")

    monkeypatch.setattr(Ledger, "mark_delivered", boom)
    with pytest.raises(IntegrityError):
        router.dispatch("request_experiment", {"experiment_id": "q_01", "request_id": "rid-c01-router"})
    assert (broker.workspace / "evidence" / "ev_q_01.h5ad").exists()
    assert broker.evidence_version == 1


def test_c02_exact_cap_and_missing_usage_are_explicit():
    budget = RuntimeBudget(deadline_monotonic=time.monotonic() + 30, max_generation_tokens=5)
    budget.consume_usage({"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})
    budget.check_after_usage()
    budget.consume_usage({}, fallback_completion_tokens=3)
    assert budget.unknown_usage_calls == 1
    assert budget.completion_tokens == 5
    assert budget.estimated_completion_tokens == 3
    with pytest.raises(TimeoutError, match="token_budget"):
        budget.check_after_usage()
    snap = budget.snapshot()
    assert snap["unit"] == "completion_tokens"
    assert snap["stop_reason"] == "token_budget"


def test_c02_generation_cap_tightens_request_and_blocks_over_submit(synthetic_episode, tmp_path):
    budget = RuntimeBudget(deadline_monotonic=time.monotonic() + 30, max_generation_tokens=5)
    budget.consume_usage({"prompt_tokens": 9000, "completion_tokens": 1, "total_tokens": 9001})
    assert budget.completion_tokens == 1
    assert budget.remaining_generation() == 4
    assert budget.output_cap_for_call(2048) == 4

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
    from tests.pertbench_long.test_second_review import _ScriptedLLMHandler

    first = _tool_response("run_python", {"code": code}, "c1")
    first["usage"] = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    second = _tool_response(
        "submit",
        {"prediction_path": "outputs/predictions.parquet", "claims_path": "outputs/claims.json"},
        "c2",
    )
    second["usage"] = {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
    _ScriptedLLMHandler.script = [first, second]
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
        "output": str(tmp_path / "tok"),
        "model": {
            "backend": "openai_compatible_chat",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "model": "mock-model",
            "auth_mode": "none",
            "max_retries": 0,
            "max_output_tokens_per_call": 2048,
        },
        "runtime": {"max_agent_steps": 6, "episode_timeout_s": 60, "max_generation_tokens": 5},
    }
    path = tmp_path / "tok.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        rc = main(["run", "--config", str(path)])
        assert rc != 0
        assert all(int(item.get("max_tokens") or 0) <= 5 for item in _ScriptedLLMHandler.captured if "max_tokens" in item)
        runs = list((tmp_path / "tok").glob("run_*"))
        assert runs
        term = json.loads((runs[0] / "termination.json").read_text())
        assert term["outcome"] != "completed"
        assert term["outcome"] == "agent_timeout"
        assert term["reason"] == "token_budget"
    finally:
        server.shutdown()
        server.server_close()


def test_c03_resume_keeps_budget_and_rejects_model_change(synthetic_episode, tmp_path):
    resolved = {
        "mode": "local_trusted_debug",
        "agent": "scripted_mock",
        "public_dir": synthetic_episode["public_dir"],
        "private_manifest": synthetic_episode["private_manifest"],
        "seed": 0,
        "runtime": {"max_generation_tokens": 1000, "episode_timeout_s": 300},
    }
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config=resolved,
        resource_limits=resolved["runtime"],
    )
    ctx = runner._prepare()
    broker = ctx["broker"]
    broker.runtime_budget.consume_usage({"prompt_tokens": 5, "completion_tokens": 50, "total_tokens": 55})
    broker.get_budget()
    broker.get_budget()
    broker.get_budget()
    broker.persist_progress()
    saved = json.loads((ctx["run_dir"] / "run_progress.json").read_text())
    assert saved["runtime_budget"]["completion_tokens"] == 50
    assert saved["runtime_budget"]["tool_calls"] >= 3
    changed = dict(resolved)
    changed["model"] = {"model": "other-model", "base_url": "http://127.0.0.1:9/v1"}
    with pytest.raises(Exception, match="cannot change"):
        EpisodeRunner(
            public_dir=Path(synthetic_episode["public_dir"]),
            private_manifest=Path(synthetic_episode["private_manifest"]),
            run_root=tmp_path / "runs",
            mode="local_trusted_debug",
            agent="scripted_mock",
            resolved_config=changed,
            resume_run_id=ctx["run_id"],
        )._prepare()
    ctx2 = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
        resolved_config=resolved,
        resource_limits=resolved["runtime"],
        resume_run_id=ctx["run_id"],
    )._prepare()
    assert ctx2["broker"].runtime_budget.completion_tokens == 50
    assert ctx2["broker"].runtime_budget.tool_calls >= 3
    assert ctx2["broker"].runtime_budget.max_generation_tokens == 1000


def test_c05_debug_synthetic_is_not_official_eligible(synthetic_episode, tmp_path):
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        resolved_config={"mode": "local_trusted_debug", "allow_unqualified": True, "seed": 0, "runtime": {}},
    )
    ctx = runner._prepare()
    assert ctx["manifest"]["isolation_qualified"] is False
    assert ctx["manifest"]["official_eligible"] is False
    assert ctx["manifest"]["synthetic"] is True


def test_c04_output_limit_kills_process_before_it_finishes():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys,time; sys.stdout.write('x'*4096); sys.stdout.flush(); time.sleep(2); sys.stdout.write('alive')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    _out, _err, overflow = _bounded_communicate(proc, timeout_s=5, max_bytes=1024)
    elapsed = time.monotonic() - started
    assert overflow == "output_limit"
    assert elapsed < 1.5
    assert proc.poll() is not None


@pytest.mark.skip(reason="Docker isolation boundary tests are unverified on this host; copying a digest is not acceptance")
def test_c05_docker_boundary_unverified():
    raise AssertionError("should be skipped")


@pytest.mark.skip(reason="live local/API model episode eval is unverified")
def test_c05_live_model_unverified():
    raise AssertionError("should be skipped")
