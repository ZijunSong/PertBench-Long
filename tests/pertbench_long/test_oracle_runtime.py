from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from pertbench_long.errors import BudgetExceeded, IdempotencyConflict, InvalidState, PathGuardError, UnavailableExperiment
from pertbench_long.oracle.ledger import Ledger
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.runner import EpisodeRunner, score_run
from pertbench_long.runtime.sandbox import PathGuard, require_isolated_backend
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload
from pertbench_long.predictors.base import assert_checkpoint_provenance, assert_no_test_paths
from pertbench_long.errors import PredictorRejected


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
    )
    return oracle, broker


def test_t05_cannot_buy_target_or_unknown(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    from pertbench_long.baselines.engine import estimate, _write_pred

    effects, probs = estimate(broker)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    with pytest.raises(UnavailableExperiment):
        broker.request_experiment("t_01", "req_bad")
    with pytest.raises(UnavailableExperiment):
        broker.request_experiment("not_an_id", "req_bad2")
    snap = oracle.get_budget(broker.run_id)
    assert snap["charged_credits"] == 0


def test_t06_idempotent_retry_and_repurchase(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    from pertbench_long.baselines.engine import estimate, _write_pred

    effects, probs = estimate(broker)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    a = broker.request_experiment("q_01", "req_0001")
    b = broker.request_experiment("q_01", "req_0001")
    assert a["charged_credits"] == 1
    assert b["charged_credits"] == 0
    c2 = broker.request_experiment("q_01", "req_0002")
    assert c2["charged_credits"] == 0
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 1
    with pytest.raises(IdempotencyConflict):
        # reuse request with different payload
        oracle.request_experiment(run_id=broker.run_id, experiment_id="q_02", request_id="req_0001", evidence_dir=broker.workspace / "evidence")


def test_t07_last_credit_race_and_submit_blocks(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    from pertbench_long.baselines.engine import estimate, _write_pred

    effects, probs = estimate(broker)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    broker.request_experiment("q_01", "req_0001")
    effects, probs = estimate(broker)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    results = []

    def buy(req, exp):
        try:
            results.append(oracle.request_experiment(run_id=broker.run_id, experiment_id=exp, request_id=req, evidence_dir=broker.workspace / "evidence"))
        except Exception as exc:
            results.append(exc)

    t1 = threading.Thread(target=buy, args=("req_a", "q_02"))
    t2 = threading.Thread(target=buy, args=("req_b", "q_03"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    successes = [r for r in results if isinstance(r, dict) and r.get("charged_credits") == 1]
    failures = [r for r in results if isinstance(r, BudgetExceeded) or (isinstance(r, dict) and r.get("charged_credits") == 0 and not isinstance(r, dict))]
    assert oracle.get_budget(broker.run_id)["remaining_credits"] >= 0
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 2
    assert len(successes) == 1
    broker.submit(p, c)
    with pytest.raises(InvalidState):
        broker.request_experiment("q_03", "req_after")


def test_t08_crash_recovery_same_artifact(synthetic_episode, tmp_path):
    oracle, broker = _oracle(synthetic_episode, tmp_path)
    from pertbench_long.baselines.engine import estimate, _write_pred

    effects, probs = estimate(broker)
    p, c = _write_pred(broker, effects, probs)
    broker.save_snapshot(p, c)
    first = broker.request_experiment("q_01", "req_0001")
    replay = oracle.request_experiment(
        run_id=broker.run_id, experiment_id="q_01", request_id="req_0001", evidence_dir=broker.workspace / "evidence"
    )
    assert replay["charged_credits"] == 0
    assert replay["artifact"]["sha256"] == first["artifact"]["sha256"]
    assert oracle.get_budget(broker.run_id)["charged_credits"] == 1


def test_t10_path_guard(synthetic_episode, tmp_path):
    _, broker = _oracle(synthetic_episode, tmp_path)
    with pytest.raises(PathGuardError):
        broker.read_artifact(str(Path(synthetic_episode["private_manifest"])))
    with pytest.raises(PathGuardError):
        broker.read_artifact("../private/manifest.json")
    guard = PathGuard([broker.workspace], denied=[Path(synthetic_episode["private_manifest"]).parent])
    with pytest.raises(PathGuardError):
        guard.check(Path(synthetic_episode["private_manifest"]))


def test_t09_isolated_mode_does_not_silently_pass():
    try:
        backend = require_isolated_backend("isolated_eval")
    except Exception as exc:
        assert "isolated" in str(exc).lower() or "Docker" in str(exc) or "isolation" in str(exc).lower()
        return
    assert backend == "docker"


def test_t15_checkpoint_with_t_rejected():
    with pytest.raises(PredictorRejected):
        assert_no_test_paths({"test_h5ad": "/tmp/test.h5ad"})
    with pytest.raises(PredictorRejected):
        assert_checkpoint_provenance(
            {
                "train_observation_hash": "abc",
                "split_id": "s",
                "includes_unpurchased_or_target": ["t_01"],
                "allowed_observation_hash": "abc",
            },
            "abc",
        )


def test_t16_trusted_snapshot_not_clobbered(synthetic_episode, tmp_path):
    runner = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "runs",
        mode="local_trusted_debug",
        agent="scripted_mock",
        agent_kwargs={"policy": "zero_query"},
    )
    result = runner.run()
    trusted = Path(json.loads((Path(result["run_dir"]) / "termination.json").read_text())["trusted_submission"]["path"])
    original = trusted.read_bytes()
    ws_pred = list((Path(result["run_dir"]) / "workspace").rglob("predictions.parquet"))[0]
    ws_pred.write_bytes(b"not-a-parquet")
    assert trusted.read_bytes() == original
    scores = score_run(Path(result["run_dir"]), Path(synthetic_episode["private_manifest"]))
    assert scores["outcome"] == "completed"


def test_t18_runs_do_not_share_workspace(synthetic_episode, tmp_path):
    r1 = EpisodeRunner(public_dir=Path(synthetic_episode["public_dir"]), private_manifest=Path(synthetic_episode["private_manifest"]), run_root=tmp_path / "runs", agent="scripted_mock", agent_kwargs={"policy": "zero_query"}).run()
    r2 = EpisodeRunner(public_dir=Path(synthetic_episode["public_dir"]), private_manifest=Path(synthetic_episode["private_manifest"]), run_root=tmp_path / "runs", agent="scripted_mock", agent_kwargs={"policy": "zero_query"}).run()
    assert Path(r1["run_dir"]).resolve() != Path(r2["run_dir"]).resolve()
    assert (Path(r1["run_dir"]) / "workspace").resolve() != (Path(r2["run_dir"]) / "workspace").resolve()
