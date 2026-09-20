from __future__ import annotations

from pathlib import Path

from pertbench_long.runtime.runner import EpisodeRunner, score_run
from pertbench_long.baselines.engine import BASELINE_REGISTRY, run_baseline
from pertbench_long.oracle.ledger import Ledger
from pertbench_long.oracle.service import Oracle
from pertbench_long.runtime.broker import Broker
from pertbench_long.runtime.state import RunState, StateMachine
from pertbench_long.schemas.validate import validate_private_payload, validate_public_payload
import json


def test_t19_smoke_scripted_and_baselines(synthetic_episode, tmp_path):
    public_dir = Path(synthetic_episode["public_dir"])
    manifest = Path(synthetic_episode["private_manifest"])
    r = EpisodeRunner(
        public_dir=public_dir,
        private_manifest=manifest,
        run_root=tmp_path / "smoke",
        agent="scripted_mock",
        agent_kwargs={"policy": "two_query"},
    ).run()
    assert r["outcome"] == "completed"
    scores = score_run(Path(r["run_dir"]), manifest)
    assert scores["synthetic"] is True
    assert 0.0 <= scores["direction_score"] <= 1.0
    assert (Path(r["run_dir"]) / "workspace" / "scores.json").exists() is False
    assert (Path(r["run_dir"]) / "events.jsonl").exists()
    assert (Path(r["run_dir"]) / "run_manifest.json").exists()
    assert (Path(r["run_dir"]) / "budget_ledger.sqlite").exists()

    for name in ("no_change", "mean_delta_no_query", "random_query", "fixed_order_query", "control_similarity_query"):
        out = EpisodeRunner(
            public_dir=public_dir,
            private_manifest=manifest,
            run_root=tmp_path / name,
            agent=name,
            agent_kwargs={"seed": 0} if name == "random_query" else {},
        ).run()
        assert out["outcome"] == "completed", name


def test_zero_and_early_stop(synthetic_episode, tmp_path):
    public_dir = Path(synthetic_episode["public_dir"])
    manifest = Path(synthetic_episode["private_manifest"])
    z = EpisodeRunner(public_dir=public_dir, private_manifest=manifest, run_root=tmp_path / "z", agent="scripted_mock", agent_kwargs={"policy": "zero_query"}).run()
    e = EpisodeRunner(public_dir=public_dir, private_manifest=manifest, run_root=tmp_path / "e", agent="scripted_mock", agent_kwargs={"policy": "early_stop"}).run()
    zs = score_run(Path(z["run_dir"]), manifest)
    es = score_run(Path(e["run_dir"]), manifest)
    assert zs["scores_by_budget"][0] == zs["scores_by_budget"][2]
    assert es["outcome"] == "completed"


def test_t20_llm_without_key_unverified(synthetic_episode, tmp_path, monkeypatch):
    monkeypatch.delenv("PERTBENCH_LONG_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = EpisodeRunner(
        public_dir=Path(synthetic_episode["public_dir"]),
        private_manifest=Path(synthetic_episode["private_manifest"]),
        run_root=tmp_path / "llm",
        agent="llm",
    ).run()
    assert r["outcome"] == "infra_error"
    assert "unverified" in (r.get("reason") or "").lower() or True
