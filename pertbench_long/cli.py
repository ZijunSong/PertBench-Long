"""CLI for PertBench-Long. Does not replace pertdiffbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from pertbench_long import SCHEMA_VERSION, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pertbench-long",
        description="Retrospective experimental-replay benchmark for scientific agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__} schema {SCHEMA_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List supported protocols, agents, and baselines")

    inspect = sub.add_parser("inspect-data", help="Audit local candidate datasets")
    inspect.add_argument("--config", default="configs/pertbench_long/pbmc_data.yaml")
    inspect.add_argument("--output", default="docs/pertbench_long/data_inventory.json")

    build = sub.add_parser("build-episodes", help="Build public/private episode packs (trusted builder only)")
    build.add_argument("--config", default="configs/pertbench_long/pbmc_pilot.yaml")
    build.add_argument("--fixture", choices=["synthetic", "pbmc"], default=None)
    build.add_argument("--output", default="runs/episodes")

    validate = sub.add_parser("validate", help="Validate a private manifest without creating directories")
    validate.add_argument("--manifest", required=True)

    smoke = sub.add_parser("smoke", help="CPU synthetic engineering smoke")
    smoke.add_argument("--fixture", default="synthetic")
    smoke.add_argument("--agent", default="scripted_mock")
    smoke.add_argument("--mode", default="local_trusted_debug", choices=["local_trusted_debug", "isolated_eval"])
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--policy", default="two_query")

    run = sub.add_parser("run", help="Run an agent or baseline on an episode")
    run.add_argument("--config", default=None)
    run.add_argument("--mode", default="local_trusted_debug", choices=["local_trusted_debug", "isolated_eval"])
    run.add_argument("--public-dir")
    run.add_argument("--private-manifest")
    run.add_argument("--agent")
    run.add_argument("--output", default="runs")

    score = sub.add_parser("score", help="Private scoring; not executed inside the agent workspace")
    score.add_argument("--run", required=True)
    score.add_argument("--private-manifest", required=True)

    summarize = sub.add_parser("summarize", help="Aggregate run folders")
    summarize.add_argument("--runs", required=True)
    summarize.add_argument("--output", required=True)

    replay = sub.add_parser("replay", help="Replay visible trace from events.jsonl")
    replay.add_argument("--run", required=True)
    return parser


def _cmd_list() -> int:
    from pertbench_long.baselines.engine import BASELINE_REGISTRY
    from pertbench_long.schemas.types import ALLOWED_PROTOCOLS

    print("protocols:")
    for item in sorted(ALLOWED_PROTOCOLS):
        print(f"  - {item}")
    print("agents:")
    for item in ("scripted_mock", "llm_adapter"):
        print(f"  - {item}")
    print("baselines:")
    for item in BASELINE_REGISTRY:
        print(f"  - {item}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text) if text.strip().startswith("{") else {}


def _cmd_inspect(args: argparse.Namespace) -> int:
    from pertbench_long.data.inventory import build_inventory, write_inventory
    from pertbench_long.extensions.moa import audit_moa_root
    from pertbench_long.extensions.temporal import audit_temporal_root

    cfg = _load_yaml(Path(args.config)) if Path(args.config).exists() else {}
    roots = cfg.get("roots")
    inventory = build_inventory(roots)
    inventory["extensions"] = {
        "moa": audit_moa_root(Path(cfg.get("moa_root", "/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseenMOA"))),
        "temporal": audit_temporal_root(Path(cfg.get("temporal_root", "/data/ppnm/data/PertDiffBench/data_ori/fig4"))),
    }
    # Open one PBMC CSV if present to classify matrix kind.
    pbmc = Path(cfg.get("pbmc_csv_dir", "/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus"))
    sample = pbmc / "task1_train_B_exp.csv"
    if sample.exists():
        from pertbench_long.data.adapter import infer_matrix_kind, load_table

        matrix, obs_ids, genes = load_table(sample)
        inventory["pbmc_sample"] = {
            "path": str(sample),
            "n_rows_sample_file": matrix.shape[0],
            "n_genes": len(genes),
            "matrix_kind": infer_matrix_kind(matrix),
            "obs_id_example": obs_ids[:3],
            "donor_column": None,
            "columns_observed": ["index-encoded perturbation suffix", "cell type from filename"],
        }
    out = write_inventory(Path(args.output), inventory)
    print(f"wrote {out}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from pertbench_long.episodes.builder import build_episode_from_store, build_synthetic_episode

    dest = Path(args.output)
    fixture = args.fixture
    cfg = _load_yaml(Path(args.config)) if Path(args.config).exists() else {}
    if fixture is None:
        fixture = cfg.get("fixture", "synthetic")
    if fixture == "synthetic":
        result = build_synthetic_episode(dest / "synthetic_pilot_001")
        print(json.dumps(result, indent=2, default=str))
        return 0
    from pertbench_long.data.adapter import import_tables

    csv_dir = Path(cfg.get("pbmc_csv_dir", "/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus"))
    files = sorted(csv_dir.glob("task1_*_exp.csv"))
    if not files:
        print("required PBMC CSVs missing; synthetic episode was not substituted for a real benchmark", file=sys.stderr)
        print(f"required-inputs: {csv_dir}/task1_{{train,valid}}_{{celltype}}_exp.csv", file=sys.stderr)
        return 2
    store = import_tables(files, study=cfg.get("study", "kang_pbmc_ifn_public"), species="human")
    result = build_episode_from_store(
        store,
        dest=dest / "pbmc_pilot_001",
        episode_id="pbmc_pilot_001",
        o_types=cfg.get("o_types", ["CD4T", "CD8T"]),
        q_types=cfg.get("q_types", ["CD14+Mono", "Dendritic", "FCGR3A+Mono"]),
        t_types=cfg.get("t_types", ["B", "NK"]),
        synthetic=False,
        n_panel=int(cfg.get("n_panel", 64)),
        study_name=cfg.get("study", "kang_pbmc_ifn_public"),
        data_release_id=cfg.get("data_release_id", "local_kang_csv"),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from pertbench_long.schemas.validate import validate_private_payload

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    spec = validate_private_payload(payload)
    print(f"ok episode_id={spec.episode_id} genes={len(spec.gene_ids)}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from pertbench_long.episodes.builder import build_synthetic_episode
    from pertbench_long.runtime.runner import EpisodeRunner, score_run

    dest = Path(args.output)
    built = build_synthetic_episode(dest / "episode")
    runner = EpisodeRunner(
        public_dir=Path(built["public_dir"]),
        private_manifest=Path(built["private_manifest"]),
        run_root=dest / "runs",
        mode=args.mode,
        agent=args.agent,
        agent_kwargs={"policy": args.policy},
    )
    result = runner.run()
    if result.get("outcome") == "completed":
        scores = score_run(Path(result["run_dir"]), Path(built["private_manifest"]))
        print(json.dumps({"run": result, "scores": {"direction_score": scores["direction_score"], "AUBC": scores["AUBC"], "synthetic": True}}, indent=2))
        return 0
    print(json.dumps({"run": result, "isolation_qualified": False, "synthetic": True}, indent=2))
    return 1 if args.mode == "isolated_eval" else 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.runner import EpisodeRunner

    cfg = _load_yaml(Path(args.config)) if args.config and Path(args.config).exists() else {}
    if not (args.public_dir or cfg.get("public_dir")):
        raise SystemExit("--public-dir or --config with public_dir is required")
    public_dir = Path(args.public_dir or cfg["public_dir"])
    private_manifest = Path(args.private_manifest or cfg["private_manifest"])
    agent = args.agent or cfg.get("agent", "scripted_mock")
    runner = EpisodeRunner(
        public_dir=public_dir,
        private_manifest=private_manifest,
        run_root=Path(args.output),
        mode=args.mode,
        agent=agent,
        agent_kwargs=dict(cfg.get("agent_kwargs") or {}),
        resource_limits=dict(cfg.get("resource_limits") or {}),
    )
    print(json.dumps(runner.run(), indent=2, default=str))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.runner import score_run

    payload = score_run(Path(args.run), Path(args.private_manifest))
    print(json.dumps({k: payload[k] for k in ("outcome", "direction_score", "AUBC", "S0", "S1", "S2", "delta_S", "synthetic") if k in payload}, indent=2))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from pertbench_long.evaluation.outcomes import summarize_runs

    root = Path(args.runs)
    runs = []
    for term in root.rglob("termination.json"):
        payload = json.loads(term.read_text(encoding="utf-8"))
        scores_path = term.parent / "scores.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        runs.append(
            {
                "run_id": term.parent.name,
                "outcome": payload.get("outcome"),
                "direction_score": scores.get("direction_score"),
                "reason": payload.get("reason"),
            }
        )
    summary = summarize_runs(runs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.replay import replay_visible_trace

    print(json.dumps(replay_visible_trace(Path(args.run)), indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "list": lambda: _cmd_list(),
        "inspect-data": lambda: _cmd_inspect(args),
        "build-episodes": lambda: _cmd_build(args),
        "validate": lambda: _cmd_validate(args),
        "smoke": lambda: _cmd_smoke(args),
        "run": lambda: _cmd_run(args),
        "score": lambda: _cmd_score(args),
        "summarize": lambda: _cmd_summarize(args),
        "replay": lambda: _cmd_replay(args),
    }
    return dispatch[args.command]()


if __name__ == "__main__":
    sys.exit(main())
