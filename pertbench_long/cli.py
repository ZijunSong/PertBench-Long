"""CLI for PertBench-Long. Does not replace pertdiffbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from pertbench_long import SCHEMA_VERSION, __version__
from pertbench_long.errors import (
    EXIT_AGENT,
    EXIT_CONFIG,
    EXIT_INFRA,
    EXIT_INTEGRITY,
    EXIT_OK,
    ConfigError,
    IntegrityError,
    PertBenchLongError,
)

DOCUMENTED_RUN_DEFAULTS = {
    "agent": "scripted_mock",
    "output": "runs",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pertbench-long",
        description="Retrospective experimental-replay benchmark for scientific agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__} schema {SCHEMA_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List implemented protocols, agents, and baselines")

    inspect = sub.add_parser("inspect-data", help="Audit local candidate datasets")
    inspect.add_argument("--config", default=None)
    inspect.add_argument("--output", default=None)

    build = sub.add_parser("build-episodes", help="Build public/private episode packs (trusted builder only)")
    build.add_argument("--config", default=None)
    build.add_argument("--fixture", choices=["synthetic", "pbmc"], default=None)
    build.add_argument("--output", default=None)

    validate = sub.add_parser("validate", help="Validate a private manifest without creating directories")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--check-artifacts", action="store_true")

    smoke = sub.add_parser("smoke", help="CPU synthetic engineering smoke")
    smoke.add_argument("--fixture", default="synthetic")
    smoke.add_argument("--agent", default="scripted_mock")
    smoke.add_argument("--mode", default="local_trusted_debug", choices=["local_trusted_debug", "isolated_eval"])
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--policy", default="two_query")

    run = sub.add_parser("run", help="Run an agent or baseline on an episode")
    run.add_argument("--config", default=None)
    run.add_argument("--mode", default=None, choices=["local_trusted_debug", "isolated_eval"])
    run.add_argument("--public-dir", default=None)
    run.add_argument("--private-manifest", default=None)
    run.add_argument("--agent", default=None)
    run.add_argument("--output", default=None)
    run.add_argument("--seed", default=None, type=int)
    run.add_argument("--allow-unqualified", action="store_true")

    score = sub.add_parser("score", help="Private scoring; not executed inside the agent workspace")
    score.add_argument("--run", required=True)
    score.add_argument("--private-manifest", required=True)

    summarize = sub.add_parser("summarize", help="Aggregate run folders")
    summarize.add_argument("--runs", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--group-by", default="agent,model,policy,budget,isolation_qualified,scoring_track,synthetic")

    replay = sub.add_parser("replay", help="Summarize visible events.jsonl; does not restore artifacts")
    replay.add_argument("--run", required=True)

    trace = sub.add_parser("trace-summary", help="Alias of replay: event counts only")
    trace.add_argument("--run", required=True)

    resume = sub.add_parser("resume", help="Resume an interrupted run with the same run ID")
    resume.add_argument("--run", required=True)
    resume.add_argument("--config", default=None)

    doctor = sub.add_parser("doctor", help="Check local install without downloading models")
    doctor.add_argument("--config", default=None)

    doctor_model = sub.add_parser("doctor-model", help="Minimal text and tool-round probe; does not send episode data")
    doctor_model.add_argument("--config", required=True)

    suite = sub.add_parser("suite", help="Run a group of models/seeds on a fixed episode release")
    suite.add_argument("--config", required=True)
    return parser


def _cmd_list() -> int:
    from pertbench_long.baselines.engine import BASELINE_REGISTRY
    from pertbench_long.runtime.runner import AGENT_REGISTRY
    from pertbench_long.schemas.types import IMPLEMENTED_PROTOCOLS, UNSUPPORTED_PROTOCOLS

    print("protocols (implemented):")
    for item in sorted(IMPLEMENTED_PROTOCOLS):
        print(f"  - {item}")
    print("protocols (named, unsupported):")
    for item in sorted(UNSUPPORTED_PROTOCOLS):
        print(f"  - {item} [unavailable]")
    print("agents:")
    for item, status in AGENT_REGISTRY.items():
        print(f"  - {item} [{status}]")
    print("baselines:")
    for item in BASELINE_REGISTRY:
        print(f"  - {item}")
    return EXIT_OK


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file does not exist: {path}")
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
    except Exception as exc:
        raise ConfigError(f"failed to parse YAML {path}: {exc}") from exc
    if payload is None:
        raise ConfigError(f"config {path} is empty")
    if not isinstance(payload, dict):
        raise ConfigError(f"config {path} must be a mapping")
    return payload


def _pick(cli_value: Any, yaml_value: Any, default: Any = None, *, name: str, required: bool = False):
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    if required:
        raise ConfigError(f"{name} is required (explicit CLI or YAML)")
    return default


def _cmd_inspect(args: argparse.Namespace) -> int:
    from pertbench_long.data.adapter import infer_matrix_kind, load_table, numeric_kind_contradiction
    from pertbench_long.data.inventory import build_inventory, write_inventory
    from pertbench_long.extensions.moa import audit_moa_root
    from pertbench_long.extensions.temporal import audit_temporal_root

    if not args.config:
        raise ConfigError("--config is required")
    cfg = _load_yaml(Path(args.config))
    roots = cfg.get("roots")
    inventory = build_inventory(roots)
    if cfg.get("moa_root"):
        inventory.setdefault("extensions", {})["moa"] = audit_moa_root(Path(cfg["moa_root"]))
    if cfg.get("temporal_root"):
        inventory.setdefault("extensions", {})["temporal"] = audit_temporal_root(Path(cfg["temporal_root"]))
    sample = Path(cfg["pbmc_csv_dir"]) / "task1_train_B_exp.csv" if cfg.get("pbmc_csv_dir") else None
    declared = cfg.get("matrix_kind")
    if sample and sample.exists():
        matrix, obs_ids, genes = load_table(sample)
        inventory["pbmc_sample"] = {
            "path": str(sample),
            "n_rows_sample_file": matrix.shape[0],
            "n_genes": len(genes),
            "declared_matrix_kind": declared,
            "numeric_range_guess_diagnostic_only": infer_matrix_kind(matrix),
            "contradictions": numeric_kind_contradiction(matrix, declared) if declared else ["matrix_kind_undeclared"],
            "obs_id_example": obs_ids[:3],
            "note": "numeric range never proves log1p and is not used for official transforms",
        }
    elif cfg.get("pbmc_csv_dir"):
        print(f"required-inputs missing: {cfg['pbmc_csv_dir']}", file=sys.stderr)
        return EXIT_CONFIG
    out = write_inventory(Path(args.output or "docs/pertbench_long/data_inventory.json"), inventory)
    print(f"wrote {out}")
    return EXIT_OK


def _cmd_build(args: argparse.Namespace) -> int:
    from pertbench_long.episodes.builder import build_episode_from_store, build_synthetic_episode

    cfg = _load_yaml(Path(args.config)) if args.config else {}
    dest = Path(_pick(args.output, cfg.get("output"), "runs/episodes", name="output"))
    fixture = _pick(args.fixture, cfg.get("fixture"), name="fixture", required=True)
    if fixture == "synthetic":
        result = build_synthetic_episode(
            dest / "synthetic_pilot_001",
            experimental_budget=int(cfg.get("experimental_budget", 2)) if "experimental_budget" in cfg else 2,
        )
        print(json.dumps(result, indent=2, default=str))
        return EXIT_OK
    if fixture != "pbmc":
        raise ConfigError("fixture must be synthetic or pbmc; they are not interchangeable")
    from pertbench_long.data.adapter import import_tables

    if not cfg.get("pbmc_csv_dir"):
        raise ConfigError("pbmc_csv_dir is required in the config")
    csv_dir = Path(cfg["pbmc_csv_dir"])
    files = sorted(csv_dir.glob("task1_*_exp.csv"))
    if not files:
        print("required PBMC CSVs missing; synthetic episode was not substituted for a real benchmark", file=sys.stderr)
        print(f"required-inputs: {csv_dir}/task1_{{train,valid}}_{{celltype}}_exp.csv", file=sys.stderr)
        return EXIT_CONFIG
    declared = cfg.get("matrix_kind")
    store = import_tables(
        files,
        study=cfg.get("study", "kang_pbmc_ifn_public"),
        species="human",
        declared_matrix_kind=declared,
        conflict_policy=cfg.get("conflict_policy", "fail_closed"),
    )
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
        experimental_budget=int(cfg.get("experimental_budget", 2)),
        protocol=cfg.get("protocol", "within_study_celltype_ood_v1"),
        label_profile=cfg.get("label_profile", "effect_proxy_v1"),
    )
    print(json.dumps(result, indent=2, default=str))
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.release import check_release
    from pertbench_long.schemas.validate import validate_private_payload

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    spec = validate_private_payload(payload)
    if args.check_artifacts:
        root = Path(args.manifest).parent
        public_dir = root.parent / "public"
        check_release(public_dir, Path(args.manifest))
    print(f"ok episode_id={spec.episode_id} genes={len(spec.gene_ids)} track={(spec.scoring_config or {}).get('scoring_track')}")
    return EXIT_OK


def _exit_for_outcome(outcome: str | None) -> int:
    if outcome == "completed":
        return EXIT_OK
    if outcome == "integrity_failure":
        return EXIT_INTEGRITY
    if outcome in {"infra_error", "invalid_episode"}:
        return EXIT_INFRA
    if outcome in {"invalid_submission", "agent_timeout", "agent_tool_failure"}:
        return EXIT_AGENT
    return EXIT_INFRA


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
        resolved_config={"mode": args.mode, "agent": args.agent},
    )
    result = runner.run()
    if result.get("outcome") == "completed":
        scores = score_run(Path(result["run_dir"]), Path(built["private_manifest"]))
        print(json.dumps({"run": result, "scores": {"direction_score": scores["direction_score"], "AUBC": scores["AUBC"], "synthetic": True}}, indent=2))
        return EXIT_OK
    print(json.dumps({"run": result, "isolation_qualified": False, "synthetic": True}, indent=2))
    return _exit_for_outcome(result.get("outcome"))


def _cmd_run(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.config import agent_kwargs_from_resolved, resolve_run_config
    from pertbench_long.runtime.runner import AGENT_REGISTRY, EpisodeRunner

    cfg = _load_yaml(Path(args.config)) if args.config else {}
    unknown = set(cfg) - {
        "config_version",
        "mode",
        "agent",
        "public_dir",
        "private_manifest",
        "output",
        "model",
        "agent_kwargs",
        "resource_limits",
        "runtime",
        "evaluation",
        "seed",
        "note",
        "allow_unqualified",
    }
    if unknown:
        raise ConfigError(f"unknown config fields: {sorted(unknown)}")
    resolved = resolve_run_config(
        cli={
            "mode": args.mode,
            "public_dir": args.public_dir,
            "private_manifest": args.private_manifest,
            "agent": args.agent,
            "output": args.output,
            "seed": args.seed,
            "allow_unqualified": True if args.allow_unqualified else None,
        },
        yaml_cfg=cfg,
        documented=DOCUMENTED_RUN_DEFAULTS,
    )
    agent = resolved["agent"]
    if agent not in AGENT_REGISTRY:
        raise ConfigError(f"unknown agent {agent}")
    agent_kwargs = agent_kwargs_from_resolved(resolved, extra=dict(cfg.get("agent_kwargs") or {}))
    runner = EpisodeRunner(
        public_dir=Path(resolved["public_dir"]),
        private_manifest=Path(resolved["private_manifest"]),
        run_root=Path(resolved["output"]),
        mode=str(resolved["mode"]),
        agent=agent,
        agent_kwargs=agent_kwargs,
        resource_limits=dict(cfg.get("resource_limits") or resolved.get("runtime") or {}),
        resolved_config=resolved,
    )
    result = runner.run()
    print(json.dumps(result, indent=2, default=str))
    return _exit_for_outcome(result.get("outcome"))


def _cmd_score(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.runner import score_run

    run_path = Path(args.run)
    if not run_path.exists():
        raise ConfigError(f"run directory does not exist: {run_path}")
    if run_path.name.endswith("*") or "*" in str(run_path):
        raise ConfigError("--run must be a single run directory, not a glob")
    payload = score_run(run_path, Path(args.private_manifest))
    print(json.dumps({k: payload[k] for k in ("outcome", "score_status", "direction_score", "AUBC", "S0", "S1", "S2", "delta_S", "synthetic") if k in payload}, indent=2))
    if payload.get("outcome") == "integrity_failure":
        return EXIT_INTEGRITY
    return EXIT_OK if payload.get("score_status") == "scored" else EXIT_INFRA


def _cmd_summarize(args: argparse.Namespace) -> int:
    from pertbench_long.evaluation.outcomes import summarize_runs

    root = Path(args.runs)
    runs = []
    for term in root.rglob("termination.json"):
        payload = json.loads(term.read_text(encoding="utf-8"))
        scores_path = term.parent / "scores.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        manifest_path = term.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        runs.append(
            {
                "run_id": term.parent.name,
                "run_dir": str(term.parent),
                "episode_id": manifest.get("episode_id"),
                "release": (manifest.get("binding") or {}).get("public_spec_hash"),
                "outcome": scores.get("outcome") or payload.get("outcome") or "not_scored",
                "execution_outcome": payload.get("outcome"),
                "scored_outcome": scores.get("outcome"),
                "direction_score": scores.get("direction_score"),
                "AUBC": scores.get("AUBC"),
                "score_status": scores.get("score_status") or payload.get("score_status") or ("not_scored" if not scores else "scored"),
                "reason": scores.get("reason") or payload.get("reason"),
                "agent": manifest.get("agent"),
                "model": ((manifest.get("model") or {}) if isinstance(manifest.get("model"), dict) else {}).get("model") or manifest.get("model"),
                "policy": manifest.get("agent"),
                "seed": (manifest.get("resolved_config") or {}).get("seed") if isinstance(manifest.get("resolved_config"), dict) else None,
                "budget": manifest.get("experimental_budget") or ((manifest.get("resolved_config") or {}).get("evaluation") or {}).get("experimental_budget_cap") if isinstance(manifest.get("resolved_config"), dict) else None,
                "isolation_qualified": manifest.get("isolation_qualified"),
                "scoring_track": (manifest.get("binding") or {}).get("scoring_track"),
                "synthetic": manifest.get("synthetic"),
                "action_format": ((manifest.get("model") or {}) if isinstance(manifest.get("model"), dict) else {}).get("action_format"),
            }
        )
    fields = [p.strip() for p in str(args.group_by).split(",") if p.strip()]
    summary = summarize_runs(runs, group_by=fields)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def _cmd_replay(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.replay import replay_visible_trace

    print(json.dumps(replay_visible_trace(Path(args.run)), indent=2))
    return EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    from pertbench_long.runtime.config import agent_kwargs_from_resolved
    from pertbench_long.runtime.runner import EpisodeRunner

    run_dir = Path(args.run)
    if not (run_dir / "run_manifest.json").exists():
        raise ConfigError("resume requires an existing run_manifest.json")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    orig = dict(manifest.get("resolved_config") or {})
    overlay = _load_yaml(Path(args.config)) if args.config else {}
    resolved = {**orig, **{k: v for k, v in overlay.items() if v is not None}}
    if orig.get("mode") and overlay.get("mode") and orig.get("mode") != overlay.get("mode"):
        raise IntegrityError("resume mode does not match the original run")
    if orig.get("agent") and overlay.get("agent") and orig.get("agent") != overlay.get("agent"):
        raise IntegrityError("resume agent does not match the original run")
    public_dir = Path(resolved.get("public_dir") or orig.get("public_dir"))
    private_manifest = Path(resolved.get("private_manifest") or orig.get("private_manifest"))
    agent_kwargs = agent_kwargs_from_resolved(resolved, extra=dict(overlay.get("agent_kwargs") or orig.get("agent_kwargs") or {}))
    runner = EpisodeRunner(
        public_dir=public_dir,
        private_manifest=private_manifest,
        run_root=run_dir.parent,
        mode=str(resolved.get("mode") or manifest.get("mode")),
        agent=str(resolved.get("agent") or manifest.get("agent")),
        agent_kwargs=agent_kwargs,
        resource_limits=dict(resolved.get("runtime") or orig.get("runtime") or {}),
        resolved_config=resolved,
        resume_run_id=run_dir.name,
    )
    result = runner.run()
    print(json.dumps(result, indent=2, default=str))
    return _exit_for_outcome(result.get("outcome"))


def _cmd_doctor(args: argparse.Namespace) -> int:
    info = {"python": sys.version.split()[0], "package": __version__, "schema": SCHEMA_VERSION}
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401

        info["numpy_pandas"] = "ok"
    except Exception as exc:
        info["numpy_pandas"] = str(exc)
        print(json.dumps(info, indent=2))
        return EXIT_INFRA
    extras = {}
    for name in ("anndata", "pyarrow", "yaml"):
        try:
            __import__(name if name != "yaml" else "yaml")
            extras[name] = "ok"
        except Exception:
            extras[name] = "missing"
    info["extras"] = extras
    print(json.dumps(info, indent=2))
    return EXIT_OK


def _cmd_doctor_model(args: argparse.Namespace) -> int:
    from pertbench_long.agents.client import ModelClient

    cfg = _load_yaml(Path(args.config))
    client = ModelClient(cfg)
    text = client.generate([{"role": "user", "content": "Reply with the single word pong."}], tool_schemas=None)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a token",
                "parameters": {"type": "object", "properties": {"token": {"type": "string"}}, "required": ["token"]},
            },
        }
    ]
    user = {"role": "user", "content": "Call echo with token=ping. Do not send episode data."}
    tool_round = None
    follow = None
    tool_ok = False
    protocol_ok = False
    try:
        tool_round = client.generate([user], tool_schemas=tools)
        echo_calls = [a for a in tool_round.actions if a.get("name") == "echo"]
        tool_ok = bool(echo_calls)
        if tool_ok and client.capabilities.action_format == "native_tools":
            assistant = dict(tool_round.assistant_message)
            token = None
            args = echo_calls[0].get("arguments") or {}
            if isinstance(args, dict):
                token = args.get("token")
            messages = [user, assistant if assistant.get("role") else {"role": "assistant", **assistant}]
            for action in tool_round.actions:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.get("id") or action.get("name") or "echo",
                        "content": json.dumps({"ok": True, "echo": token or "ping"}),
                    }
                )
            follow = client.generate(messages + [{"role": "user", "content": "Acknowledge the echo result in one word."}], tool_schemas=tools)
            protocol_ok = follow.assistant_message.get("role") == "assistant" or bool(follow.assistant_message)
        elif tool_ok:
            protocol_ok = True
        elif client.capabilities.action_format == "json_action":
            from pertbench_long.agents.client import json_action_instruction

            instructed = client.generate(
                [{"role": "system", "content": json_action_instruction(tools)}, user],
                tool_schemas=None,
            )
            tool_ok = any(a.get("name") == "echo" for a in instructed.actions)
            protocol_ok = tool_ok
            tool_round = instructed
    except Exception as exc:
        tool_round = {"error": type(exc).__name__, "message": str(exc)}
    payload = {
        "text_probe": {"finish_reason": text.finish_reason, "content_present": bool(text.assistant_message.get("content"))},
        "tool_probe": {"ok": tool_ok, "actions": getattr(tool_round, "actions", None) or tool_round},
        "tool_followup": {"ok": protocol_ok, "finish_reason": getattr(follow, "finish_reason", None)},
        "endpoint_reachable": True,
        "tool_protocol_usable": bool(tool_ok and protocol_ok),
        "note": "endpoint reachable is not the same as tool protocol usable; live eval remains unverified until a full episode succeeds",
        "resolved": client.resolved_profile,
    }
    print(json.dumps(payload, indent=2, default=str))
    return EXIT_OK if tool_ok and protocol_ok else EXIT_INFRA


def _cmd_suite(args: argparse.Namespace) -> int:
    from pertbench_long.evaluation.suite import run_suite

    summary = run_suite(Path(args.config))
    print(json.dumps(summary, indent=2, default=str))
    return EXIT_OK if not summary.get("failed") else EXIT_AGENT


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
        "trace-summary": lambda: _cmd_replay(args),
        "resume": lambda: _cmd_resume(args),
        "doctor": lambda: _cmd_doctor(args),
        "doctor-model": lambda: _cmd_doctor_model(args),
        "suite": lambda: _cmd_suite(args),
    }
    try:
        return dispatch[args.command]()
    except ConfigError as exc:
        print(f"config error: {exc.message}", file=sys.stderr)
        return EXIT_CONFIG
    except IntegrityError as exc:
        print(f"integrity error: {exc.message}", file=sys.stderr)
        return EXIT_INTEGRITY
    except PertBenchLongError as exc:
        print(f"error {exc.error_code}: {exc.message}", file=sys.stderr)
        return EXIT_INFRA


if __name__ == "__main__":
    sys.exit(main())
