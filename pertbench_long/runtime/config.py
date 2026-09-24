"""Shared run-config resolution. Explicit CLI > YAML > documented defaults. None is not 0."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from pertbench_long.errors import ConfigError

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PATH_FIELDS = ("public_dir", "private_manifest", "output")


def first_defined(*values: Any, default: Any = None) -> Any:
    """Return the first value that is not None. 0 and False are kept."""
    for value in values:
        if value is not None:
            return value
    return default


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def expand_config_string(value: str) -> str:
    """Expand ~ and ${VAR} in paths. Does not read or store API key values."""

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"unresolved environment variable ${{{name}}}")
        return os.environ[name]

    return os.path.expanduser(_ENV_VAR.sub(_replace, value))


def resolve_config_path(raw: str, *, config_path: Path | None) -> str:
    text = expand_config_string(str(raw))
    path = Path(text)
    if not path.is_absolute() and config_path is not None:
        path = (Path(config_path).resolve().parent / path).resolve()
    return str(path)


def resolve_run_config(
    *,
    cli: Mapping[str, Any] | None = None,
    yaml_cfg: Mapping[str, Any] | None = None,
    documented: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Build the config object that run/resume/suite must actually consume."""
    cli = _as_mapping(cli)
    yaml_cfg = _as_mapping(yaml_cfg)
    documented = _as_mapping(documented)
    runtime = {
        **_as_mapping(documented.get("runtime")),
        **_as_mapping(yaml_cfg.get("runtime")),
        **_as_mapping(cli.get("runtime")),
    }
    evaluation = {
        **_as_mapping(documented.get("evaluation")),
        **_as_mapping(yaml_cfg.get("evaluation")),
        **_as_mapping(cli.get("evaluation")),
    }
    model = yaml_cfg.get("model")
    if cli.get("model") is not None:
        model = cli.get("model")
    seed = first_defined(cli.get("seed"), yaml_cfg.get("seed"), evaluation.get("seed"), documented.get("seed"))
    resolved = {
        "mode": first_defined(cli.get("mode"), yaml_cfg.get("mode"), documented.get("mode")),
        "agent": first_defined(cli.get("agent"), yaml_cfg.get("agent"), documented.get("agent")),
        "public_dir": first_defined(cli.get("public_dir"), yaml_cfg.get("public_dir"), documented.get("public_dir")),
        "private_manifest": first_defined(cli.get("private_manifest"), yaml_cfg.get("private_manifest"), documented.get("private_manifest")),
        "output": first_defined(cli.get("output"), yaml_cfg.get("output"), documented.get("output")),
        "model": model,
        "runtime": runtime,
        "evaluation": evaluation,
        "seed": seed,
        "allow_unqualified": bool(first_defined(cli.get("allow_unqualified"), yaml_cfg.get("allow_unqualified"), evaluation.get("allow_unqualified"), False)),
        "priority": "explicit_cli > yaml > documented_defaults",
        "unsupported_noted": [],
    }
    if resolved["mode"] is None:
        raise ConfigError("mode is required (explicit CLI or YAML)")
    if resolved["public_dir"] is None or resolved["private_manifest"] is None:
        raise ConfigError("public_dir and private_manifest are required")
    for field in PATH_FIELDS:
        if resolved.get(field):
            resolved[field] = resolve_config_path(str(resolved[field]), config_path=config_path)
    report = (resolved.get("runtime") or {}).get("isolation_acceptance_report")
    if report:
        resolved["runtime"]["isolation_acceptance_report"] = resolve_config_path(str(report), config_path=config_path)
    return resolved


def run_identity(resolved: Mapping[str, Any], *, agent: str, mode: str) -> dict[str, Any]:
    runtime = dict(resolved.get("runtime") or {})
    model = resolved.get("model")
    return {
        "agent": agent,
        "mode": mode,
        "model": model,
        "seed": resolved.get("seed"),
        "runtime_caps": {
            "max_generation_tokens": runtime.get("max_generation_tokens"),
            "max_total_tokens": runtime.get("max_total_tokens"),
            "episode_timeout_s": runtime.get("episode_timeout_s"),
            "max_agent_steps": runtime.get("max_agent_steps"),
            "max_external_tool_calls": runtime.get("max_external_tool_calls"),
            "analysis_image": runtime.get("analysis_image"),
        },
    }


def agent_kwargs_from_resolved(resolved: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(extra or {})
    model = resolved.get("model")
    if isinstance(model, dict):
        kwargs = {**model, **kwargs, "model": model}
    elif model is not None:
        kwargs["model"] = model
    if resolved.get("seed") is not None:
        kwargs["seed"] = resolved["seed"]
    kwargs["runtime"] = dict(resolved.get("runtime") or {})
    kwargs["evaluation"] = dict(resolved.get("evaluation") or {})
    kwargs["max_agent_steps"] = first_defined(
        kwargs.get("max_agent_steps"),
        (resolved.get("runtime") or {}).get("max_agent_steps"),
    )
    return kwargs
