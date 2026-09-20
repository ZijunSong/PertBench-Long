"""Filesystem and command guards used by the broker. Naming a folder private/ is not isolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from pertbench_long.errors import IsolationUnavailable, PathGuardError

BLOCKED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MINIMAX_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
)

BLOCKED_COMMAND = re.compile(r"\b(curl|wget|nc|ncat|ssh|docker|python\s+-c\s+['\"]import\s+urllib)")


def resolve_under(path: Path | str, roots: Sequence[Path]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (roots[0] / candidate).resolve()
    else:
        candidate = candidate.resolve()
    allowed = [root.resolve() for root in roots]
    for root in allowed:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PathGuardError("path is outside the allowed workspace")


class PathGuard:
    def __init__(self, allowed_roots: Iterable[Path | str], *, denied: Iterable[Path | str] = ()) -> None:
        self.allowed = [Path(p).resolve() for p in allowed_roots]
        self.denied = [Path(p).resolve() for p in denied]

    def check(self, path: Path | str) -> Path:
        raw = Path(path)
        # Reject parent traversal before resolve if the original string tries to escape via ..
        text = str(path)
        if text.startswith("~") or text.startswith("/proc") or text.startswith("/etc"):
            raise PathGuardError("path is outside the allowed workspace")
        resolved = resolve_under(raw, self.allowed)
        for denied in self.denied:
            try:
                resolved.relative_to(denied)
                raise PathGuardError("path is outside the allowed workspace")
            except ValueError:
                continue
        if resolved.is_symlink():
            target = resolved.resolve()
            try:
                resolve_under(target, self.allowed)
            except PathGuardError:
                raise PathGuardError("symlink target is outside the allowed workspace") from None
        return resolved


def scrub_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for key in list(env):
        upper = key.upper()
        if any(tok in upper for tok in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "DOCKER")):
            env.pop(key, None)
        if key in BLOCKED_ENV_KEYS:
            env.pop(key, None)
    env.pop("DOCKER_HOST", None)
    return env


def command_allowed(command: str) -> None:
    if BLOCKED_COMMAND.search(command):
        raise PathGuardError("command is not allowed")
    if " /" in f" {command}" and any(tok in command for tok in ("/etc", "/root", "/home", "/data", "/proc", "/var/run/docker.sock")):
        raise PathGuardError("command is not allowed")


def docker_available() -> bool:
    import subprocess

    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=8, check=False)
        return proc.returncode == 0
    except Exception:
        return False


def require_isolated_backend(mode: str) -> str:
    if mode == "local_trusted_debug":
        return "debug_untrusted"
    if mode != "isolated_eval":
        raise IsolationUnavailable(f"unknown mode {mode}")
    if docker_available():
        return "docker"
    raise IsolationUnavailable(
        "isolated_eval requires an OS sandbox (Docker). This host cannot provide isolation; "
        "debug mode may still be used and is not a passing isolation result."
    )
