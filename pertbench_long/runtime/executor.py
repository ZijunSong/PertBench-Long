"""Python/shell workers. Model-generated code does not run in the oracle process.

debug: host subprocess with timeout (not isolation-qualified).
isolated_eval: Docker worker with no network, no private mounts, no Docker socket.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from pertbench_long.errors import IsolationUnavailable, PathGuardError
from pertbench_long.runtime.sandbox import docker_available, scrub_env


class PythonExecutor:
    isolation_qualified = False
    backend = "debug_untrusted"

    def run_python(self, code: str, *, timeout_s: int, cwd: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def run_shell(self, command: str, *, timeout_s: int, cwd: Path) -> dict[str, Any]:
        raise NotImplementedError


class DebugPythonExecutor(PythonExecutor):
    isolation_qualified = False
    backend = "debug_untrusted"

    def run_python(self, code: str, *, timeout_s: int, cwd: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        cwd = Path(cwd)
        script = cwd / "outputs" / "_worker_job.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code, encoding="utf-8")
        env = scrub_env()
        env["PERTBENCH_WORKSPACE"] = str(cwd)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1].parent) + os.pathsep + env.get("PYTHONPATH", "")
        if extra_env:
            env.update(extra_env)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=int(timeout_s),
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "error_code": "TOOL_TIMEOUT",
                "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
                "stderr": "timeout",
                "elapsed_s": time.monotonic() - started,
                "exit_code": None,
            }
        stdout = (proc.stdout or "")[-8000:]
        stderr = (proc.stderr or "")[-8000:]
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_s": time.monotonic() - started,
            "exit_code": proc.returncode,
            "isolation_qualified": False,
        }

    def run_shell(self, command: str, *, timeout_s: int, cwd: Path) -> dict[str, Any]:
        raise PathGuardError("run_shell is disabled in debug; use run_python")


class DockerPythonExecutor(PythonExecutor):
    isolation_qualified = True
    backend = "docker"

    def __init__(self, image: str, *, memory_gib: int = 8, cpu: int = 2, pids: int = 64) -> None:
        self.image = image
        self.memory_gib = memory_gib
        self.cpu = cpu
        self.pids = pids
        if not docker_available():
            raise IsolationUnavailable("isolated_eval requires Docker; this host cannot isolate tool code")

    def _docker(self, args: list[str], *, timeout_s: int) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=int(timeout_s) + 15, check=False)

    def run_python(self, code: str, *, timeout_s: int, cwd: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        cwd = Path(cwd)
        (cwd / "outputs").mkdir(parents=True, exist_ok=True)
        (cwd / "evidence").mkdir(parents=True, exist_ok=True)
        job = cwd / "outputs" / "_worker_job.py"
        job.write_text(code, encoding="utf-8")
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--cpus",
            str(self.cpu),
            "--memory",
            f"{self.memory_gib}g",
            "--pids-limit",
            str(self.pids),
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--mount",
            f"type=bind,src={cwd / 'evidence'},dst=/workspace/evidence,readonly",
            "--mount",
            f"type=bind,src={cwd / 'outputs'},dst=/workspace/outputs",
            "--mount",
            f"type=bind,src={cwd / 'episode.json'},dst=/workspace/episode.json,readonly",
            "-w",
            "/workspace",
            "--tmpfs",
            "/tmp:size=64m",
            "-e",
            "HOME=/tmp",
            "-e",
            "PERTBENCH_WORKSPACE=/workspace",
            self.image,
            "python",
            "/workspace/outputs/_worker_job.py",
        ]
        started = time.monotonic()
        try:
            proc = self._docker(cmd, timeout_s=timeout_s)
        except FileNotFoundError:
            raise IsolationUnavailable("docker binary is not available")
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error_code": "TOOL_TIMEOUT",
                "stdout": "",
                "stderr": "timeout",
                "elapsed_s": time.monotonic() - started,
                "exit_code": None,
                "isolation_qualified": True,
            }
        if proc.returncode != 0 and "Unable to find image" in (proc.stderr or ""):
            raise IsolationUnavailable(f"analysis image {self.image!r} is not available")
        return {
            "ok": proc.returncode == 0,
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
            "elapsed_s": time.monotonic() - started,
            "exit_code": proc.returncode,
            "isolation_qualified": True,
            "image": self.image,
        }

    def run_shell(self, command: str, *, timeout_s: int, cwd: Path) -> dict[str, Any]:
        raise PathGuardError("run_shell is not enabled; use run_python")


def make_executor(mode: str, *, image: str | None = None, memory_gib: int = 8, cpu: int = 2) -> PythonExecutor:
    if mode == "local_trusted_debug":
        return DebugPythonExecutor()
    if mode != "isolated_eval":
        raise IsolationUnavailable(f"unknown mode {mode}")
    if not image:
        raise IsolationUnavailable("isolated_eval requires runtime.analysis_image")
    return DockerPythonExecutor(image, memory_gib=memory_gib, cpu=cpu)
