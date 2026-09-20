"""Python/shell workers. Model-generated code does not run in the oracle process.

debug: host subprocess with timeout (not isolation-qualified).
isolated_eval: Docker worker with no network, no private mounts, no Docker socket.
isolation_qualified is an instance result of digest+attestation, never a class constant.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from pertbench_long.errors import IsolationUnavailable, PathGuardError
from pertbench_long.runtime.sandbox import docker_available, scrub_env

MAX_CAPTURE_BYTES = 8_000_000


def normalize_digest(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if "@sha256:" in text:
        text = "sha256:" + text.split("@sha256:", 1)[1]
    if text.startswith("sha256:"):
        return text
    if len(text) >= 64 and all(c in "0123456789abcdef" for c in text[:64].lower()):
        return "sha256:" + text
    return text


def isolation_is_qualified(*, mode: str, backend: str, resolved_digest: str | None, attestation_digest: str | None) -> bool:
    if mode != "isolated_eval" or backend != "docker":
        return False
    got = normalize_digest(resolved_digest)
    want = normalize_digest(attestation_digest)
    if not got or not want:
        return False
    return got == want


def _bounded_communicate(proc: subprocess.Popen, *, timeout_s: float, max_bytes: int = MAX_CAPTURE_BYTES) -> tuple[str, str, str | None]:
    """Read stdout/stderr with a byte cap. Does not wait for unbounded output."""
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            leftover_out, leftover_err = proc.communicate(timeout=5)
        except Exception:
            leftover_out, leftover_err = b"", b""
        stdout = (leftover_out or b"").decode("utf-8", errors="replace")[-8000:]
        return stdout, "timeout", "timeout"
    stdout_b = stdout_b or b""
    stderr_b = stderr_b or b""
    overflow = None
    if len(stdout_b) + len(stderr_b) > max_bytes:
        overflow = "output_truncated"
        stdout_b = stdout_b[: max_bytes // 2]
        stderr_b = stderr_b[: max_bytes // 2]
    def _decode(raw: bytes) -> str:
        text = raw.decode("utf-8", errors="replace")
        return text[-8000:]
    return _decode(stdout_b), _decode(stderr_b), overflow


class PythonExecutor:
    backend = "debug_untrusted"

    def __init__(self) -> None:
        self.isolation_qualified = False
        self.resolved_digest = None
        self.image = None

    def run_python(self, code: str, *, timeout_s: int, cwd: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def run_shell(self, command: str, *, timeout_s: int, cwd: Path) -> dict[str, Any]:
        raise NotImplementedError

    def public_mounts(self, cwd: Path) -> list[str]:
        return []


class DebugPythonExecutor(PythonExecutor):
    backend = "debug_untrusted"

    def __init__(self) -> None:
        super().__init__()
        self.isolation_qualified = False

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
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr, overflow = _bounded_communicate(proc, timeout_s=float(timeout_s))
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
            raise
        if overflow == "timeout":
            proc.kill()
            proc.wait(timeout=5)
            return {
                "ok": False,
                "error_code": "TOOL_TIMEOUT",
                "stdout": stdout,
                "stderr": "timeout",
                "elapsed_s": time.monotonic() - started,
                "exit_code": None,
                "isolation_qualified": False,
            }
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_s": time.monotonic() - started,
            "exit_code": proc.returncode,
            "isolation_qualified": False,
            "output_truncated": overflow == "output_truncated",
        }

    def run_shell(self, command: str, *, timeout_s: int, cwd: Path) -> dict[str, Any]:
        raise PathGuardError("run_shell is disabled in debug; use run_python")


class DockerPythonExecutor(PythonExecutor):
    backend = "docker"

    def __init__(
        self,
        image: str,
        *,
        memory_gib: int = 8,
        cpu: int = 2,
        pids: int = 64,
        required_digest: str | None = None,
        mode: str = "isolated_eval",
    ) -> None:
        super().__init__()
        self.image = image
        self.memory_gib = memory_gib
        self.cpu = cpu
        self.pids = pids
        self.required_digest = required_digest
        self.isolation_qualified = False
        if not docker_available():
            raise IsolationUnavailable("isolated_eval requires Docker; this host cannot isolate tool code")
        self.resolved_digest = self._resolve_digest(image)
        self.isolation_qualified = isolation_is_qualified(
            mode=mode,
            backend="docker",
            resolved_digest=self.resolved_digest,
            attestation_digest=required_digest,
        )
        if required_digest and not self.isolation_qualified:
            raise IsolationUnavailable("analysis image digest does not match runtime.isolation_attestation.digest")

    def _resolve_digest(self, image: str) -> str:
        pinned = normalize_digest(image if "@sha256:" in image else None)
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}} {{json .RepoDigests}}", image],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except FileNotFoundError as exc:
            raise IsolationUnavailable("docker binary is not available") from exc
        except subprocess.TimeoutExpired as exc:
            raise IsolationUnavailable("docker image inspect timed out") from exc
        if proc.returncode != 0:
            raise IsolationUnavailable(f"analysis image {image!r} is not available")
        line = (proc.stdout or "").strip()
        image_id = line.split(" ", 1)[0] if line else ""
        digest = pinned or normalize_digest(image_id)
        if not digest:
            raise IsolationUnavailable(f"could not resolve a digest for analysis image {image!r}")
        return digest

    def _kill_container(self, name: str) -> None:
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=20, check=False)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=20, check=False)

    def _container_running(self, name: str) -> bool:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return (proc.stdout or "").strip().lower() == "true"

    def _mounts(self, cwd: Path) -> list[str]:
        cwd = Path(cwd)
        mounts = [
            "--mount",
            f"type=bind,src={cwd / 'evidence'},dst=/workspace/evidence,readonly",
            "--mount",
            f"type=bind,src={cwd / 'outputs'},dst=/workspace/outputs",
            "--mount",
            f"type=bind,src={cwd / 'episode.json'},dst=/workspace/episode.json,readonly",
        ]
        for name in ("genes_v1.tsv", "submission_contract.json"):
            src = cwd / name
            if src.exists():
                mounts.extend(["--mount", f"type=bind,src={src},dst=/workspace/{name},readonly"])
        return mounts

    def run_python(self, code: str, *, timeout_s: int, cwd: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        cwd = Path(cwd)
        (cwd / "outputs").mkdir(parents=True, exist_ok=True)
        (cwd / "evidence").mkdir(parents=True, exist_ok=True)
        job = cwd / "outputs" / "_worker_job.py"
        job.write_text(code, encoding="utf-8")
        name = f"pertbench-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker",
            "run",
            "--name",
            name,
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
            *self._mounts(cwd),
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
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise IsolationUnavailable("docker binary is not available") from exc
        stdout, stderr, overflow = _bounded_communicate(proc, timeout_s=float(timeout_s) + 5)
        timed_out = overflow == "timeout" or (time.monotonic() - started) > float(timeout_s)
        if timed_out:
            self._kill_container(name)
            if self._container_running(name):
                return {
                    "ok": False,
                    "error_code": "TOOL_TIMEOUT",
                    "stdout": stdout,
                    "stderr": "timeout; container still running after kill",
                    "elapsed_s": time.monotonic() - started,
                    "exit_code": None,
                    "isolation_qualified": False,
                    "container": name,
                }
            return {
                "ok": False,
                "error_code": "TOOL_TIMEOUT",
                "stdout": stdout,
                "stderr": "timeout",
                "elapsed_s": time.monotonic() - started,
                "exit_code": None,
                "isolation_qualified": self.isolation_qualified,
                "container": name,
            }
        self._kill_container(name)
        if proc.returncode not in (0, None) and "Unable to find image" in (stderr or ""):
            raise IsolationUnavailable(f"analysis image {self.image!r} is not available")
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_s": time.monotonic() - started,
            "exit_code": proc.returncode,
            "isolation_qualified": self.isolation_qualified,
            "image": self.image,
            "image_digest": self.resolved_digest,
            "output_truncated": overflow == "output_truncated",
            "container": name,
        }

    def run_shell(self, command: str, *, timeout_s: int, cwd: Path) -> dict[str, Any]:
        raise PathGuardError("run_shell is not enabled; use run_python")


def make_executor(
    mode: str,
    *,
    image: str | None = None,
    memory_gib: int = 8,
    cpu: int = 2,
    required_digest: str | None = None,
) -> PythonExecutor:
    if mode == "local_trusted_debug":
        return DebugPythonExecutor()
    if mode != "isolated_eval":
        raise IsolationUnavailable(f"unknown mode {mode}")
    if not image:
        raise IsolationUnavailable("isolated_eval requires runtime.analysis_image")
    return DockerPythonExecutor(image, memory_gib=memory_gib, cpu=cpu, required_digest=required_digest, mode=mode)
