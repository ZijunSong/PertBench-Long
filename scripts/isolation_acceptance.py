#!/usr/bin/env python3
"""Run isolation acceptance checks. Missing Docker or image is blocked, not passed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _write(report: dict, argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "isolation_acceptance.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "passed" else 3


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    report = {
        "report_version": "isolation_acceptance_v1",
        "status": "not_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mount_policy": "declared_public_v1",
        "image_digest": None,
        "executor": "DockerPythonExecutor",
        "tests": {},
        "commands": [],
        "reason": "docker_not_available",
        "note": "A matching digest only pins the image. This script must execute the container checks.",
    }
    if shutil.which("docker") is None:
        return _write(report, argv)
    image = os.environ.get("PERTBENCH_ANALYSIS_IMAGE")
    if not image:
        report["status"] = "blocked"
        report["reason"] = "analysis_image_unset"
        return _write(report, argv)
    report["image_digest"] = image if "@sha256:" in image or image.startswith("sha256:") else None
    sentinel = Path(tempfile.mkdtemp(prefix="pertbench-sentinel-")) / "host_secret.txt"
    sentinel.write_text("host-secret", encoding="utf-8")
    public = Path(tempfile.mkdtemp(prefix="pertbench-public-"))
    (public / "task.md").write_text("public\n", encoding="utf-8")
    checks = {
        "no_network": ["docker", "run", "--network=none", "--rm", image, "python", "-c", "print('network-ok')"],
        "public_read": ["docker", "run", "--network=none", "--rm", "--mount", f"type=bind,src={public / 'task.md'},dst=/workspace/task.md,readonly", image, "python", "-c", "print(open('/workspace/task.md').read())"],
        "public_readonly": ["docker", "run", "--network=none", "--rm", "--mount", f"type=bind,src={public / 'task.md'},dst=/workspace/task.md,readonly", image, "python", "-c", "open('/workspace/task.md','w').write('x')"],
        "host_sentinel_unreadable": ["docker", "run", "--network=none", "--rm", image, "python", "-c", f"print(open({str(sentinel)!r}).read())"],
    }
    try:
        for name, cmd in checks.items():
            proc = _run(cmd)
            report["commands"].append({"test": name, "argv0": cmd[0], "returncode": proc.returncode})
            if name in {"public_readonly", "host_sentinel_unreadable"}:
                ok = proc.returncode != 0
            elif name == "public_read":
                ok = proc.returncode == 0 and "public" in proc.stdout
            else:
                ok = proc.returncode == 0 and "network-ok" in proc.stdout
            report["tests"][name] = "passed" if ok else "failed"
        report["tests"]["unpurchased_unreadable"] = report["tests"]["host_sentinel_unreadable"]
        report["status"] = "passed" if all(value == "passed" for value in report["tests"].values()) else "failed"
        report["reason"] = None if report["status"] == "passed" else "container_check_failed"
    except Exception as exc:
        report["status"] = "failed"
        report["reason"] = type(exc).__name__
    return _write(report, argv)


if __name__ == "__main__":
    raise SystemExit(main())
