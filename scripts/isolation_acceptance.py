#!/usr/bin/env python3
"""Write an isolation acceptance report. Docker absence is not_run, not passed."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    report = {
        "report_version": "isolation_acceptance_v1",
        "status": "not_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mount_policy": "declared_public_v1",
        "image_digest": None,
        "tests": {},
        "reason": "docker_not_available",
        "note": "Digest equality is image_pinned only. This report does not grant isolation_qualified.",
    }
    if shutil.which("docker") is None:
        out = Path(sys.argv[1] if len(sys.argv) > 1 else "isolation_acceptance.json")
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 3
    report["reason"] = "acceptance_commands_not_executed"
    report["note"] = "Docker is installed, but this environment did not execute the container boundary suite."
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "isolation_acceptance.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
