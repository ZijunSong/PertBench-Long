# Isolation

## Modes

| mode | meaning |
|---|---|
| `local_trusted_debug` | Broker PathGuard + env scrubbing. Agent code can still open host files if it knows the path. **Not** an isolation pass. |
| `isolated_eval` | Requires Docker. Probe: `ubuntu:22.04 --network=none --read-only --cap-drop=ALL`, mounts public+workspace only. Private manifest must not be visible in the container. |

## What was verified on this host (2026-09-20)

- Docker daemon reachable.
- `python:3.10-slim` was **not** local; pulling Docker Hub timed out. Probe uses local `ubuntu:22.04`.
- Probe: container prints `isolated_ok`; private manifest path is `blocked`.
- CPU Agent/baseline loop still runs on the host broker afterwards. `isolation_qualified=false`. This is not a silent pass of the official isolation gate.
- PathGuard rejects private paths, `/etc`, `/proc`, docker.sock, pickle/joblib, and obvious network commands in the shell tool.
- Broker never pickle-loads user artifacts.

If Docker is unavailable, `isolated_eval` returns `infra_error` / `ISOLATION_UNAVAILABLE` and does not fall back to debug while claiming isolation.
