# Isolation

## Modes

| mode | meaning |
|---|---|
| `local_trusted_debug` | Host subprocess for `run_python`, PathGuard, env scrub. Agent code can still open host files. **Never** `isolation_qualified`. |
| `isolated_eval` | Requires Docker and `runtime.analysis_image`. Tool code runs with `--network=none`, no private mounts, no Docker socket. If Docker/image is missing, the run is `infra_error` / `ISOLATION_UNAVAILABLE` and does **not** fall back to debug while claiming isolation. |

The model client may stay on the trusted host. Oracle, ledger, and ToolRouter stay on the host. Untrusted Python must not run in the process that holds private labels.

## What this revision implements

- Removed the `echo isolated_ok` probe that then executed the CPU agent on the host broker.
- `DockerPythonExecutor` bind-mounts declared public inputs only: `evidence/` (ro), `outputs/` (rw), `episode.json`, `genes_v1.tsv`, `submission_contract.json`.
- `isolation_qualified` is an instance result of `docker image inspect` digest matching `runtime.isolation_attestation.digest`. The class does not set this to True.
- Each tool task uses a unique container name; timeout kills and removes that container.
- Debug executor remains for engineering tests and is recorded as unqualified.

## Still unverified (2026-09-20)

- Building `docker/analysis.Dockerfile` to a pinned digest on this host.
- Running model-generated code that tries to read the private manifest, host secrets, or the network inside that image, and confirming those attempts fail while legal analysis succeeds.
- Disk quota for `workspace_gib`.

PathGuard string checks are not a substitute for that OS sandbox.
