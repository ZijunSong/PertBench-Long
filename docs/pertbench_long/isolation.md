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

## Isolation acceptance (must be run on a machine with Docker)

Copying `runtime.isolation_attestation.digest` into YAML is **not** acceptance. A person or CI job must execute the following against the **same** pinned image that `docker run` will use (`run_image_reference` in the run manifest):

```bash
# 1. Build and record the immutable image id
docker build -f docker/analysis.Dockerfile -t pertbench-long-analysis:local .
docker image inspect --format '{{.Id}}' pertbench-long-analysis:local

# 2. Legal path: read public evidence and write outputs (expect success)
# 3. Forbidden path: python that opens the private manifest, /etc/shadow, Docker socket, or a URL
#    (expect failure / timeout; container must not remain running)
# 4. After a forced timeout, `docker ps -a --filter name=pertbench-` is empty
```

Who is responsible: the operator who claims `isolation_qualified=true` for a formal suite. This repository does not treat a matching digest string as proof those commands passed.

`workspace_gib` is **not** enforced on the current host backends. If that field is set, the run cannot be isolation-qualified.

Formal / paper tables must use `official_eligible=true` only. Debug and unqualified Docker runs stay on an unofficial track.
