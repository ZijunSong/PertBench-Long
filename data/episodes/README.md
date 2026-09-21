# Built episode packs

These packs were built on the author machine and are enough to `run` / `score` without calling `build-episodes`.

| Pack | Track | Rebuild command |
|---|---|---|
| `synthetic_pilot_001` | synthetic / not official | `pertbench-long build-episodes --fixture synthetic --output data/episodes` |
| `pbmc_pilot_001` | diagnostic PBMC IFN | unpack CSVs, then `pertbench-long build-episodes --config configs/pertbench_long/pbmc_pilot.yaml --fixture pbmc --output data/episodes` |

Private labels are in `*/private/`. That split is for the Agent at runtime, not a claim that the files are secret from the repository.
