# Implementation status (PertBench-Long v0.1)

Status values: **done** / **partial** / **blocked** / **deferred**.  
Date: 2026-09-20. Commands were actually run unless a row says otherwise.

## R01 · P0 · repository and data audit — done

- Files: `docs/pertbench_long/repository_audit.md`, `docs/pertbench_long/data_inventory.json`
- Command: `pertbench-long inspect-data --config configs/pertbench_long/pbmc_data.yaml --output docs/pertbench_long/data_inventory.json`
- Local HEAD `ca2c628`; dirty PertDiffBench tree preserved. PBMC CSVs classified `log1p`; donor=null.

## R02 · P0 · new package, old CLI intact — done

- Files: `pyproject.toml`, `pertbench_long/`, CLI `pertbench-long`
- `pip install -e .` in conda env `pertdiffbench`; `pertbench-long list` works without extra PYTHONPATH
- `pertdiffbench list-tasks` still works
- `import pertbench_long` does not import torch/anndata

## R03 · P0 · public/private schema — done

- Files: `pertbench_long/schemas/types.py`, `validate.py`, `schemas/jsonschema/public_episode.schema.json`
- Tests: unknown field, incompatible version, public leak keys
- Validator creates no directories

## R04 · P0 · data adapter — done

- Files: `pertbench_long/data/adapter.py`
- Import summary: kept/dedup/conflict/dropped; gene-order hash
- Test: `tests/pertbench_long/test_import.py`

## R05 · P0 · measurement protocol — done

- Files: `pertbench_long/data/preprocess.py`
- counts → 10k then log1p on full universe; existing log1p not reapplied; scaled/unknown rejected
- T03: hidden T shift does not change public fingerprint

## R06 · P0 · two-layer split — done

- Files: `pertbench_long/episodes/split.py`, `runs/episodes/*/private/split_audit.json`
- T01 overlap rejected; T02 public development overlap → pilot flag
- PBMC split_audit flags `single_public_study_fold_related`

## R07 · P0 · condition-bundle purchases — done

- Unit cost 1; Q-only; `UNAVAILABLE_EXPERIMENT` for T/unknown without revealing mapping
- Duplicate purchase does not charge again

## R08 · P0 · oracle + SQLite ledger — done

- Files: `pertbench_long/oracle/ledger.py`, `service.py`
- Tests T05–T08: budget, idempotency, last-credit race, recovery

## R09 · P0 · sandbox isolation — partial

- Files: `pertbench_long/runtime/sandbox.py`, `docs/pertbench_long/isolation.md`
- PathGuard tests T09/T10 pass
- `isolated_eval` Docker probe with local `ubuntu:22.04` succeeded; CPU agent still host-side; `isolation_qualified=false`
- Not a silent debug downgrade

## R10 · P0 · public export scrub — done

- Files: `pertbench_long/episodes/export.py`
- T04: marker in uns/raw/layers not present in rebuilt h5ad

## R11 · P0 · state machine — done

- Files: `pertbench_long/runtime/state.py`, `broker.py`
- Scripted policies: zero-query, two-query, early-stop
- Submit blocks further purchase

## R12 · P0/P1 · tools/artifacts — done

- list/read evidence, purchase, snapshot, submit; helpers QC/effects/similarity
- Parquet predictions; no pickle load
- Path traversal rejected

## R13 · P1 · LLM adapter + mock — partial

- `scripted_mock` runs through oracle/tools (CI)
- `LLMAdapter` implemented; **live run unverified** (no key/endpoint)
- T20: missing key → `infra_error`, not a fake live success

## R14 · P0 · submission format — done

- Full target×gene grid; no silent renormalize; T11

## R15 · P0/P1 · labels — done

- Default `effect_proxy_v1` / `effect_direction_proxy`
- `replicate_de_v1` refused without donor+counts (T13)
- PBMC label card written

## R16 · P0 · scores — done

- T12: perfect=1, wrong one-hot=0, uniform=2/3; constant Pearson NA
- Full precision; no LLM judge

## R17 · P1 · budget curves — done

- S(0)/S(1)/S(2), AUBC unit-cost; negative gain kept (PBMC random_query ΔS=−0.0039)

## R18 · P0 · failures/aggregation — done

- T17: failures stay in denominator; `nanmean` not used
- Infra vs science vs invalid_episode distinguished

## R19 · P1 · non-LLM baselines — done

- Five baselines ran on synthetic **and** real PBMC
- Commands: `pertbench-long run --public-dir ... --agent {no_change,mean_delta_no_query,random_query,fixed_order_query,control_similarity_query}`
- Results: `runs/pbmc_baselines/summary.json`, `docs/pertbench_long/interaction_value_report.md`
- LLM_no_query / LLM_adaptive_query: adapter only, live unverified

## R20 · P0/P1 · traces/replay — done

- `run_manifest.json`, `events.jsonl`, `budget_ledger.sqlite`, trusted submissions
- `pertbench-long replay --run ...` executed
- Scores written beside the run, not in Agent workspace

## R21 · P1 · interaction meaning — done (descriptive)

- Synthetic: queries move scores by design
- PBMC 64-gene panel: |ΔS| ≲ 0.005; random slightly worse
- Report states limited sequential information; no “LLM must win” criterion

## R22 · P2 · predictor adapter — partial (interface only)

- `fit(visible_evidence, config)` / `predict(...)`; T15 rejects T-provenance checkpoints
- Old `BenchmarkRunner.run` is not an Agent tool

## R23 · P2 · knowledge retrieval — deferred/stub

- Disabled corpus; main score needs no retrieval/judge
- Claim checks: cited evidence must be visible

## R24 · P2 · MOA / temporal — blocked/unsupported

- Audits: `extensions/moa.py`, `extensions/temporal.py`
- Dataset cards written; no random-matrix labels

## Tests T01–T20

Executed via `python -m pytest tests/pertbench_long -q` (28 passed). Mapping:

- T01–T04 schema/split/export: pass
- T05–T08 oracle: pass
- T09 isolated official claim: **not qualified**; PathGuard + blocked silent pass: pass
- T10 path/pickle: pass
- T11–T14 scoring/labels/AUBC: pass
- T15 predictor: pass
- T16 trusted snapshot: pass
- T17 outcomes: pass
- T18 workspace isolation between runs: pass
- T19 old+new CLI / smoke: pass
- T20 resource/LLM unverified: pass (tool-count limit configured; live LLM not called)

## Not executed

- Live LLM eval (no credentials)
- Full OS jail of the Agent loop inside Docker
- Official MOA/temporal labels
- Cluster bootstrap over independent studies (only one study)
- Any claim that a long-horizon scientific discovery benchmark is complete
