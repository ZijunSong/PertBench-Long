# Implementation status (PertBench-Long)

Status values: **done** / **partial** / **blocked** / **deferred**.  
Date: 2026-09-20 (audit-fix revision). Commands were actually run unless a row says otherwise.  
Per-item audit mapping: `docs/pertbench_long/review_fix_status.md`.

## R01 · P0 · repository and data audit — done

- Files: `docs/pertbench_long/repository_audit.md`
- `inspect-data` requires `--config`; numeric range is diagnostic only

## R02 · P0 · new package — done

- CLI `pertbench-long`. Old `pertdiffbench` is a separate install, not bundled.
- `import pertbench_long` does not import torch/anndata

## R03 · P0 · public/private schema — done

- Unimplemented protocols are rejected, not just renamed
- `validate --check-artifacts` checks labels and Q files

## R04 · P0 · data adapter — done

- Gene align before identity/stack; official conflicts fail closed
- `declared_matrix_kind` required for official scoring

## R05 · P0 · measurement protocol — partial

- Declared kind drives transform; hidden values do not
- Undeclared PBMC is diagnostic, not official

## R06 · P0 · two-layer split — done

- T01 overlap rejected; T02 public development overlap → pilot flag

## R07 · P0 · condition-bundle purchases — done

- Unit cost 1; Q-only; T/unknown → `UNAVAILABLE_EXPERIMENT`

## R08 · P0 · oracle + SQLite ledger — done

- AUTHORIZED → MATERIALIZED → DELIVERED; unauthorized files are not visible
- E03/E04 regressions pass

## R09 · P0 · sandbox isolation — partial

- No probe-then-host fallback. Debug never qualified.
- Analysis image digest acceptance **unverified**

## R10 · P0 · public export scrub — done

- T04: named hidden layers/uns markers not present (anndata may expose a `None` layer key)

## R11 · P0 · state machine — done

- completed requires SUBMITTED + trusted receipt

## R12 · P0/P1 · tools/artifacts — done

- JSON ToolRouter; artifact registry; trusted snapshot index

## R13 · P1 · LLM loop — partial

- Mock HTTP CLI e2e passed; live local/API **unverified**

## R14 · P0 · submission format — done

- Full target×gene grid; no silent renormalize

## R15 · P0/P1 · labels — done

- NaN/Inf rejected; donor passed when present; `replicate_de_v1` still refused

## R16 · P0 · scores — done

- Binding hashes; workspace history cannot change AUBC

## R17 · P1 · budget curves — partial

- Trusted snapshots only. Historical PBMC ΔS not re-run after processing-history fixes.

## R18 · P0 · failures/aggregation — done

- `not_scored` / `integrity_failure` are not model score 0; group-by in summarize

## R19 · P1 · non-LLM baselines — done (synthetic CI)

- Five baselines on synthetic in tests. Real PBMC table is exploratory/pre-fix.

## R20 · P0/P1 · traces/replay — partial

- `replay`/`trace-summary` do not restore artifacts. `resume` now requires `run_progress.json` (and LLM `agent_messages.json`); OS crash-injection still unverified.

## R21 · P1 · interaction meaning — descriptive only

- PBMC query gain near zero remains a pilot limitation, not a solved long-horizon task

## R22 · P2 · predictor adapter — partial (interface only)

## R23 · P2 · knowledge retrieval — deferred/stub

## R24 · P2 · MOA / temporal — blocked/unsupported

## Long-horizon expansion (2026-09-22) — partial (Phase A + synthetic T1/T2/T3 loop)

- C01 task registry: `chemical_dose_acquisition_v1`, `genetic_pair_acquisition_v1`, `context_campaign_acquisition_v1` are implemented; MOA/temporal stay blocked
- C02 `cid_v1` condition identity: 1 uM == 1000 nM; A+B == B+A; unknown units refused for exact-dose protocols
- C03 adapters `data/adapters/sciplex.py` and `norman.py` require an explicit data directory
- C04 sparse row extract / library size without unconditional full `.toarray()`
- C05 `audit-conditions` and `data/audit_conditions.py` emit `insufficient_eligible_conditions`
- C06 generic builder from condition IDs; PBMC wrapper unchanged
- C07 dose interpolation/extrapolation, pair-holdout, context few-shot, and release cross-contamination checks
- C08 `lognorm_cellmean_delta_v1` is independent of PBMC `effect_proxy_v1`
- C09–C13 helpers/baselines/scoring dispatch by protocol; full `scores_curve` S0…SB
- Real sci-Plex3 / Norman downloads and live local/API model runs: **not_run**

## New-task readiness fixes (2026-09-22 review b286396) — partial

| Item | Status | Evidence |
|---|---|---|
| R01 fetch + source lock | implemented / tested_synthetic | `fetch-data`; null URLs listed, no guessed GEO links |
| R02 author-format prepare | implemented / tested_synthetic | `prepare-data` on fixture slices, not a GEO download |
| R03 real build CLI | implemented / tested_synthetic | `--dataset/--adapter/--protocol` + YAML; no synthetic fallback |
| R04 strict parse | implemented / tested_synthetic | no implicit counts/nM/h; `"false"` is false |
| R05 control match | implemented / tested_synthetic | study/context/assay/time/vehicle; no `candidates[0]` |
| R06 C/T isolation | implemented / tested_synthetic | build + `check_release` reject T-in-C |
| R07 replicate estimand | implemented / tested_synthetic | 9:1 cells → 5 vs 1 by named estimand |
| R08 sparse transform | implemented / tested_synthetic | no full densify sentinel |
| R09 release partitions | implemented / tested_synthetic | missing partition / eval T in public dev rejected |
| R10 seeded sampling | implemented / tested_synthetic | reorder + same seed keeps T; unknown kwargs rejected |
| R11 official gate | implemented / tested_synthetic | `local_unreleased` stays pilot |
| R12 public artifacts | implemented / tested_synthetic | task.md / mapping copied and hashed |
| R13 task baselines | implemented / tested_synthetic | nearest dose / additivity / context transfer; live model **not_run** |
| R14 docs | implemented | README + cards updated; remaining GEO lock is honest |

## Tests

`python -m pytest tests/pertbench_long -q` after the readiness fixes: **96 passed, 6 skipped**.  
Skipped items are live model / Docker isolation paths (`not_run`).

## Not executed / not claimed

- Live LLM eval (no credentials / no running local server in this revision)
- Pinned Docker analysis image isolation acceptance
- Official PBMC scoring after confirmed processing history
- Blank-venv wheel install on a new machine
- Any claim that a long-horizon scientific discovery benchmark is complete
