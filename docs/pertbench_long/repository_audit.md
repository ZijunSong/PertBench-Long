# Repository audit (R01)

Generated: 2026-09-20  
Workspace: `/data/ppnm`  
New package root: `/data/ppnm/PertBench-Long`  
Sibling legacy repo: `/data/ppnm/PertDiffBench`

## Local HEAD and dirty tree

PertDiffBench `git rev-parse HEAD`:

`ca2c62810befc430bec76a6b4e4ec65e01daf371`  
message: `Unify experiment seeds and evaluation sample counts across pipelines.`  
branch: `main` tracking `origin/main`

This matches the commit named in the implementation spec. Local tree is **dirty**. Uncommitted edits are concentrated in encoder/baseline YAML, shell scripts, and a few `src/` files. Those files were **not** overwritten. PertBench-Long is a new sibling folder, not an in-place rename of PertDiffBench.

PertBench-Long itself is not a git repository.

## Section 1 of the spec vs local code

| Spec location | Local fact | Action taken |
|---|---|---|
| `pyproject.toml` package `pertdiffbench`, CLI `pertdiffbench`, find `pertdiffbench*` | Unchanged | New package `pertbench-long` / CLI `pertbench-long` in the sibling folder |
| `TaskSpec` creates output dirs on init | Confirmed in `PertDiffBench/pertdiffbench/tasks/base.py` | New schemas validate with no I/O |
| `BenchmarkRunner` reads test for n-samples | Confirmed | Old runner is not used as an Agent tool |
| `cross_celltype_plus` LOO + control fraction | Confirmed | Rebuilt as O/Q/T/C; control fraction is **not** a purchase |
| PBMC LOO preprocessor | Confirmed; barcodes may overlap train/valid | Canonical import dedups identical keys, reports numeric conflicts |
| MOA filenames contain MOA labels | Confirmed | MOA track blocked/unsupported; names must not be Agent-visible answers |
| Temporal A549 0/2/8/10h → 4/6h | Confirmed; file is `lognorm_scale_hvg3000` | Interpolation protocol only; official labels blocked |
| `evaluate.py` regex metrics | Confirmed | New scorer returns structured numbers |
| `utils/metrics.py` rounds early | Confirmed | New metrics keep full precision; old PDS/DES/MMD not used as main score |
| HVG script fits per input file and re-normalizes | Confirmed | Old HVG files are not treated as leak-free inputs |

No `AGENTS.md` exists in PertDiffBench.

## CLI and tests

- Legacy: `/data/ppnm/miniconda3/envs/pertdiffbench/bin/pertdiffbench list-tasks` works (`known_condition`, `cross_celltype`, `moa_*`, `temporal`, …).
- New: `/data/ppnm/miniconda3/envs/pertdiffbench/bin/pertbench-long` after `pip install -e .` in this folder. No extra PYTHONPATH required.
- New tests: `python -m pytest tests/pertbench_long` → 28 passed (2026-09-20).
- PertDiffBench has no first-party package tests (only vendored `src/*/tests`).

## Data locations

Canonical CSVs: `/data/ppnm/data/PertDiffBench/data_ori/`  
Processed h5ad: `/data/ppnm/data/PertDiffBench/data/` and some copies under `PertDiffBench/data/` (gitignored).

See `docs/pertbench_long/data_inventory.json`.

PBMC fig2 plus CSVs are present (14 train/valid files, seven cell types). Opened `task1_train_B_exp.csv`: 1811 × 6998, values in [0, ~7], classified `log1p`. No donor/study/batch columns. Perturbation is encoded in the barcode suffix (`-control` / `-stimulated`). Cell type comes from the filename / `Cell.Type` when present.

Fig4 `GSM3770930_A549_lognorm_scale_hvg3000.csv` exists; scale+HVG provenance cannot be undone.

MOA CSVs exist; many filenames contain the MOA class.

## Dependencies (conda env `pertdiffbench`)

Python 3.10.19. Present: numpy, pandas, anndata 0.11.4, scanpy, scipy, h5py, pyarrow, pyyaml, pytest 9.0.3, torch (not imported by `import pertbench_long`).

Docker client is installed. `ubuntu:22.04` is local. Isolated probe can run. Official isolation comparison is **not** claimed: the CPU agent loop still uses the host broker after the probe.

## Required inputs if data were missing

```
/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus/task1_{train,valid}_{B,CD4T,CD8T,CD14+Mono,Dendritic,FCGR3A+Mono,NK}_exp.csv
```

Synthetic smoke does not need those files.
