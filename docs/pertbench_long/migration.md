# Migration from PertDiffBench

PertBench-Long is a new evaluation protocol. It is not a rename of PertDiffBench.

## Still usable from the old repo

- Raw/local CSVs under `/data/ppnm/data/PertDiffBench/data_ori/` as **sources** for a new canonical store.
- Old CLI `pertdiffbench list-tasks` / `pertdiffbench run ...` for diffusion-model figure reproduction.
- Old model checkpoints only in an explicit diagnostic predictor track, and only with training-split provenance that excludes current Q (unpurchased) and T.

## Must be rebuilt

- train/test/combined h5ad as Agent-visible episode packs.
- LOO + control-fraction tasks as O/Q/T/C (do not sell the test cell type’s IFN).
- HVG/PCA/encoders fitted on files that included Q/T.
- Regex-parsed PDS/DES/MMD as the Agent main score.
- MOA filenames as Agent-visible paths if MOA is the answer.

## Reproduce old PertDiffBench

```bash
conda activate pertdiffbench
cd /data/ppnm/PertDiffBench
pip install -e .
pertdiffbench list-tasks
```

See `/data/ppnm/PertDiffBench/README.md` and `docs/`.

## Shortest new-environment commands

```bash
conda activate pertdiffbench
cd /data/ppnm/PertBench-Long
pip install -e '.[dev]'
pertbench-long list
pertbench-long smoke --fixture synthetic --agent scripted_mock --mode local_trusted_debug
```
