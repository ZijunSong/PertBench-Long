# Dataset card: A549 temporal (P2, blocked)

- **Local files:** `/data/ppnm/data/PertDiffBench/data_ori/fig4/GSM3770930_A549_lognorm_scale_hvg3000.csv` and `GSM3770930_A549_cell_annotate.txt`
- **Status:** `blocked`
- **Why:** filename asserts log-normalization, scaling, and HVG=3000. Full-data feature selection cannot be undone. PBMC `tau=0.10` log1p proxy must not be reused.
- **If enabled later:** retrospective interpolation, e.g. O={0h,10h}, Q={2h,8h}, T={4h,6h}, B=1. This is not future forecasting. 0h is a `reference_time`, not automatically a matched untreated control.
- **Validator:** `pertbench_long.extensions.temporal.audit_temporal_root`
