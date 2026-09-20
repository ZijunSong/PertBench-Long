# Dataset card: MOA (P2, not enabled)

- **Local files:** `/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseenMOA/`
- **Status:** `unsupported`
- **Why:** train/test filenames include MOA class strings. If the task is to recover MOA, those names cannot be Agent-visible. Assay (single-cell vs bulk), dose units, and donor/replicates are unverified.
- **Hypothesis-discrimination scoring:** not enabled (no pre-registered H1/H2 experiments).
- **Validator:** `pertbench_long.extensions.moa.audit_moa_root`
- Official MOA benchmark is **not** reported as complete.
