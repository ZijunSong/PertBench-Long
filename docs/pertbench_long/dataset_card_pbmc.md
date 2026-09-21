# Dataset card: PBMC IFN pilot

- **Study alias:** `kang_pbmc_ifn_public`
- **Species:** human (from dataset convention, not a per-cell metadata column)
- **Assay:** scRNA-seq (cells × genes tables)
- **Perturbation:** IFN vs Control, encoded in barcode suffix `-stimulated` / `-control`
- **Shipped source:** `data/releases/kang_pbmc_ifn_public.tar.gz` (extract with `pertbench-long unpack-data`)
- **Local fallback:** `/data/ppnm/data/PertDiffBench/data_ori/fig2/task2_unseen_celltype_plus/*.csv`
- **Processing history:** unknown beyond “already numeric, looks like log1p”. Original counts were not found. HVG scripts in PertDiffBench apply `normalize_total`+`log1p` again and are not used here.
- **Matrix kind:** `log1p` (inferred). Official scoring used `effect_proxy_v1_existing_log1p` (no second log1p).
- **Donor:** null. `replicate_de_v1` disabled.
- **Control pairing:** same cell type, `perturbation=Control`, listed as C. `target_control_available=true`.
- **Default split:** O = CD4T/CD8T IFN; Q = CD14+Mono, Dendritic, FCGR3A+Mono IFN; T = B, NK IFN. Control fraction shopping is not offered.
- **Labels:** `effect_direction_proxy`, tau=0.10 log1p mean difference, cell-weighted. Observed distribution on the 64-gene panel × 2 targets: down 49, neutral 52, up 27.
- **Gene panel:** episode-specific, variance on O+C only, n=64. Frozen G does not grow after purchases.
- **Independence:** public single-study pilot, fold-related, possible pretraining contamination. Not a private independent test.
- **Do not use:** old `combined_h5ad`, test IFN of B/NK as queryable items, or HVG files fitted on Q/T.
