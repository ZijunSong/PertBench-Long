# Interaction value report

This report is a development diagnostic. It is not an official scientific score.

## Synthetic fixture (designed programs)

Querying myeloid Q conditions improved `direction_score` from 0.75 (no-query mean-delta) to ~0.88. That only shows the fixture can produce different trajectories. It is not biology.

## PBMC public-data pilot (`pbmc_pilot_001`)

- Protocol: `within_study_celltype_ood_v1`
- Labels: `effect_direction_proxy` / `effect_proxy_v1`, tau=0.10 log1p, cell-weighted (no donor)
- Gene panel: 64 genes selected from O+C only (episode-specific)
- O: CD4T/CD8T IFN; Q: CD14+Mono / Dendritic / FCGR3A+Mono IFN; T: B/NK IFN; C: controls for listed types
- Budget B=2, unit cost

| strategy | direction_score | S(0) | S(B)-S(0) |
|---|---:|---:|---:|
| no_change (neutral one-hot) | 0.406 | 0.406 | 0 |
| mean_delta_no_query | 0.800 | 0.800 | 0 |
| random_query | 0.796 | 0.800 | **-0.0039** |
| fixed_order_query | 0.802 | 0.800 | +0.0019 |
| control_similarity_query | 0.805 | 0.800 | +0.0047 |

Negative query gain is retained (not clipped). LLM adaptive query was **not** run (no API key).

## What this does and does not support

The PBMC candidate pool, on this 64-gene episode-specific panel, produces almost no sequential information gain. That is recorded as limited interaction value. It must not be padded into a long-horizon task by forcing extra tool calls or longer prompts.

This is one public study. Folds that swap cell types are related, not independent biological tasks. Pretraining contamination of public Kang/PBMC IFN data is untested. Donor metadata is absent.

Tool-call counts are diagnostic only and are not rewarded.
