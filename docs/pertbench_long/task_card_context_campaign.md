# Task card: context_campaign_acquisition_v1

- Purpose: allocate a shared credit budget across cell contexts and predict held-out conditions.
- Data: sci-Plex3 when a prepared directory is provided; otherwise synthetic fixtures only.
- Real/synthetic: synthetic fixtures are marked `synthetic=true` and `readiness=pilot`.
- Intervention: chemical / dose / multiple cell-line contexts.
- Condition identity: `cid_v1`.
- O/Q/T/C: richer source-context evidence in O, purchasable remaining conditions in Q, held-out target-context conditions in T, matched vehicle/time controls in C.
- Budget: B=16 default, cost=1 per bundle. Scores average equally across contexts.
- Labels: `lognorm_cellmean_delta_v1` with a named estimand (`replicate_equal_weight` or `cell_weighted_descriptive`).
- Split variant: `few_shot_target_context` (not zero-shot).
- Status: `tested_synthetic`. Real sci-Plex prepare/build: depends on user-supplied author files. Live model: `not_run`.
