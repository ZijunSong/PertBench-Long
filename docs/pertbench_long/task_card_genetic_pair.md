# Task card: genetic_pair_acquisition_v1

- Purpose: use single-gene and other pair evidence to predict held-out two-gene CRISPRa responses.
- Data: Norman 2019 when a user-supplied directory is provided; otherwise synthetic fixtures only.
- Intervention: CRISPRa; A+B and B+A are one unordered condition.
- Split variant (v1): `pair_holdout_seen_genes` — every target gene has support in O∪Q. This is not unseen-gene generalization.
- Labels: `lognorm_cellmean_delta_v1`. Interaction residual is optional and NA when a single-gene reference is missing.
- Do not synthesize official pair truth with an additivity model.
