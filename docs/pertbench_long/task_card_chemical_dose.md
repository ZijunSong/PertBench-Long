# Task card: chemical_dose_acquisition_v1

- Purpose: choose dose evidence under a unit-cost budget and predict held-out dose conditions.
- Data: sci-Plex3 when a user-supplied directory is provided; otherwise synthetic fixtures only.
- Real/synthetic: synthetic fixtures are marked `synthetic=true` and `readiness=pilot`.
- Intervention: chemical / dose / cell-line context.
- Condition identity: `cid_v1` (molar-normalized dose; 1 uM = 1000 nM).
- O/Q/T/C: lowest or support doses in O, purchasable other doses in Q, held-out doses in T, matched vehicle in C.
- Budget: B=8 or 16, cost=1 per bundle.
- Labels: `lognorm_cellmean_delta_v1` with a named estimand (`replicate_equal_weight` or `cell_weighted_descriptive`). Scoring: `continuous_effect_aubc_v1`. `a_family=0.5` is a fixture/pilot default, not a completed calibration.
- Split variants: `dose_interpolation`, `dose_extrapolation`.
- External knowledge: no hidden target responses; no claim of clinical efficacy.
- Known limits: first-party full sci-Plex rebuild is not shipped; scale is audited and may fail closed.
