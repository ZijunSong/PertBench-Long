# Protocol (v1)

Retrospective experimental replay. The Agent sees O and C, may buy items from Q with integer credits, and is scored on T that is never sold.

- `O ∩ Q = O ∩ T = Q ∩ T = ∅` at observation IDs and condition keys.
- Shared controls are listed as `C`. They are not purchased IFN results.
- v1 items are existing condition bundles, price 1 credit. No control-fraction shopping. No generative stand-in for missing experiments (`UNAVAILABLE_EXPERIMENT`).
- A frozen prediction snapshot for the current evidence version is required before a new purchase. Snapshots are format-checked only; hidden scores are not returned.
- After submit, purchases and final edits are rejected. Scores are computed by a trusted process outside the Agent workspace.
- Main metric: `direction_score` = 1 - (multiclass Brier)/2 on the full target×gene grid. Optional DE recovery is not reported without `replicate_de_v1`.
- `cpu_pilot_v1` configured limits: 2 CPU, 8 GiB, 1 GiB workspace, 60 s/tool, 600 s/episode, 80 external tool calls. Enforcement is applied to broker tool counts and Docker probe flags; in-process debug is not a hard OS jail.
