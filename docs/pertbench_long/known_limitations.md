# Known limitations

- Public Kang/PBMC IFN data may already be in model pretraining corpora. Hiding labels here only blocks this episode, not prior exposure.
- A single public study cannot support a private independent test. Cell-type folds are related.
- Donor metadata is absent in the local CSVs. Donor is recorded as null. Cells are not biological replicates. `replicate_de_v1` is refused.
- Matrices are classified as `log1p` from value range, not from a recovered counts pipeline. `effect_proxy_v1` does not undo unknown scaling. Neutral is `|delta|<=0.10` log1p, not significant DE and not biological no-change.
- Official scoring used an episode-specific 64-gene panel chosen from O+C. It is not a frozen pan-study gene universe.
- Query catalogue is three IFN condition bundles, budget 2. That is a short sequential-reveal pilot. On the PBMC panel, S(B)-S(0) was near zero (random query slightly negative). That is limited interaction value, not a solved long-horizon discovery task.
- Correct predictions do not identify a unique mechanism, individual-level counterfactuals, or a wet-lab finding. This is retrospective replay of already-collected measurements.
- Isolated evaluation: Docker probe with `ubuntu:22.04 --network=none` can hide the private manifest from a container. The v0.1 CPU agent loop still talks to the host broker. `isolation_qualified` is false. Debug mode has no leakage guarantee.
- LLM adapter is implemented. No live model call was made (no endpoint/key). Status: adapter implemented, live run unverified. External model draws are non-deterministic.
- MOA filenames can leak the class to predict. MOA official track is unsupported. Temporal file is `lognorm_scale_hvg3000`; official proxy labels are blocked.
- Old PertDiffBench HVG/combined h5ad files are not leak-free Agent inputs.
