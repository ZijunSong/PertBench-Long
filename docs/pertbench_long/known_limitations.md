# Known limitations

- Public Kang/PBMC IFN data may already be in model pretraining corpora. Hiding labels here only blocks this episode, not prior exposure.
- A single public study cannot support a private independent test. Cell-type folds are related.
- Donor metadata is absent in the local CSVs. Donor is recorded as null. Cells are not biological replicates. `replicate_de_v1` is refused.
- Matrices **must not** be classified as `log1p` from `max≤20` for official scoring. Undeclared imports are `unknown` / diagnostic. Neutral is `|delta|<=0.10` in the declared effect unit, not significant DE and not biological no-change.
- Official scoring, when processing is declared, uses an episode-specific gene panel from O+C. It is not a frozen pan-study gene universe.
- Query catalogue is three IFN condition bundles, budget 2. That is a short sequential-reveal pilot. Historical PBMC `|ΔS|≲0.005` numbers are exploratory and were not re-run after the 2026-09-20 audit fixes.
- Correct predictions do not identify a unique mechanism, individual-level counterfactuals, or a wet-lab finding.
- Isolated evaluation: debug is never qualified. `isolated_eval` refuses without Docker/analysis image. A Docker *probe* is not a passing isolation result. Image-digest acceptance is **unverified**.
- LLM loop and openai-compatible transport are implemented. Mock HTTP e2e is tested. **Live local model and live API episode eval are unverified.** Filling an endpoint does not complete model evaluation.
- External model draws are non-deterministic. Resume does not claim bitwise LLM reproduction. `replay` / `trace-summary` only summarize events.jsonl.
- MOA filenames can leak the class to predict. MOA official track is unsupported. Temporal file is `lognorm_scale_hvg3000`; official proxy labels are blocked.
- Old PertDiffBench HVG/combined h5ad files are not leak-free Agent inputs.
