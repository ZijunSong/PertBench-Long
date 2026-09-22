# Dataset card: sci-Plex3 (planned)

- Accession: GSM4150378
- Adapter: `pertbench_long.data.adapters.sciplex.import_sciplex`
- Acquisition manifest: `data/acquisition/sciplex3_v1.json`
- Input: user-supplied directory only. `/data/ppnm` is never implied.
- Matrix kind: declared by the caller. A third-party h5ad is not labeled raw counts unless declared.
- Identity: `cid_v1` over context, compound, molar-normalized dose, time, assay.
- Status: adapter and synthetic fixture exist; full source rebuild is **not_run** in this environment.
