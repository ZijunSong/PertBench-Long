# Dataset card: sci-Plex3 (planned)

- Accession: GSM4150378
- Author entry: https://github.com/cole-trapnell-lab/sci-plex
- Fetch: `pertbench-long fetch-data --dataset sciplex3 --data-root $PERTBENCH_DATA_ROOT`
- Prepare: author-format `cells.csv` + `counts.csv|counts.h5ad` → `prepared/sciplex3/v1/{matrix.h5ad,provenance.json,conditions.parquet,qc_report.json}`
- Adapter: `pertbench_long.data.adapters.sciplex.import_sciplex` (requires declared `matrix_kind` and an explicit layer when the h5ad has layers)
- Acquisition manifest: `data/acquisition/sciplex3_v1.json` (raw URLs are null; they are not guessed)
- Input: user-supplied directory only. `/data/ppnm` is never implied.
- Matrix kind: declared by the caller. A third-party h5ad is not labeled raw counts unless declared.
- Identity: `cid_v1` over context, compound, molar-normalized dose, time, assay, and vehicle/reference type.
- Source lock: **blocked**. No publisher checksum is recorded. `cells.csv` / `counts.csv` are an interchange profile, not verified GEO filenames.
- Status: fetch/prepare/adapter implemented and tested on interchange fixtures. Full GEO download is **not_run**.

