# Dataset card: Norman 2019 CRISPRa (planned)

- Accession: GSE133344
- Author entry: https://github.com/thomasmaxwellnorman/Perturbseq_GI
- Fetch: `pertbench-long fetch-data --dataset norman2019 --data-root $PERTBENCH_DATA_ROOT`
- Prepare: author-format `cell_identities.csv` + `counts.csv|counts.h5ad` (+ optional `guides.csv`) → `prepared/norman2019/v1/`
- Adapter: `pertbench_long.data.adapters.norman.import_norman`
- Acquisition manifest: `data/acquisition/norman2019_v1.json` (raw URLs are null; they are not guessed)
- Intervention: CRISPRa, not a genome-wide knockout screen.
- Pair identity: unordered `A+B` == `B+A`. `gene+NTC` is a single-gene condition; NTC is not a second intervened gene.
- Missing combinations are not filled with additivity models.
- Source lock: **blocked**. No publisher checksum is recorded. CSV and Cell Ranger MEX (`matrix.mtx`, `barcodes.tsv`, `genes.tsv`, `cell_identities.csv`) are interchange profiles. `guides.csv` is optional unless the profile says the identity table needs it.
- Status: fetch/prepare/adapter implemented and tested on interchange fixtures. Full GEO download is **not_run**.
