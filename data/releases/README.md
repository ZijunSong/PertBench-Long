# PertBench-Long data release

This folder ships the **PBMC IFN pilot source tables** used by `build-episodes --fixture pbmc`.

| Artifact | Role |
|---|---|
| `kang_pbmc_ifn_public.tar.gz` | 14 `task1_{train,valid}_{celltype}_exp.csv` files (~15 MiB compressed, ~303 MiB extracted) |
| `kang_pbmc_ifn_public.sha256` | SHA-256 of each CSV inside the archive |
| `kang_pbmc_ifn_public/` | Local extract (gitignored). Created by `pertbench-long unpack-data` |

```bash
# from the repository root
pertbench-long unpack-data
# or
tar -xzf data/releases/kang_pbmc_ifn_public.tar.gz -C data/releases
sha256sum -c data/releases/kang_pbmc_ifn_public.sha256
```

Built episode packs (public evidence + private labels) are in `data/episodes/`. Synthetic smoke does not need this archive.

**Not included:** MOA (~8 GiB) and temporal / fig1 tables. Those protocols are unsupported or not required for the current PBMC diagnostic pilot.

Source study: public Kang PBMC IFN scRNA-seq tables redistributed by PertDiffBench. Processing history remains **unknown**; `matrix_kind` stays `unknown` and scoring stays **diagnostic**.
