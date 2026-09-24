# PertBench-Long

回顾式实验回放（retrospective experimental replay）环境：Agent 只能看到部分真实扰动证据，可购买有限的额外条件包，最后对从未向其揭示的干预结果评分。

这不是 PertDiffBench 的改名，也不是湿实验执行器或任意干预模拟器。旧的扩散模型评测仍用单独安装的 `pertdiffbench`。

**当前状态：工程测评链路已按 2026-09-20 审计修订，但尚未完成 live 模型 / 隔离镜像 / 确认处理历史的 official PBMC 重跑。不能仅填写 endpoint 或 API key 就声称模型接入完成。** 详见 [review_fix_status.md](docs/pertbench_long/review_fix_status.md)。

## 最短启动（synthetic / CPU mock）

```bash
python -m pip install -c constraints/eval.txt -e '.[eval,agent,dev]'
pertbench-long doctor
pertbench-long list
pertbench-long smoke --fixture synthetic --agent scripted_mock --mode local_trusted_debug
```

`local_trusted_debug` 不是隔离合格结果。synthetic 分数不可当科学结论。

## 长程新任务（工程闭环，合成 fixture）

T1 药物剂量、T2 双基因组合、T3 跨背景预算已有独立协议与构建器。仓库不附带完整 sci-Plex/Norman 矩阵；没有用户提供的数据目录时，只能跑合成 fixture：

```bash
pertbench-long list
pertbench-long build-episodes --fixture synthetic_dose --output /tmp/pb_long
pertbench-long build-episodes --fixture synthetic_pair --output /tmp/pb_long
```

真实源数据必须走 `fetch-data` → `prepare-data` → `doctor-data` → `build-episodes`（YAML `dataset`/`adapter`/`protocol`，无需把真实数据叫 fixture）。缺少源文件或未准备时非零退出，**不会**退回合成数据。当前 GEO 文件 URL/hash 未锁定：`fetch-data` 会列出必须离线放入 `raw/<dataset>/<release>/` 的作者格式文件（sci-Plex: `cells.csv`+`counts.csv`；Norman: `cell_identities.csv`+`counts.csv`），不要把二次处理 h5ad 当成 raw counts。MOA / temporal 正式 profile 仍为 blocked。旧 PBMC `effect_proxy_v1` 分数定义未改。

```bash
export PERTBENCH_DATA_ROOT=/absolute/path/to/pertbench_data
pertbench-long fetch-data --dataset sciplex3 --data-root "$PERTBENCH_DATA_ROOT"
pertbench-long prepare-data --dataset sciplex3 --data-root "$PERTBENCH_DATA_ROOT"
pertbench-long doctor-data --config configs/pertbench_long/sciplex_dose_pilot.yaml
pertbench-long build-episodes --config configs/pertbench_long/sciplex_dose_pilot.yaml
```

## 模型测评（配置存在 ≠ live 完成）

```bash
# 不下载权重、不启动 GPU。endpoint 可达 ≠ 工具协议可用。
pertbench-long doctor-model --config configs/pertbench_long/local_openai_compatible.yaml

# 需要：已启动的 openai-compatible 服务、声明的 served model name、
# isolated_eval 下可用的分析镜像 digest。缺任一项都会非零退出，而不是假成功。
pertbench-long run --config configs/pertbench_long/local_openai_compatible.yaml
pertbench-long score --run RUN_DIR --private-manifest data/episodes/synthetic_pilot_001/private/manifest.json
pertbench-long summarize --runs runs/eval --group-by model,policy,budget,isolation_qualified,scoring_track --output runs/eval/summary.json
```

`auth_mode: none` 用于本机无鉴权服务；`bearer_env` 只读配置里的 `api_key_env`，不会自动借用 `OPENAI_API_KEY`。  
`llm_no_query` 在工具层禁止购买，不是只靠 prompt。

本仓库**没有**在作者环境完成真实开源模型或付费 API 的 episode 闭环。那一项保持未验证。

## 数据

仓库自带：

- `data/releases/kang_pbmc_ifn_public.tar.gz`：PBMC IFN 14 张源表（约 15 MiB；解压后约 303 MiB）
- `data/episodes/synthetic_pilot_001` 与 `data/episodes/pbmc_pilot_001`：已构建的 public/private 包

```bash
pertbench-long unpack-data
pertbench-long inspect-data --config configs/pertbench_long/pbmc_data.yaml
```

未打包 MOA（约 8 GiB）和 temporal/fig1。synthetic 不需要这套 CSV。

## 真实 PBMC

处理历史在 dataset card 中为 **unknown**。`configs/pertbench_long/pbmc_pilot.yaml` 声明 `matrix_kind: unknown`，构建结果走 **diagnostic** 轨道，不能与 official 分数表合并。

```bash
pertbench-long inspect-data --config configs/pertbench_long/pbmc_data.yaml
pertbench-long build-episodes --config configs/pertbench_long/pbmc_pilot.yaml --fixture pbmc --output data/episodes
pertbench-long validate --manifest data/episodes/pbmc_pilot_001/private/manifest.json --check-artifacts
```

2026-09-20 的 CPU baseline 表（O=CD4T/CD8T IFN，Q=三个髓系 IFN，T=B/NK IFN，B=2，64 个 O+C 基因）是修复前、按数值范围当作 log1p 的 exploratory 记录，**本轮未重跑，不能当 official 生物学能力**：

| 策略 | direction_score | S(B)-S(0) |
|---|---:|---:|
| no_change | 0.406 | 0 |
| mean_delta_no_query | 0.800 | 0 |
| random_query | 0.796 | −0.0039 |
| fixed_order_query | 0.802 | +0.0019 |
| control_similarity_query | 0.805 | +0.0047 |

查询收益接近零。见 `docs/pertbench_long/interaction_value_report.md`。确认原始预处理并声明 `matrix_kind` 之后才能进入 official `effect_proxy_v1`。

## 隔离

| mode | 含义 |
|---|---|
| `local_trusted_debug` | 宿主 subprocess 跑分析代码。永远 `isolation_qualified=false`。 |
| `isolated_eval` | 要求 Docker 分析镜像；无 Docker / 无镜像则拒绝，不回落到 debug 还声称隔离。模型 client 可留在宿主，**模型生成的 Python 必须进容器**。 |

配方：`docker/analysis.Dockerfile`。`isolation_qualified` 只有在镜像 digest 与 `runtime.isolation_attestation.digest` 一致时才为 true，**不是** executor 类常量。本环境未做真实容器边界验收。

`resume` 需要 `run_progress.json`（LLM 还要 `agent_messages.json`），恢复 token/工具次数/剩余 wall time 与已购证据，并拒绝更换 model/seed/runtime 或已 SUBMITTED 的 run。暂停不 refill 完整 timeout。预实验请用全新 run，不要用 resume 覆盖失败结果。

正式表只收 `official_eligible=true`（隔离合格 ∧ official track ∧ 非 synthetic）。`workspace_gib` 当前未执行磁盘配额，设置后不能取得隔离资格。复制镜像 digest **不是**隔离验收。

HTTP 401/403/连接失败记为 `infra_error`，科学分为 null，不进模型零分。

## 文档

- [review_fix_status.md](docs/pertbench_long/review_fix_status.md) — A01–A20、B01–B11 与 C01–C05 逐条状态
- [task_card_chemical_dose.md](docs/pertbench_long/task_card_chemical_dose.md) / [task_card_genetic_pair.md](docs/pertbench_long/task_card_genetic_pair.md) / [task_card_context_campaign.md](docs/pertbench_long/task_card_context_campaign.md)
- [implementation_status.md](docs/pertbench_long/implementation_status.md)
- [dataset_card_pbmc.md](docs/pertbench_long/dataset_card_pbmc.md)
- [known_limitations.md](docs/pertbench_long/known_limitations.md)
- [isolation.md](docs/pertbench_long/isolation.md)
- [migration.md](docs/pertbench_long/migration.md) — 旧 PertDiffBench 需单独安装

## 测试

```bash
python -m pytest tests/pertbench_long -q
```

缺少 Docker、真实模型和科学数据时，对应测试会 skip 并写明原因，而不是 `return` 冒充通过。

## 旧入口

新包装好后**不包含** `pertdiffbench` CLI。若本机另有旧环境，可单独调用；没有则不能声称旧 CLI 可用。
