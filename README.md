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

## 模型测评（配置存在 ≠ live 完成）

```bash
# 不下载权重、不启动 GPU。endpoint 可达 ≠ 工具协议可用。
pertbench-long doctor-model --config configs/pertbench_long/local_openai_compatible.yaml

# 需要：已启动的 openai-compatible 服务、声明的 served model name、
# isolated_eval 下可用的分析镜像 digest。缺任一项都会非零退出，而不是假成功。
pertbench-long run --config configs/pertbench_long/local_openai_compatible.yaml
pertbench-long score --run RUN_DIR --private-manifest runs/episodes/synthetic_pilot_001/private/manifest.json
pertbench-long summarize --runs runs/eval --group-by model,policy,budget,isolation_qualified,scoring_track --output runs/eval/summary.json
```

`auth_mode: none` 用于本机无鉴权服务；`bearer_env` 只读配置里的 `api_key_env`，不会自动借用 `OPENAI_API_KEY`。  
`llm_no_query` 在工具层禁止购买，不是只靠 prompt。

本仓库**没有**在作者环境完成真实开源模型或付费 API 的 episode 闭环。那一项保持未验证。

## 真实 PBMC

处理历史在 dataset card 中为 **unknown**。`configs/pertbench_long/pbmc_pilot.yaml` 声明 `matrix_kind: unknown`，构建结果走 **diagnostic** 轨道，不能与 official 分数表合并。

```bash
pertbench-long inspect-data --config configs/pertbench_long/pbmc_data.yaml
pertbench-long build-episodes --config configs/pertbench_long/pbmc_pilot.yaml --fixture pbmc --output runs/episodes
pertbench-long validate --manifest runs/episodes/pbmc_pilot_001/private/manifest.json --check-artifacts
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

`resume` 会从账本恢复已购证据与 evidence_version，并拒绝已 SUBMITTED 的 run；不会覆盖原始 `run_manifest.json`。

HTTP 401/403/连接失败记为 `infra_error`，科学分为 null，不进模型零分。

## 文档

- [review_fix_status.md](docs/pertbench_long/review_fix_status.md) — A01–A20 与 B01–B11 逐条状态
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
