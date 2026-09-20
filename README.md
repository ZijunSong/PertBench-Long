# PertBench-Long

回顾式实验回放（retrospective experimental replay）环境：Agent 只能看到部分真实扰动证据，可购买有限的额外条件包，最后对从未向其揭示的干预结果评分。

这不是 PertDiffBench 的改名，也不是湿实验执行器或任意干预模拟器。旧的扩散模型评测仍用 `pertdiffbench`。

**v0.1 = 工程核心 + 公开 PBMC pilot。** 不据此宣称已建成长期科学发现 benchmark，也不宣称优于已有评测。

## 最短启动

```bash
conda activate pertdiffbench
cd /data/ppnm/PertBench-Long
pip install -e '.[dev]'

pertbench-long list
pertbench-long smoke --fixture synthetic --agent scripted_mock --mode local_trusted_debug
```

实测：synthetic smoke `direction_score≈0.88`（**synthetic，不可当科学结论**）。

## 真实 PBMC（本地 CSV 可用时）

```bash
pertbench-long inspect-data --config configs/pertbench_long/pbmc_data.yaml
pertbench-long build-episodes --config configs/pertbench_long/pbmc_pilot.yaml --fixture pbmc --output runs/episodes
pertbench-long validate --manifest runs/episodes/pbmc_pilot_001/private/manifest.json

pertbench-long run \
  --public-dir runs/episodes/pbmc_pilot_001/public \
  --private-manifest runs/episodes/pbmc_pilot_001/private/manifest.json \
  --agent random_query \
  --mode local_trusted_debug \
  --output runs/pbmc_baselines/random_query

pertbench-long score \
  --run runs/pbmc_baselines/random_query/run_* \
  --private-manifest runs/episodes/pbmc_pilot_001/private/manifest.json
```

2026-09-20 本地 PBMC pilot（O=CD4T/CD8T IFN，Q=三个髓系 IFN，T=B/NK IFN，B=2，64 个 O+C 基因）：

| 策略 | direction_score | S(B)-S(0) |
|---|---:|---:|
| no_change | 0.406 | 0 |
| mean_delta_no_query | 0.800 | 0 |
| random_query | 0.796 | −0.0039 |
| fixed_order_query | 0.802 | +0.0019 |
| control_similarity_query | 0.805 | +0.0047 |

查询收益接近零，且 random 为负（未截断）。见 `docs/pertbench_long/interaction_value_report.md`。

## 隔离

```bash
pertbench-long smoke --fixture synthetic --agent scripted_mock --mode isolated_eval
```

Docker 探针对 `ubuntu:22.04 --network=none` 可通过；Agent 主循环仍在宿主 broker 上。`isolation_qualified=false`。debug 模式没有防泄漏保证。

LLM：`configs/pertbench_long/llm_agent.yaml`。无密钥时不会报 live success（adapter implemented, live run unverified）。

## 文档

- [implementation_status.md](docs/pertbench_long/implementation_status.md) — R01–R24
- [repository_audit.md](docs/pertbench_long/repository_audit.md)
- [dataset_card_pbmc.md](docs/pertbench_long/dataset_card_pbmc.md)
- [known_limitations.md](docs/pertbench_long/known_limitations.md)
- [migration.md](docs/pertbench_long/migration.md) — 如何继续复现旧 PertDiffBench
- [isolation.md](docs/pertbench_long/isolation.md)
- [protocol.md](docs/pertbench_long/protocol.md)

## 测试

```bash
python -m pytest tests/pertbench_long -q
```

## 旧入口

```bash
pertdiffbench list-tasks
```
