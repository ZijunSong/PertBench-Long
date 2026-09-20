# A01–A20 / B01–B11 修订状态

日期：2026-09-20。本文件记录每条审计要求对应的代码、测试命令、结果和**未验证**项。  
A 轮相对 `89d6961`；B 轮相对第二轮复审基线 `b7cfeac`。  
不要把“接口存在”“检测到 Docker”“无密钥时报错”写成“模型接入完成”。

测试命令：

```bash
/data/ppnm/miniconda3/envs/pertdiffbench/bin/python -m pytest tests/pertbench_long -q
```

本轮（B01–B11）在该环境结果：**56 passed, 3 skipped**（live 模型 ×2、Docker 隔离实跑 ×1 显式 skip）。未重跑真实 PBMC 全量 baseline，也未调用付费 API 或本机 GPU 服务，也未做真实容器边界验收。

---

## A01 · LLM 主循环 — 已实现，live 未验证

- 文件：`pertbench_long/agents/client.py`、`agents/loop.py`、`agents/llm.py`、`agents/protocol.py`、`runtime/tools.py`
- 行为：`ModelClient.generate(messages, tool_schemas)` + `AgentLoop` + `ToolRouter`。`llm_no_query` 在 router 层关闭 `request_experiment`。普通文本不等于 submit。
- 测试：`test_a03_native_tools_sent_and_cli_e2e`（CLI → mock HTTP → tools → submit → score）
- 结果：mock HTTP 闭环 **passed**
- 未验证：真实开源模型服务、真实 API、模型写出分析代码后的科学质量

## A02 · 可序列化工具 — 已实现（debug worker）

- 文件：`runtime/tools.py`、`runtime/broker.py`、`runtime/executor.py`、`agents/scripted.py`
- 行为：JSON schema 工具；`run_python` 由 runner 注入 executor；`inspect_artifact` 返回 shape/preview/container_path；购买串行
- 测试：E08/E09、scripted_mock 经 ToolRouter、A03 e2e 中 `run_python` 写 parquet
- 结果：passed
- 未验证：固定 digest 分析镜像内的 Docker worker 实跑（配方在 `docker/analysis.Dockerfile`）

## A03 · openai_compatible_chat transport — 已实现，live 未验证

- 文件：`agents/client.py`；配置 `configs/pertbench_long/local_openai_compatible.yaml`、`api_openai_compatible.yaml`
- 行为：`auth_mode none|bearer_env`；真正发送 tools；401/400 fail-fast；429/5xx 可重试；`json_action` 显式 fallback；usage 缺失记 unknown 不填 0
- 测试：e2e native tools；`test_a03_401_and_json_action`（401 + bearer + json_action）
- 结果：mock **passed**
- 未验证：真实 vLLM/API、429 后成功、超时、doctor-model 对真实服务。`test_v32_live_model_unverified` skip

## A04 · CLI 优先级与非零退出 — 已实现

- 文件：`cli.py`
- 行为：未指定参数为 `None`；`显式 CLI > YAML > 有文档默认`；`run --mode` 不再默认 debug；缺配置/坏 YAML 退出 2；非 completed 非零（3/4/5）
- 测试：`test_e13_yaml_mode_not_overridden_by_cli_default`、`test_e14_infra_error_nonzero_cli`、`test_missing_config_is_nonzero`
- 结果：passed
- 未验证：生产调度系统对接这些退出码

## A05 · 工具进程隔离 — 部分

- 文件：`runtime/executor.py`、`runtime/sandbox.py`、`docker/analysis.Dockerfile`
- 行为：`isolated_eval` 不再 `echo isolated_ok` 后回落到宿主执行未隔离 Python；无 Docker 则 `IsolationUnavailable`。debug worker 是宿主 subprocess，**永远 unqualified**
- 测试：E13（YAML isolated 不会变成 debug）；T09 Docker 可用性
- 结果：工程路径 passed
- 未验证：在真实分析镜像中执行“读 private/密钥/联网均失败、正常分析成功”。本环境未构建并钉死镜像 digest。`isolation_qualified` 不会仅因 Docker 探针为 true

## A06 · 先授权再揭示 — 已实现

- 文件：`oracle/ledger.py`、`oracle/service.py`
- 行为：`AUTHORIZED → MATERIALIZED → DELIVERED`；staging 在 private `_oracle_staging`；失败不写 evidence
- 测试：`test_e03_over_budget_does_not_reveal_file`、`test_e04_idempotency_conflict_does_not_reveal_new_experiment`、T06/T07 文件不可见
- 结果：passed

## A07 · 受信快照冻结 — 已实现

- 文件：`runtime/broker.py` `SnapshotIndex`；`runtime/runner.py` `score_run`
- 行为：评分只读 `trusted_snapshots/index.json`；workspace parquet 改写不影响分数；受信文件损坏 → `integrity_failure`
- 测试：E05、`test_trusted_snapshot_tamper_is_integrity_failure`、T16
- 结果：passed

## A08 · completed 必须有合法 final — 已实现

- 文件：`runtime/runner.py`、`evaluation/outcomes.py`
- 行为：仅 `SUBMITTED` + trusted receipt 为 completed；只 snapshot 不 submit → `invalid_submission`，官方分 0
- 测试：E06
- 结果：passed

## A09 · run/标签绑定 — 已实现

- 文件：`runtime/runner.py` `bind_public_private` / `score_run`；builder 相对 `labels.parquet`；`validate --check-artifacts`；未实现 protocol 拒绝
- 测试：E07 换 manifest → integrity_failure
- 结果：passed
- 未验证：整包目录迁移到另一台机器后的相对路径手测（代码按 manifest 目录解析）

## A10 · 禁止猜测处理历史后正式评分 — 已实现

- 文件：`data/adapter.py`、`episodes/builder.py`、`configs/pertbench_long/pbmc_pilot.yaml`（`matrix_kind: unknown`）
- 行为：未声明 → `unknown` + diagnostic track；声明 counts 不会被 max≤20 改成 log1p；隐藏值不决定变换
- 测试：E10/E11、T03
- 结果：passed
- 未验证：用户从原始预处理记录确认 PBMC 为 log1p 之前，**不得**把旧 0.8 分数当 official 生物学能力。旧 PBMC 表保留为 exploratory，本轮未重跑

## A11 · 基因对齐后再堆叠 — 已实现

- 文件：`data/adapter.py`、`tools/helpers.py`
- 测试：E12 不同列宽/重排列；冲突 fail-closed
- 结果：passed

## A12 · NaN 拒绝与 donor 传递 — 已实现

- 文件：`evaluation/labels.py`、`episodes/builder.py`、`episodes/export.py`
- 测试：E15 NaN 构建失败；无 donor 的 synthetic 仍走 cell-weighted
- 结果：passed
- 未验证：两个不均衡配对 donor 的独立 fixture（代码路径存在，本轮未加专用数据）

## A13 · artifact registry 闭环 — 已实现

- 文件：`runtime/broker.py`、`evaluation/submission.py`
- 测试：E08 购买后可引用 `ev_q_01`；E09 列表路径可 inspect/read
- 结果：passed

## A14 · 资源/超时 — 部分

- 文件：`runtime/executor.py`（每工具 timeout）、`runner.py`（episode deadline）、`agents/client.py`（token 累计，缺失不填 0）
- 测试：e2e 短超时配置可跑完
- 结果：工程约束存在
- 未验证：故意超内存/磁盘/孤儿进程的 OS 级杀掉；token budget 耗尽的真实模型

## A15 · 汇总未评分/分组 — 已实现

- 文件：`evaluation/outcomes.py`、`cli.py` summarize `--group-by`
- 测试：E16 completed 无分 → incomplete，不进 0 分母；T17 失败仍在科学分母
- 结果：passed
- JSON 禁止裸 NaN：`json_safe`

## A16 · resume/复现信息 — 部分

- 文件：`cli.py` `resume` / `trace-summary`；`run_manifest` 含 binding、resolved_config、model 非密钥字段；`replay` 不再声称 restore
- 测试：resume 有代码路径；未做 kill-after-purchase 注入
- 结果：命名诚实
- 未验证：购买后杀进程再 resume 只扣一次费的故障注入

## A17 · 安装与文档 — 部分

- 文件：`pyproject.toml` extras `eval`/`agent`/`dev`；`constraints/eval.txt`；README
- 行为：核心仍 numpy/pandas/yaml；eval 才保证 h5ad/parquet。旧 `pertdiffbench` 不再假装本仓库自带；缺失则 skip
- 测试：本环境 pytest passed；**未**在空白 venv 从 wheel 安装验收
- 未验证：全新机器干净安装

## A18 · 回归测试 — 已实现核心 E01–E16，B 轮已补异常路径

- 文件：`tests/pertbench_long/test_review_regressions.py`、`test_second_review.py`
- 结果：本环境 **56 passed / 3 skipped**（live 模型、Docker 实跑）
- 未验证：CI 分栏（CPU / mock-HTTP / Docker / live）尚未建 GitHub Actions

## A19 · suite 配置 — 接口已落地，未跑真实模型组

- 文件：`evaluation/suite.py`、`configs/pertbench_long/suite_example.yaml`、`cli.py suite`
- 未验证：两个真实模型 × no_query/adaptive × 多种子的合法成绩表。无 key 的 infra_error **不得**当性能数据

## A20 · pilot 边界 — 文档保留，结果未重跑

- 旧 PBMC `|ΔS|≲0.005` 表仍是未知处理历史上的 exploratory 记录，本轮修复后**没有**重新生成可信 official 表
- 不把 seed/prompt 当独立生物任务；不把转录预测称为机制发现
- 未实现湿实验、RL、多 Agent、新生物预测器

---

## B01 · FAILED 请求不可免费重放 — 已实现（mock 注入）

- 文件：`oracle/ledger.py` `authorize`/`fail_and_refund`/`mark_*`；`oracle/service.py`
- 根因：同 request_id 的 FAILED 行被当成 replay，staging 失败退款后可零费用交付
- 行为：FAILED 同 ID 永久 `FAILED_REQUEST`，须新 ID；`fail_and_refund` 幂等；禁止 FAILED→MATERIALIZED/DELIVERED
- 测试：`test_b01_failed_request_same_id_does_not_free_reveal`
- 未验证：真实进程在 mark_delivered 前被 SIGKILL 的 OS 级恢复

## B02 · resume 恢复原状态 — 已实现（工程路径）

- 文件：`runtime/runner.py` `_prepare`；`runtime/broker.py` `restore_from_ledger`；`cli.py` `_cmd_resume`
- 根因：resume 重写 manifest、evidence_version=0、已购未注册
- 行为：校验原 binding/config；从账本恢复 registry 与 version；不覆盖原 manifest；SUBMITTED 拒绝
- 测试：`test_b02_resume_restores_purchase_and_refuses_submitted`
- 未验证：真实 LLM 对话逐 token 恢复（明确不保证）

## B03 · isolation_qualified 不是类常量 — 已实现逻辑，Docker 实跑未验证

- 文件：`runtime/executor.py` `isolation_is_qualified`、`DockerPythonExecutor`
- 根因：`isolation_qualified = True` 写在类上，浮动 tag 即可合格
- 行为：实例默认 false；须 inspect digest 且匹配 `runtime.isolation_attestation.digest`；命名容器，超时 kill/rm
- 测试：`test_b03_isolation_qualified_requires_matching_digest`；`test_b11_docker_isolation_live_unverified` **skip**
- 未验证：真实 Docker 读 private/密钥/联网失败、超时后容器已停止、磁盘限额

## B04 · /workspace 路径契约与 gene 挂载 — 已实现（debug 映射）

- 文件：`runtime/broker.py` `map_agent_path`；executor 只读挂载 genes/contract
- 根因：提示用 `/workspace`，submit 走宿主 PathGuard 被拒；漏挂 gene universe
- 测试：`test_b04_workspace_virtual_path_is_accepted`；B08 用 `/workspace/outputs/...` 提交
- 未验证：真实容器内读 genes 再写 parquet 的 OS 级验收

## B05 · deadline/token/tool 限制在执行中生效 — 已实现（mock）

- 文件：`runtime/budget.py`；broker `get_budget` 计数；loop/client 在每次模型/工具前 check
- 根因：deadline 只在 agent 返回后检查；get_budget 不计工具次数；LLMAdapter 未传 deadline
- 测试：`test_b05_deadline_stops_get_budget`
- 未验证：无限 Python 在真实 Docker 中被 kill；provider 缺 usage 时的严格 token cap 实跑

## B06 · resolved config 装配到 loop/client — 已实现

- 文件：`runtime/config.py` `first_defined`/`resolve_run_config`；cli/suite/llm/client
- 根因：`or 60` 吞掉 0；嵌套 model 时丢顶层 seed；suite 不走同一解析函数
- 测试：`test_b06_seed_and_max_steps_reach_client_and_loop`
- 未验证：真实服务忽略 seed 时的 sampling 固定（记录 seed_status=unsupported/sent）

## B07 · TransportError 不计科学零分 — 已实现

- 文件：`runtime/runner.py` 单独捕获 `TransportError`；`score_run` 对 infra 保持 null
- 根因：401 落入 `PertBenchLongError` → `agent_tool_failure` → direction_score=0
- 测试：`test_b07_http_401_is_infra_not_science_zero`
- 未验证：真实 429 后成功、5xx 耗尽的付费 API

## B08 · 公开提交契约与 tool result 补齐 — 已实现（mock）

- 文件：`evaluation/contract.py`；builder 写入 `submission_contract.json`；loop/tools/doctor-model
- 根因：prompt 无列约束；json_action 只有名字；畸形 native 参数不回 tool result
- 测试：`test_b08_malformed_native_args_still_return_tool_result`
- 未验证：真实模型只靠公开契约、无测试硬编码 parquet 的首次合法提交

## B09 · 白名单 + 绑定 hash 预检 — 已实现

- 文件：`runtime/release.py` `check_release`/`copy_declared_public_inputs`；validate 与 run 共用
- 根因：`glob('*.h5ad')` 复制未声明 Q；validate 不比对绑定 hash
- 测试：`test_b09_extra_and_tampered_public_artifacts_rejected`
- 未验证：用户手改 private.public 与 public 两边 hash 后另立 episode 身份（那是新包，不是绕过）

## B10 · suite 分组保留轨道/预算/失败归属 — 已实现

- 文件：`evaluation/suite.py`；`cli.py` summarize；outcomes 默认 group_by 含 track/synthetic/budget
- 测试：`test_b10_suite_does_not_pool_incompatible_tracks`
- 未验证：真实两模型 × 两 policy × 多种子付费矩阵

## B11 · 真实接入验收 — 未完成，测试诚实 skip

- 文件：`tests/pertbench_long/test_second_review.py` skip；本文未验证表
- 去掉 E16 永真断言
- 未验证：干净 venv 安装、真实本地 openai-compatible 全 episode、真实远端 API、PBMC official

---

## 明确未完成、因此不能写“可直接测评”

| 项 | 状态 |
|---|---|
| 真实本地开源模型闭环 | **未验证** |
| 真实 API 闭环 | **未验证** |
| Docker 分析镜像 digest + 隔离验收 | **未验证** |
| PBMC official（处理历史已确认）重跑 | **未做** |
| 空白环境 wheel 安装 | **未做** |

达到审计第 5 节“可直接测评”还需要：用户启动模型服务 → `doctor-model` → isolated 分析镜像可用 → 对绑定 episode 跑 `run`/`score`/`summarize`。当前仓库提供了这条链路的代码，但没有在本机完成 live gate。
