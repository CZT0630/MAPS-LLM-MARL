# MAPS Research Codebase

本项目正在将旧版“LLM + MADDPG 任务卸载”原型重构为可复现、可审计的新版 MAPS 实验系统。当前已完成 Phase 0–2：冻结旧版本、建立可信工程基线、修复技术可信的端边云环境。

**NOTE: Phase 2 由人工完成初稿，已由 Codex 于 2026-06-11 审查并修复。**

## 当前状态

- 旧代码、旧配置和原始 LaTeX 已冻结在 `legacy/`，并记录 SHA-256。
- 旧实验脚本、未完成的 MAAC 原型和合成曲线工具已从活动代码树删除。
- MADDPG 已改为真实 joint replay 与 centralized critic，不再复制单智能体动作。
- MADDPG、Legacy MAPS、MAPPO、HAPPO 可通过统一入口运行。
- 每次运行保存配置、seed、Git commit、运行环境、指标和模型。
- Gate 1 已通过，证据见 `artifacts/phase1/baseline_audit.json`。
- Gate 2 已通过，证据见 `artifacts/phase2/environment_audit.json`。
- Phase 2 使用独立的 `environment.physics_version: 2`，不再与结果格式版本混用。
- **C1 已完成**：策略/回放统一使用 `[partition_probs, edge_probs]`；MADDPG
  使用双 softmax，MAPPO/HAPPO 使用 softmax 均值的 simplex 分布与 categorical ES head。
- **C2 已完成**：mixed distillation 使用 partition squared L2 与 edge CE，
  支持 `maddpg`、`maps_no_annealing`、`maps` 三种模式及 environment-step 退火，
  并在 checkpoint/manifest 中保存调度和专家缓存版本。
- **C3 已完成**：使用 MiMo-V2.5 生成 500-state state-keyed expert cache；
  parser success、valid action rate 均为 100%，fallback rate 为 0%；LLM-only
  在配套冻结 scenario bank 上验证通过，部署期 `online_api_calls=0` 且 cache miss 为 0。
- **C4 已完成**：新增正式 evaluation 协议、drain horizon、逐任务记录、
  P95/DVR/TCR/energy/decision-latency 指标和 convergence AUC/threshold summary。
- **C5 已完成**：新增 EI scenario bank 生成、配置片段合成、train/evaluate/analyze
  入口和正式 artifact pipeline；已冻结 S1/S2/S3 train/test banks。

Phase 1 仅证明基线可运行、可复现，不代表论文方法或性能结论已经成立。Phase 2
修复了环境物理模型，但尚未用于正式论文实验。C1-C5 已完成，当前下一步是 C6
两 seed pilot gate。

## 快速开始

```powershell
pip install -r requirements.txt

# 查看入口
python main.py --help

# 单算法训练
python main.py --mode train --algorithm maddpg --config configs/smoke.yaml

# 全部基线 smoke run
python main.py --mode smoke --algorithm all --config configs/smoke.yaml

# Phase 1 正式审计
python main.py --mode audit `
  --config configs/smoke.yaml `
  --seeds 42,43 `
  --output-root results/phase1 `
  --audit-path artifacts/phase1/baseline_audit.json

# C4 evaluation smoke
python main.py --mode eval `
  --algorithm greedy_min_cost `
  --config configs/smoke_phase2.yaml `
  --episodes 1 `
  --steps 2 `
  --drain-steps 1 `
  --output-root artifacts/ei/c4_eval_smoke

# C5 EI scenario bank generation
python -m experiments.ei.scenario_bank `
  --base-config configs/ei/base.yaml `
  --fragment configs/ei/scales/s1_u10.yaml `
  --fragment configs/ei/deadlines/medium.yaml `
  --scenario-id s1_u10_medium `
  --output-dir artifacts/ei/scenario_banks

# 测试
python -m pytest
```

也支持从项目父目录运行：

```powershell
python -m LLM4RL.main --help
```

## 关键文件

- `docs/EI_SUBMISSION_MASTER_PLAN.md`：EI 代码、实验和论文修改的唯一执行方案。
- `docs/PHASE1_BASELINE_AUDIT.md`：Phase 0–1 完成记录和边界。
- `docs/PHASE2_IMPLEMENTATION.md`：Phase 2 实现记录和 Gate 2 检查清单。
- `docs/MIMO_API_SETUP.md`：MiMo API、`.env` 和本地验证方法。
- `legacy/PHASE0_MANIFEST.md`：冻结归档及校验值。
- `configs/smoke.yaml`：Phase 1 最小配置。
- `configs/smoke_phase2.yaml`：Phase 2 最小配置。
- `environment/channel_model.py`：Shannon 无线信道模型。
- `environment/backhaul_model.py`：有线回传链路模型。
- `environment/snapshot.py`：环境快照与纯函数式评估。
- `experiments/runner.py`：统一训练和审计入口。
- `experiments/ei/scenario_bank.py`：生成分离的 EI train/test scenario banks。
- `experiments/ei/train.py`：EI 配置片段 + train bank 训练入口。
- `experiments/ei/evaluate.py`：EI 配置片段 + test bank 评估入口。
- `experiments/ei/analyze.py`：汇总 `evaluation_metrics.json` 与 `task_records.csv`。
- `fixtures/legacy_expert_cache.json`：仅用于工程验证的固定专家缓存。
- `configs/ei/base.yaml`：EI 正式配置基底。
- `configs/ei/methods/`：正文方法配置片段。
- `configs/ei/scales/`：S1/S2 UE 规模配置片段。
- `configs/ei/deadlines/`：S3 deadline sensitivity 配置片段。
- `configs/ei/formal_s1.yaml`：C3 cache 和 C4 evaluation 的正式 S1 配置。
- `artifacts/ei/scenario_banks/`：C5 冻结 train/test scenario bank 产物。
- `llm_assistant/ei_prompt_builder.py`：EI-v1 prompt builder。
- `llm_assistant/expert_cache.py`：state-keyed expert cache 与 audit trail。
- `llm_assistant/cached_expert_provider.py`：cache hit / fallback provider。
- `baselines/llm_only.py`：LLM-only evaluator baseline。
- `scripts/generate_expert_cache.py`：从真实 API 生成 expert cache。
- `tests/test_c4_evaluation.py`：C4 drain horizon、task records 和 evaluation artifact 测试。
- `tests/test_c5_pipeline.py`：C5 scenario bank、配置合成和 artifact pipeline 测试。

## 投稿路线

- `master`：Phase 0-2 公共可信基线。
- `publication/ei-conference`：执行 `docs/EI_SUBMISSION_MASTER_PLAN.md`。

该主文档是 EI 路线的唯一方案，统一定义代码修改顺序、baseline、场景、指标、
图表和论文逐节修改要求。C1 混合动作 codec、C2 mixed distillation/退火、
C3 真实 MiMo cache、C4 正式评估协议和 C5 正式实验 pipeline 已完成。
完成这些工作前，`legacy_maps` 只用于工程 smoke test，不能作为论文结果。

MiMo API 的无密钥配置和本地验证方法见
`docs/MIMO_API_SETUP.md`。
