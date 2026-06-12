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

Phase 1 仅证明基线可运行、可复现，不代表论文方法或性能结论已经成立。Phase 2 修复了环境物理模型，但尚未用于正式论文实验。

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

# 测试
python -m pytest
```

也支持从项目父目录运行：

```powershell
python -m LLM4RL.main --help
```

## 关键文件

- `docs/MAPS_REDEVELOPMENT_PLAN.md`：完整研究与实现路线。
- `docs/PHASE1_BASELINE_AUDIT.md`：Phase 0–1 完成记录和边界。
- `docs/PHASE2_IMPLEMENTATION.md`：Phase 2 实现记录和 Gate 2 检查清单。
- `legacy/PHASE0_MANIFEST.md`：冻结归档及校验值。
- `configs/smoke.yaml`：Phase 1 最小配置。
- `configs/smoke_phase2.yaml`：Phase 2 最小配置。
- `environment/channel_model.py`：Shannon 无线信道模型。
- `environment/backhaul_model.py`：有线回传链路模型。
- `environment/snapshot.py`：环境快照与纯函数式评估。
- `experiments/runner.py`：统一训练和审计入口。
- `fixtures/legacy_expert_cache.json`：仅用于工程验证的固定专家缓存。

## 投稿路线

- `master`：Phase 0-2 公共可信基线。
- `publication/letter-revision`：执行 `docs/MAPS_REDEVELOPMENT_PLAN.md` 中的完整 Letter 增强路线。
- `publication/ei-conference`：执行 `docs/EI_CONFERENCE_REVISION_PLAN.md` 中的低成本 EI 会议修改路线。

当前 EI 分支的下一步是修复混合动作表示、实现 mixed distillation 与退火消融，并建立正式论文指标和实验配置。完成这些工作前，`legacy_maps` 只用于工程 smoke test，不能作为论文结果。
