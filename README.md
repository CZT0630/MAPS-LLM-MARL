# MAPS Research Codebase

本项目正在将旧版“LLM + MADDPG 任务卸载”原型重构为可复现、可审计的新版 MAPS 实验系统。当前已完成 Phase 0–1：冻结旧版本并建立可信工程基线。

## 当前状态

- 旧代码、旧配置和原始 LaTeX 已冻结在 `legacy/`，并记录 SHA-256。
- 旧实验脚本、未完成的 MAAC 原型和合成曲线工具已从活动代码树删除。
- MADDPG 已改为真实 joint replay 与 centralized critic，不再复制单智能体动作。
- MADDPG、Legacy MAPS、MAPPO、HAPPO 可通过统一入口运行。
- 每次运行保存配置、seed、Git commit、运行环境、指标和模型。
- Gate 1 已通过，证据见 `artifacts/phase1/baseline_audit.json`。

Phase 1 仅证明基线可运行、可复现，不代表论文方法或性能结论已经成立。环境仍采用旧版简化物理模型；固定专家缓存也是工程测试夹具，不能用作论文实验数据。

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
- `legacy/PHASE0_MANIFEST.md`：冻结归档及校验值。
- `configs/smoke.yaml`：Gate 1 的最小配置。
- `experiments/runner.py`：统一训练和审计入口。
- `fixtures/legacy_expert_cache.json`：仅用于工程验证的固定专家缓存。

## 下一阶段

Phase 2 将修复队列演化、时隙、无线速率、UE→ES 与 ES→CS 能耗、下行时延及资源约束。完成 Phase 2 前，不应开展论文主实验或引用当前 smoke 指标作为方法优势。
