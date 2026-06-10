# Phase 0–1 Baseline Audit

完成日期：2026-06-11

## 结论

Phase 0–1 已完成，Gate 1 通过。当前代码具备继续开展 Phase 2 系统模型修复的工程基础，但当前 smoke 结果不能作为论文性能证据。

## Phase 0：冻结与清理

- 在 `legacy/LLM4RL_phase0_legacy_20260611.zip` 冻结清理前的完整项目。
- 保存原始 LaTeX 源文件和旧配置，SHA-256 见 `legacy/PHASE0_MANIFEST.md`。
- 初始化 Git，Phase 0–1 基线提交为 `f1ae8ea`。
- 删除活动树中的废弃实验脚本、未完成 MAAC、合成曲线工具、字节码缓存和生成报告。
- 新旧实验通过 `model_version`、独立目录和 run manifest 隔离。

## Phase 1：可信基线修复

- 统一 `python main.py` 与 `python -m LLM4RL.main` 入口。
- MADDPG transition 保存完整多智能体状态、动作、奖励和下一状态。
- centralized critic 使用真实 joint state/action。
- target action 由每个对应智能体的 target actor 生成。
- edge 数量由配置驱动，不再硬编码为 5。
- Legacy MAPS 使用固定缓存专家和显式解析器回退路径。
- 每次运行保存配置哈希、seed、Git commit、系统环境、逐回合指标、loss 和 checkpoint。

## Gate 1 证据

正式审计使用 `configs/smoke.yaml`：

- 算法：MADDPG、Legacy MAPS、MAPPO、HAPPO；
- seeds：42、43；
- 每个组合：20 episodes × 10 steps；
- 共 8 个正式 smoke runs；
- 所有运行成功，聚合指标均为有限值；
- joint replay、centralized critic 和固定专家缓存检查通过；
- MADDPG 同 seed 双跑的逐回合指标与训练 loss 完全一致，最大绝对差为 0。

机器可读审计：`artifacts/phase1/baseline_audit.json`。

自动测试：11 项通过，覆盖 replay shape、对应 target actor、动态 edge 数量、联合更新、LLM parser/cache、seed 和真实短程训练。

## 明确边界

- 固定专家缓存是合成工程夹具，`paper_evidence=false`。
- 环境仍是旧版简化模型，尚未修复标准队列演化和链路能耗。
- MAPPO/HAPPO 暂时沿用连续 edge selector，Phase 3 需要改为混合动作分布。
- 当前 completion rate 等数值只用于发现 NaN、shape error 和空输出，不用于比较算法优劣。

下一步严格进入 Phase 2，先修系统模型，再开展任何论文主实验。
