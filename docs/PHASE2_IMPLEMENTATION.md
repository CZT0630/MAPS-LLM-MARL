# Phase 2 实现与审计记录

完成日期：2026-06-11

**来源说明：Phase 2 由人工完成初稿，随后由 Codex 对照 Gate 2 独立审查、复现缺陷并修复。**

## 审查结论

人工初稿的 45 项测试可以运行，但其中多项只验证“结果为正”或复用了实现公式，未能证明环境正确。独立数值探针复现了以下问题：

- `model_version` 被误用为物理模型开关，Phase 1 回归实际进入 Phase 2；
- 同一 UE 同时到达的多个任务只保留第一个；
- `arrival_time` 使用 wall-clock 时间；
- 远端任务在传输完成前获得计算服务；
- 当前任务提前完成后，时隙剩余服务量未用于后续任务；
- 纯本地动作仍被收取无线下行时延；
- ES/CS 计算能耗未计入系统能耗；
- snapshot 丢失 UE 距离和 CS 队列；
- evaluator 未建模多 UE 资源竞争，与真实队列不一致；
- 待决策队列等待时间未计入 deadline；
- Phase 2 新增的信道、语义和云负载未进入 MARL 观察。

上述问题已修复，Gate 2 重新通过。

## 最终实现

### 版本隔离

- `model_version` 仅表示结果和 manifest 格式版本。
- `environment.physics_version` 控制物理模型。
- `configs/smoke.yaml` 未设置该字段，保持 Phase 1 行为。
- `configs/smoke_phase2.yaml` 显式设置 `physics_version: 2`。

### 时间与队列

- 仿真采用长度为 `delta_t` 秒的离散时隙。
- 任务的 `arrival_time`、释放时间、开始时间和完成时间均使用仿真秒。
- UE、ES、CS 使用非抢占、按释放时间排序的 FCFS 计算队列。
- 远端子任务仅在上行或回传完成后进入可服务状态。
- 一个时隙内可连续完成多个任务，不丢弃剩余服务能力。
- 同一 UE 的多个泊松到达保存在独立待决策队列中，每时隙最多调度队首任务。

### 通信与返回路径

- UE→ES：`R = B log2(1 + p h / (N0 B))`，当前采用正交接入。
- ES→CS：独立有线回传时延和每 bit 能耗。
- 本地分支不产生下行时延。
- 边缘分支返回路径为 ES→UE。
- 云分支返回路径为 CS→ES→UE。
- 总任务时延为三个完整并行分支时延的最大值。

### 能耗口径

- 计算能耗：`E_comp = kappa * cycles * f_Hz^2`。
- `kappa` 单位为 `J / (cycle * Hz^2)`。
- `total_energies` 报告系统总能耗，包括 UE 计算、无线传输、ES/CS 计算和回传能耗。
- `user_energies` 报告 UE 侧能耗。
- `reward.energy_scope` 显式选择奖励使用 `user` 或 `system`，避免混合口径。

### 任务与观察

- 支持 `video_analytics`、`ar_vr`、`ai_inference`、`control` 四类任务。
- 任务包含 `output_ratio`、`priority`、`arrival_slot` 和仿真到达时间。
- Phase 2 状态包括 UE/ES/CS 负载、待决策积压、UE→ES 速率、deadline slack、priority 和语义 one-hot。
- Phase 1 的全局状态和局部状态维度保持不变。

### Snapshot 与联合动作评估

- snapshot 保存完整计算队列、释放时间、剩余服务时间、UE 距离和当前任务。
- evaluator 按与环境相同的释放时间 FCFS 规则联合调度所有 UE。
- evaluator 输出每个分支的释放、开始、完成、通信、计算和能耗明细。
- `evaluate_action()` 无副作用；`step()` 应用同一评估计划。

## 量纲

| 量 | 代码单位 |
|---|---|
| 数据量 | MB，传输前乘 `8e6` 转 bits |
| CPU 工作量 | cycles |
| CPU 频率 | 配置为 GHz，计算时转 Hz |
| 带宽与速率 | Hz、bits/s |
| 时间 | 仿真秒 |
| 功率 | W |
| 能量 | J |
| 噪声功率谱密度 | W/Hz |
| `kappa` | J / (cycle × Hz²) |

## Gate 2 证据

- Phase 2 环境测试：48 项；
- Phase 1 回归测试：11 项；
- 全套测试：59 项；
- Phase 1 四算法 smoke：通过；
- Phase 2 四算法 smoke：通过；
- evaluator 与 step 在共享 ES、反序传输到达和已有队列条件下逐 UE 一致；
- 同一 UE 多任务到达不丢失；
- Phase 1 的 `model_version: 2` 不再错误启用 Phase 2。

机器可读审计记录：`artifacts/phase2/environment_audit.json`。

## 明确边界

- 无线模型当前为静态路径损耗和正交接入，尚未采样快衰落与干扰。
- 每个 UE/ES 使用单服务 FCFS；CS 用 `parallel_factor` 表示聚合并行能力。
- deadline 结果由确定性队列计划在任务准入时计算，尚未实现事件回调式完成日志。
- 下行使用可配置发射功率并复用同一路径损耗模型。
- MAPPO/HAPPO 的离散 edge selector 仍待 Phase 3 修复。
- 本阶段 smoke 只证明工程正确性，不构成论文性能结论。
