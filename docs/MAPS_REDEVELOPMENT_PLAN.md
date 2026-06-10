# MAPS 论文与实验系统完整改造方案

> 版本：v1.0  
> 制定日期：2026-06-11  
> 目标：将当前“LLM + MADDPG 任务卸载”原型改造成技术可信、可复现、可扩展，并能支撑重新投稿的研究系统。  
> 核心方法：Constraint-Verified Counterfactual Expert Distillation，简称 CV-CED。  
> 配套调研：[MAPS 创新定位与最新研究调研](../MAPS_novelty_positioning_report_2026.md)

## 1. 文档用途

本文档是后续代码开发、实验执行和论文重写的主依据。任何新增功能或实验，应能够回答以下问题：

1. 它修复了哪一个技术正确性问题？
2. 它验证了 CV-CED 的哪一个研究假设？
3. 它对应哪一个实验组、指标或论文图表？
4. 它满足什么条件后才算完成？

不直接服务于上述目标的重构、算法堆叠和大规模实验，暂不进入主线。

## 2. 项目最终目标

### 2.1 论文定位

论文不再以“LLM 辅助任务卸载”作为主要创新，而定位为：

> 面向不可靠 LLM 专家先验的、约束验证与反事实信用分配驱动的样本高效多智能体强化学习。

建议论文标题：

> MAPS: Constraint-Verified Counterfactual LLM Prior Distillation for Sample-Efficient Multi-Agent Task Partitioning in Terminal-Edge-Cloud Networks

### 2.2 核心研究问题

在端边云联合资源竞争中，LLM 对不同状态和不同 UE 的建议质量并不一致。统一模仿 LLM 可能导致错误卸载、服务器拥塞和负迁移。

新版 MAPS 需要回答：

> 如何利用网络物理模型验证 LLM 专家动作，并通过逐智能体反事实贡献，只把有益的专家先验蒸馏给 MARL 策略？

### 2.3 研究假设

后续实验围绕以下可证伪假设设计：

- **H1 样本效率：** 与无 LLM 指导的 MARL 相比，经过验证的专家先验能提高训练初期 Reward AUC，并减少达到固定性能阈值所需的环境交互次数。
- **H2 负迁移抑制：** 与统一蒸馏、固定退火和置信度门控相比，CV-CED 在 LLM 建议被扰动时具有更低的 Negative Transfer Rate。
- **H3 多智能体必要性：** 逐 UE 反事实门控优于只使用全局专家优势的门控，尤其在多个 UE 争用同一 ES 时。
- **H4 泛化能力：** 在未见过的 UE 数量、任务类型分布、负载强度或信道条件下，CV-CED 保持比原始 MAPS 更稳定的性能。
- **H5 部署成本：** LLM 仅用于训练阶段或稀疏查询阶段，部署时策略不依赖 LLM，推理时延与相同骨干 MARL 接近。

## 3. 当前项目审计结论

### 3.1 可以复用的部分

- `environment/task_generator.py` 已有泊松到达和多任务类型的基础实现。
- `environment/device_models.py` 已有 UE、ES 队列和计算时延的基础结构。
- `environment/cloud_edge_env.py` 已有三元任务划分、时延、能耗和 deadline 统计入口。
- `algos/mappo/`、`algos/happo/` 已有 PPO 系多智能体算法骨架。
- `algos/maac/` 已有时序 actor 和 attention critic 的部分实现。
- `utils/path_manager.py` 已有按实验隔离结果的目录体系。
- `utils/metrics.py` 和 `utils/csv_saver.py` 已有基础指标记录能力。

### 3.2 必须先修复的问题

以下问题存在时，任何大规模实验结果都不能用于论文：

| 编号 | 当前问题 | 影响 |
|---|---|---|
| B1 | 从项目根目录执行 `python main.py --help` 时无法导入 `LLM4RL` | 主入口不可复现 |
| B2 | `experiments/llm_maddpg/train_llm_maddpg_complete.py` 整体被注释 | 原始 MAPS 实际不可训练 |
| B3 | `llm_assistant/` 三个核心文件整体或大部分被注释 | LLM 专家链路不可用 |
| B4 | MADDPG critic 使用单个 agent 状态和动作的重复值代替真实联合状态动作 | 当前实现不构成可信的 MADDPG |
| B5 | MADDPG 的 edge 数量硬编码为 5 | 无法进行规模扩展 |
| B6 | edge id 被连续标量回归后再取整 | 混合动作建模不正确，梯度语义不清 |
| B7 | `simple_mode` 每步清空队列 | 不存在真实队列演化，无法研究动态任务流 |
| B8 | 泊松生成器可能产生同一 UE 多个任务，但环境只取第一个 | 到达模型和执行模型不一致 |
| B9 | 云服务器被设为无限资源且无队列 | 扩展性和拥塞结论不可信 |
| B10 | UE 到 ES 使用常数速率，缺少信道增益、路径损耗和噪声带宽 | 无线模型不完整 |
| B11 | ES 到 CS 的有线链路与无线发射能耗边界不清 | 能耗模型不严谨 |
| B12 | 当前指标以均值为主，缺少 P95、DVR、TCR、AUC 和负迁移率 | 无法证明可靠性与创新机制 |
| B13 | 配置只有单一 `config.yaml`，训练、消融、规模实验容易互相污染 | 实验难以复现 |
| B14 | 当前目录不是 Git 仓库，也没有自动测试 | 变更和实验版本不可追踪 |

### 3.3 总体改造策略

采用“保留领域环境骨架，重构训练主链路”的方案：

- 不完全重写项目。
- 保留任务、设备、指标和算法目录的领域划分。
- 重写 MADDPG 联合经验格式和训练逻辑。
- 恢复并结构化 LLM 专家链路。
- 新增无副作用的约束验证器和反事实评估器。
- 统一算法使用同一种混合动作编码、环境和评估协议。

论文文本按新论文重写，旧稿仅复用场景背景、部分符号和可验证的实验设置。

## 4. 目标系统模型

### 4.1 时间模型

系统采用离散时隙：

- 时隙索引：`t = 0, 1, ...`
- 时隙长度：`delta_t` 秒
- 所有到达量、服务量和队列更新均显式包含 `delta_t`

禁止混用 wall-clock 时间戳和仿真时间。任务的 `arrival_time`、`deadline` 和 `completion_time` 均使用仿真秒。

### 4.2 任务模型

UE `u` 在时隙 `t` 的任务定义为：

`task = (data_size, cpu_cycles, deadline, output_ratio, semantic_type, priority, arrival_slot)`

至少支持四类任务：

| 类型 | 典型含义 | 主要特征 |
|---|---|---|
| `video_analytics` | 视频分析 | 数据大、计算密集 |
| `ar_vr` | AR/VR 交互 | deadline 严格、输出较小 |
| `ai_inference` | AI 推理 | 计算密集、输入中等 |
| `control` | 实时控制 | 数据小、极低时延 |

语义类型的作用是帮助 LLM 形成任务优先级和资源选择先验。真实数值约束仍由环境和 verifier 决定。

### 4.3 队列模型

每个 UE、ES 和 CS 均维护以 CPU cycles 或任务对象表示的队列。实现时只能选择一种主量纲，推荐以任务对象维护，以 cycles 计算服务量。

标准更新满足：

`Q_n(t+1) = max(Q_n(t) - mu_n(t) * delta_t, 0) + A_n(t)`

需要处理：

- 一个时隙内同一 UE 的多个任务到达；
- 同一任务的三个并行子任务；
- 各节点有限并发能力；
- 跨时隙未完成任务；
- deadline 到期后的失败或超时完成；
- episode 结束时仍在队列中的任务。

主实验必须关闭 memoryless queue clearing。

### 4.4 通信模型

#### UE 到 ES

使用可配置的无线速率模型：

`R_u,e(t) = B_u,e log2(1 + p_u h_u,e(t) / (N0 B_u,e + I_u,e(t)))`

至少包括：

- 带宽 `B`
- 发射功率 `p`
- 信道增益 `h`
- 路径损耗
- 噪声功率谱密度 `N0`
- 可选干扰项 `I`

第一版可以采用正交接入令干扰为 0，但必须在论文中明确。

#### ES 到 CS

作为有线或回传链路独立建模：

- 传输时延：数据量 / backhaul rate + propagation latency
- 网络能耗：每 bit 能耗，或链路设备功率乘传输时间
- 不使用 UE 无线发射功率解释有线链路

#### 下行返回

结果返回数据量由 `output_ratio` 决定。可以建模为：

`D_down = output_ratio * D_input`

若忽略下行，应通过 `output_ratio` 很小的设置和敏感性实验说明其影响。

### 4.5 计算与能耗模型

本地和边缘计算能耗采用具有频率关系的模型，例如：

`E_comp = kappa * cycles * f^2`

需要明确：

- `kappa` 单位；
- CPU frequency 单位；
- DVFS 是否可控；
- ES 和 CS 的能耗是系统总能耗还是用户侧能耗。

建议报告两种口径：

- `device_energy`：UE 计算和无线发送能耗；
- `system_energy`：UE、ES、CS 和 backhaul 总能耗。

### 4.6 并行完成时延

任务被划分为本地、边缘和云三个子任务时，总完成时延应取并行分支完成时间的最大值：

`T_task = max(T_local, T_edge, T_cloud) + T_down`

每个分支包含：

- 排队时间；
- 上行或回传时间；
- 计算时间；
- 必要的结果返回时间。

### 4.7 混合动作空间

每个 UE 的动作分为两个头：

- 连续动作 `phi = [phi_local, phi_edge, phi_cloud]`，通过 softmax 保证和为 1；
- 离散动作 `edge_id`，通过 categorical 分布或 Gumbel-Softmax 训练。

禁止再将 `edge_id` 当作 `[0,1]` 连续数回归并取整。

### 4.8 Dec-POMDP 定义

#### 局部观察

每个 agent 至少观察：

- 自身 CPU、剩余能量或功耗参数；
- 自身队列长度和当前任务属性；
- 与各 ES 的信道状态或可用速率；
- 各 ES 的公开负载摘要；
- CS/backhaul 状态摘要；
- 任务 semantic type、deadline slack 和 priority。

#### 全局状态

集中训练 critic 使用所有 UE、ES、CS、信道、队列和任务状态。

#### 奖励

建议使用归一化成本：

`r_t = -(w_T T_norm + w_E E_norm + w_D DVR + w_Q Q_norm + penalties)`

需要满足：

- 所有项量纲归一化；
- reward 中的权重符号不与论文目标函数中的符号混淆；
- deadline 违约使用明确惩罚；
- 不通过 reward 泄露未来信息。

## 5. CV-CED 方法设计

### 5.1 方法流程

```text
网络状态与任务语义
        |
        v
LLM 生成候选联合专家动作 a_E
        |
        v
Action Projector 修复格式与硬约束
        |
        v
Constraint Verifier 评估可行性与预测代价
        |
        +--> 与当前策略动作 a_pi 比较全局优势
        |
        +--> 逐 UE 反事实替换，计算边际贡献
        |
        v
生成每个 UE 的 trust weight
        |
        v
混合动作蒸馏：比例 MSE + 节点选择 CE
        |
        v
MARL actor 更新；部署阶段不需要 LLM
```

### 5.2 LLM 专家输出

LLM 输出严格 JSON：

```json
{
  "schema_version": "1.0",
  "actions": [
    {
      "ue_id": 0,
      "partition": {
        "local": 0.2,
        "edge": 0.6,
        "cloud": 0.2
      },
      "edge_id": 2,
      "task_class": "ar_vr",
      "reason_code": "deadline_sensitive"
    }
  ]
}
```

LLM 可以输出自报 confidence 作为分析变量，但不能直接作为主门控权重。

### 5.3 Action Projector

在验证前执行确定性修复：

- 缺失 UE 使用 policy action 或安全默认动作；
- 比例裁剪到非负并归一化；
- `edge_id` 裁剪到有效范围；
- 对无任务 UE 生成 no-op mask；
- 记录所有格式错误和修复次数。

Projector 只能修复格式和硬约束，不能把低质量专家动作优化成近似最优动作，否则会混淆 LLM 与 verifier 的贡献。

### 5.4 Constraint Verifier

新增无副作用接口：

```python
evaluation = verifier.evaluate(state_snapshot, joint_action, horizon=1)
```

输出：

```python
{
    "feasible": True,
    "predicted_cost": 0.42,
    "latency": ...,
    "energy": ...,
    "deadline_violations": ...,
    "queue_backlog": ...,
    "constraint_violations": [...]
}
```

第一版采用解析环境模型的一步预测，不引入额外神经网络。只有在一步模型无法覆盖队列长期影响时，再增加短视界 rollout 或 learned world model。

### 5.5 全局专家优势

比较当前策略动作与专家动作：

`delta_global = J_hat(s, a_pi) - J_hat(s, a_E)`

只有满足以下条件时才允许进入蒸馏：

- 专家动作可行；
- `delta_global > global_margin`；
- verifier 未发现硬约束违规；
- 专家数据不是 parser fallback。

### 5.6 逐智能体反事实贡献

对每个 UE `u`，保持其他 UE 的专家动作不变，只把 UE `u` 替换为当前策略动作：

`delta_u = J_hat(s, (a_pi_u, a_E_-u)) - J_hat(s, a_E)`

解释：

- `delta_u > 0`：专家对 UE `u` 的建议有正边际贡献；
- `delta_u <= 0`：不应强制 UE `u` 模仿专家；
- 多个 UE 集中到同一 ES 时，反事实评估可以识别拥塞耦合。

计算量为每次专家查询 `U + 2` 次 verifier 调用。解析 verifier 足够轻量时可以接受；大规模场景可使用候选 UE 筛选、向量化评估或分组近似。

### 5.7 Trust Gate

第一版使用可解释的确定性门控：

`w_u = feasible * I(delta_global > m_g) * sigmoid((delta_u - m_u) / tau)`

可选项：

- verifier uncertainty；
- parser validity；
- LLM 多次采样一致性；
- 状态分布外分数。

这些只能作为附加因子，不能替代环境验证。

### 5.8 混合动作蒸馏

连续比例：

`L_part = ||phi_pi - phi_E||^2`

离散节点：

`L_edge = CE(edge_logits_pi, edge_id_E)`

总蒸馏损失：

`L_CED = mean(sum_u w_u * (L_part_u + eta * L_edge_u))`

actor 损失：

`L_actor_total = L_MARL + lambda_0 * L_CED`

`lambda_0` 是总体尺度，不进行按 episode 的固定三阶段退火。有效权重主要由 `w_u` 决定。

### 5.9 理论性质

论文不承诺证明非凸深度 MARL 的全局收敛。建议证明安全接受性质：

若 verifier 的代价误差满足：

`|J_hat(s,a) - J(s,a)| <= epsilon`

且只在：

`delta_global > m_g`，其中 `m_g > 2 epsilon`

时接受专家动作，则被接受的专家联合动作真实代价优于当前策略动作。

该结论直接支撑“抑制错误指导”，比泛化的收敛声明更可靠。

## 6. 目标代码架构

### 6.1 计划目录

```text
LLM4RL/
├── algos/
│   ├── common/
│   │   ├── hybrid_action.py
│   │   ├── joint_batch.py
│   │   └── seed.py
│   ├── maddpg/
│   ├── mappo/
│   └── happo/
├── environment/
│   ├── cloud_edge_env.py
│   ├── device_models.py
│   ├── task_generator.py
│   ├── channel_model.py
│   ├── backhaul_model.py
│   ├── action_codec.py
│   └── snapshot.py
├── expert_guidance/
│   ├── llm_expert.py
│   ├── prompt_builder.py
│   ├── response_parser.py
│   ├── action_projector.py
│   ├── verifier.py
│   ├── counterfactual.py
│   ├── trust_gate.py
│   └── cache.py
├── experiments/
│   ├── runner.py
│   ├── evaluate.py
│   ├── aggregate.py
│   └── presets/
├── configs/
│   ├── base.yaml
│   ├── smoke.yaml
│   ├── baselines/
│   ├── ablations/
│   └── scale/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── docs/
├── main.py
└── README.md
```

不要求一次性创建全部文件。目录按阶段逐步演进。

### 6.2 现有文件改造映射

| 文件或模块 | 计划改造 |
|---|---|
| `main.py` | 修复包导入；将算法选择、配置覆盖、seed 和结果目录统一交给 runner |
| `config.yaml` | 保留为兼容入口；迁移到分层配置，并增加 schema 校验 |
| `environment/cloud_edge_env.py` | 拆分状态推进、动作评估和统计；移除主实验队列清空；增加 snapshot/evaluate 接口 |
| `environment/device_models.py` | 加入有限云资源、并发容量、标准服务过程和能耗单位 |
| `environment/task_generator.py` | 支持一个 UE 多任务、不丢弃任务；增加语义类型、输出比例和可复现 RNG |
| `algos/maddpg/maddpg_actor_critic.py` | 替换为连续比例头和 categorical edge 头 |
| `algos/maddpg/maddpg_agent.py` | 使用真实 joint batch、其他 agent target actor 和混合动作损失 |
| `algos/maddpg/replay_buffer.py` | 一条 transition 存储完整联合状态、联合动作、联合奖励、mask 和专家元数据 |
| `algos/mappo/` | 改为相同混合动作接口，避免 Beta 分布回归 edge id |
| `algos/happo/` | 与 MAPPO 使用同一 action codec 和环境协议 |
| `llm_assistant/` | 恢复后迁移为 `expert_guidance/`，保留兼容导入一段时间 |
| `experiments/llm_maddpg/` | 由统一 runner 替代被注释脚本 |
| `utils/metrics.py` | 增加可靠性、样本效率、专家质量和负迁移指标 |
| `utils/csv_saver.py` | 禁止补齐缺失数据；改为长表格式并记录 seed/config hash |
| `utils/path_manager.py` | 增加 run manifest、config snapshot、system info 和 commit id |

## 7. 配置与复现规范

### 7.1 配置层次

建议最终配置合成顺序：

```text
base.yaml
  + algorithm preset
  + experiment preset
  + CLI overrides
```

例如：

```powershell
python -m LLM4RL.main `
  --config configs/base.yaml `
  --algorithm cv_ced_maddpg `
  --preset configs/scale/ue50_es10.yaml `
  --seed 42
```

### 7.2 每次运行必须保存

- 合并后的完整配置；
- 随机种子；
- Python、PyTorch、CUDA 和 GPU 信息；
- Git commit id；若无 Git，记录代码快照 hash；
- LLM 模型、temperature、prompt version；
- LLM 原始响应 hash 和缓存命中信息；
- 开始/结束时间和异常状态；
- 训练与测试指标；
- 模型 checkpoint；
- 运行命令。

### 7.3 随机性

统一设置：

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- Gym environment
- task generator
- channel generator

测试时使用与训练不同但固定的 seed 集合。

### 7.4 LLM 缓存

相同 prompt 和模型配置必须复用缓存，以保证：

- 不同算法使用相同专家建议；
- 消融实验公平；
- 减少 API 成本和运行时间；
- 结果可重复。

缓存 key 至少包括：

`model + prompt_version + normalized_state + temperature + seed`

## 8. 分阶段实施计划

估时按一名开发者计算，不包含等待远程 LLM 或大规模 GPU 实验的排队时间。

### Phase 0：版本与研究基线冻结

预计：0.5 至 1 天。

任务：

- 备份当前 PDF、LaTeX、原始图和当前代码；
- 初始化 Git，或建立只读代码快照；
- 保存当前 `config.yaml`；
- 建立 `docs/` 和实验决策记录；
- 禁止把旧实验结果与新环境结果混用。

产物：

- `legacy/` 或独立归档；
- 当前代码 commit/tag；
- 旧论文结果清单。

完成门槛：

- 任意旧结果可以追溯到原配置和原代码；
- 新实验目录明确标记 `model_version >= 2`。

### Phase 1：可运行、可复现的真实基线

预计：3 至 5 天。

任务：

- 修复 `python -m LLM4RL.main --help` 和项目根目录运行方式；
- 恢复或重写原始 MAPS 训练入口；
- 修复 MADDPG joint replay 和 centralized critic；
- 移除 edge 数量硬编码；
- 统一 seed、日志、checkpoint 和结果 manifest；
- 建立最小 unit/smoke tests；
- 分别跑通 MADDPG、MAPPO、HAPPO 和原始 MAPS。

测试：

- action shape 和范围；
- joint replay 样本 shape；
- target actor 来自对应 agent；
- 20 steps smoke training；
- 同 seed 结果一致；
- 不同 seed 任务序列不同。

完成门槛 Gate 1：

- 所有基线可由命令行启动；
- smoke run 无 NaN、shape error 和空结果；
- MADDPG 不再复制单 agent 动作冒充 joint action；
- 原始 MAPS 有真实 LLM 或固定缓存专家路径；
- 结果包含配置和 seed。

未通过 Gate 1，不进入系统模型实验。

### Phase 2：技术可信的端边云环境

预计：5 至 8 天。

任务：

- 引入仿真时隙 `delta_t`；
- 实现不丢任务的到达队列；
- 关闭主实验的 memoryless clearing；
- 实现有限 UE、ES、CS 计算资源；
- 实现无线信道速率；
- 分离 backhaul 时延与能耗；
- 加入下行结果返回；
- 正确计算并行分支最大完成时延；
- 增加 environment snapshot 和纯函数式候选动作评估。

测试：

- 无到达时队列不增长；
- 服务量大于队列时队列不为负；
- 增大带宽时传输时延单调下降；
- 增大 CPU 频率时执行时延单调下降；
- 增大发射功率时无线速率单调不降；
- 三个分支总时延等于最大分支加返回时延；
- `evaluate(action)` 不改变真实环境状态；
- 手工小例子与解析计算一致。

完成门槛 Gate 2：

- 所有量纲在文档和代码中一致；
- 队列跨时隙演化；
- verifier 与环境一步执行在确定性条件下结果一致；
- Reviewer 1 和 Reviewer 3 的系统模型问题均有代码对应。

未通过 Gate 2，不实现 CV-CED。

### Phase 3：统一混合动作接口

预计：3 至 5 天。

任务：

- 实现 `HybridAction` 数据结构；
- 连续比例统一使用 softmax；
- 离散 edge 统一使用 categorical；
- 环境、MADDPG、MAPPO、HAPPO、LLM parser 使用相同 codec；
- 处理 no-task mask 和 invalid-action mask；
- 更新 replay/trajectory buffer。

测试：

- 比例非负且和为 1；
- edge id 始终有效；
- batch encode/decode 可逆；
- 无任务 agent 不贡献 loss；
- 离散和连续 loss 的梯度均存在。

完成门槛 Gate 3：

- 所有算法在相同环境中输出相同 schema；
- 不再存在连续 edge id 取整逻辑；
- 原始 MAPS 的统一 MSE 被标记为 legacy baseline。

### Phase 4：可审计的 LLM 专家链路

预计：3 至 5 天。

任务：

- 设计 prompt v2；
- 输出严格 JSON schema；
- 实现 parser、projector、fallback 和缓存；
- prompt 中加入任务语义、deadline slack、链路和队列摘要；
- 记录 query latency、tokens、parser error 和 fallback；
- 支持固定专家缓存用于公平实验。

测试：

- 正常、缺字段、额外文本、非法比例、非法 edge 的 parser 测试；
- 同一缓存 key 返回相同专家动作；
- fallback 不被标记为有效专家样本；
- LLM 不可用时基线仍可运行。

完成门槛 Gate 4：

- 100 个离线状态的解析成功率可报告；
- 所有专家动作有来源、版本和有效性标记；
- 不使用 LLM 自报 confidence 直接决定蒸馏。

### Phase 5：实现 CV-CED

预计：5 至 8 天。

任务：

- 实现 `ConstraintVerifier`；
- 实现全局 expert advantage；
- 实现逐 UE counterfactual evaluator；
- 实现 trust gate；
- 实现连续 MSE + 离散 CE；
- 在 replay 中保存 expert action、validity、cost 和 trust；
- 实现 corrupted expert 注入器；
- 增加理论命题所需的误差和 margin 统计。

性能优化：

- verifier 批量评估；
- 只对存在任务且专家与策略动作不同的 UE 做反事实；
- 大规模场景允许 top-k UE 近似；
- 缓存相同状态动作对的 verifier 结果。

测试：

- 专家明显更优时 gate 开启；
- 专家不可行时 gate 为 0；
- 仅一个 UE 建议有益时只有该 UE 获得正权重；
- policy 与 expert 相同时蒸馏梯度接近 0；
- corrupted expert 不应比统一蒸馏获得更高接受率；
- 关闭 verifier 或 counterfactual 后退化为对应消融。

完成门槛 Gate 5：

- CV-CED 组件可独立单测；
- 小场景能够记录每个 UE 的 trust weight；
- 不依赖固定 episode 退火；
- 部署测试完全关闭 LLM。

### Phase 6：小规模机制验证

预计：3 至 5 天。

固定场景：

- 10 UE；
- 5 ES；
- 1 CS；
- 100 至 300 episodes；
- 5 个随机种子；
- 使用固定 LLM 缓存。

比较：

1. MADDPG；
2. Legacy MAPS：统一 MSE + 固定退火；
3. Hybrid Loss MAPS；
4. Global Verified Distillation；
5. Confidence-Gated Distillation；
6. CV-CED。

必须回答：

- CV-CED 是否提高 Reward AUC？
- 是否降低达到目标回报所需环境步数？
- corrupted expert 下是否降低负迁移？
- counterfactual 是否比 global-only 有额外收益？
- verifier 额外计算成本是否可接受？

完成门槛 Gate 6：

- H1、H2、H3 至少在趋势上成立；
- CV-CED 不只在单个 seed 有效；
- 指标差异不是由不同 LLM 建议或不同环境序列造成；
- 若 H2 不成立，停止大规模实验并修改 gate。

### Phase 7：完整论文实验

预计：1 至 3 周，取决于 GPU 和 LLM 缓存生成速度。

实验分层如下。

#### E1 主性能实验

算法：

- MADDPG；
- MAPPO；
- HAPPO；
- LLM-only；
- Legacy MAPS；
- CV-CED-MADDPG；
- 可选 CV-CED-MAPPO。

场景：

- 30 UE、10 ES；
- normal load；
- 5 seeds。

#### E2 消融实验

- w/o constraint verification；
- w/o counterfactual，仅 global gate；
- w/o hybrid loss；
- fixed annealing；
- self-reported confidence gate；
- w/o task semantics；
- full CV-CED。

#### E3 扩展性实验

- UE：10、30、50、100；
- ES：5、10、15；
- 500 UE 仅作为压力测试；
- 测量性能和 wall-clock cost。

#### E4 任务负载与可靠性

- low、normal、peak arrival rate；
- 宽松、中等、严格 deadline；
- 不同任务类型分布；
- 节点失效或 backhaul 降速；
- 信道质量变化。

#### E5 专家质量鲁棒性

- normal expert；
- small LLM expert；
- random perturbation；
- wrong-edge bias；
- cloud-heavy bias；
- stale state expert；
- random expert。

#### E6 泛化实验

- 小规模训练、大规模测试；
- 未见任务类型比例；
- 未见到达率；
- 未见信道分布；
- 未见 ES capacity。

#### E7 LLM 对比与开销

建议只选择两个不同模型族，避免实验失控：

- 当前 Qwen 系模型；
- 一个不同模型族的同量级或较小模型。

报告：

- query latency；
- token 数；
- parser success rate；
- cache hit rate；
- 专家接受率；
- 训练附加 wall-clock；
- 部署时延。

### Phase 8：统计分析与图表

预计：3 至 5 天。

统计原则：

- 主结果至少 5 seeds；
- 报告 mean、standard deviation 和 95% confidence interval；
- 同一 seed 和任务序列下做配对比较；
- 同时报告 p-value 与 effect size；
- 曲线显示置信带，不只显示平滑均值；
- 不以目测决定收敛点。

建议的收敛定义：

- 先由无指导基线最终 10% 训练区间定义目标回报；
- 计算各算法首次达到并连续保持该阈值 `K` 个 evaluation points 所需环境步数；
- 同时报告 Reward AUC。

论文图表建议：

| 编号 | 内容 |
|---|---|
| Fig. 1 | 端边云系统与任务并行划分 |
| Fig. 2 | CV-CED 训练流程 |
| Fig. 3 | Reward/AUC 与样本效率 |
| Fig. 4 | corrupted expert 下负迁移率和接受率 |
| Fig. 5 | UE/ES 规模扩展 |
| Table I | 系统参数 |
| Table II | 主性能：平均/P95 时延、能耗、DVR、TCR |
| Table III | 消融与计算开销 |

### Phase 9：论文重写与投稿准备

预计：5 至 8 天，不含合作者修改。

推荐结构：

1. Introduction
2. Related Work and Research Gap
3. System Model and Problem Formulation
4. CV-CED Method
5. Analysis
6. Experiments
7. Conclusion

如果目标仍是 4 页 IEEE Networking Letters，需要压缩为：

1. Introduction
2. System Model
3. CV-CED
4. Evaluation
5. Conclusion

完整系统模型推导、更多实验和 prompt 可放补充材料或代码仓库。若实验内容已经明显超过 Letter 容量，应考虑完整期刊或更长篇幅 venue，而不是再次把关键证据删掉。

## 9. 实验指标定义

### 9.1 网络性能

- Mean Latency
- P95 Latency
- Device Energy
- System Energy
- Deadline Violation Ratio，DVR
- Task Completion Rate，TCR
- On-Time Completion Rate
- Queue Backlog
- Jain's Fairness Index
- ES/CS Utilization

### 9.2 样本效率

- Reward AUC
- Environment Steps to Threshold
- Wall-Clock Time to Threshold
- Final Performance
- Training Stability

### 9.3 专家指导质量

- Expert Feasibility Rate
- Expert Global Advantage
- Agent-Level Positive Contribution Rate
- Expert Acceptance Rate
- Parser Failure Rate
- Fallback Rate
- Policy-Expert Similarity

### 9.4 负迁移

定义一次被接受的专家指导在真实环境执行或可靠 rollout 后使成本变差为负迁移：

`NTR = harmful_accepted_guidance / accepted_guidance`

同时报告：

- harmful guidance 总数；
- harmful guidance 被拒绝率；
- 不同扰动强度下的 NTR；
- global gate 和 counterfactual gate 的差异。

## 10. 最终实验矩阵

### 10.1 论文正文最小集合

若篇幅受限，正文至少包含：

- MADDPG、MAPPO、Legacy MAPS、CV-CED；
- 5 seeds；
- 10/30/50/100 UE；
- full、w/o verifier、w/o counterfactual、fixed annealing；
- normal/corrupted expert；
- Reward AUC、P95 latency、DVR、energy、NTR；
- LLM 查询与部署开销。

### 10.2 扩展材料集合

- HAPPO、MAAC；
- 第二种 LLM；
- 更多任务分布；
- 节点失效；
- 500 UE 压力测试；
- prompt 和完整参数敏感性。

### 10.3 公平比较要求

所有算法必须共享：

- 相同训练和测试任务序列；
- 相同信道序列；
- 相同 episode/step budget；
- 相同环境模型；
- 相同奖励；
- 相同评估频率；
- 可比的网络规模；
- 相同 LLM 缓存，若算法使用 LLM。

不能将不同训练频率直接解释为算法样本效率优势。样本效率按 environment interactions 计量，计算效率按 wall-clock 单独报告。

## 11. Reviewer 意见映射

| Reviewer 问题 | 方案对应 |
|---|---|
| 创新仅为 MSE 蒸馏和退火 | CV-CED：物理验证、逐 agent 反事实贡献、混合动作蒸馏 |
| 规模只有 10 UE、5 ES | E3：10 至 100 UE、5 至 15 ES，500 UE 压力测试 |
| 基线单一 | MADDPG、MAPPO、HAPPO、LLM-only、Legacy MAPS |
| LLM 模型单一 | E7：至少两个模型族，并报告专家质量而非只比模型名 |
| 每 UE 最多一个任务不现实 | 多任务到达队列，不丢弃同一时隙多个任务 |
| 一次只执行一个任务 | 显式配置节点并发能力和服务过程 |
| 队列公式量纲错误 | 显式 `delta_t`、到达和服务统一量纲 |
| 有线链路能耗错误 | 独立 backhaul 模型 |
| 无信道增益和 `N0B` | 标准无线速率模型 |
| 混合动作损失错误 | softmax partition + categorical edge；MSE + CE |
| 下行时延被忽略 | output ratio 和下行返回模型 |
| Algorithm 1 太抽象 | 给出 query、project、verify、counterfactual、gate、distill 全流程 |
| LLM 与 MDP 连接不清 | 专家动作为状态条件 prior，经 verifier 后进入 actor loss |
| actor/critic/target 未解释 | 在方法和实现文档中明确 CTDE 更新 |
| 要求数学证明 | 提供 verifier 误差有界下的安全接受性质 |
| Normalized Reward 未定义 | 使用明确归一化和原始物理指标共同报告 |
| LLM 推理开销未讨论 | E7 报告 token、query latency、cache、训练和部署开销 |

## 12. 风险与应对

### 12.1 CV-CED 没有提高最终性能

这是可接受结果。主张应聚焦样本效率和负迁移抑制，不强制声称最终最优性能更高。

应对：

- 检查 expert 是否本来就弱；
- 报告 early-stage AUC；
- 调整 global margin 和 temperature；
- 使用更稀疏的专家查询；
- 不通过删除不利 seed 美化结果。

### 12.2 Verifier 与环境不一致

应对：

- 共享同一套物理计算函数；
- verifier 使用 environment snapshot；
- 对一步 deterministic 场景做逐项一致性测试；
- 将预测误差纳入 margin。

### 12.3 反事实计算量过大

应对顺序：

1. 向量化 verifier；
2. 跳过 no-task UE；
3. 只评估专家和策略差异大的 UE；
4. top-k UE；
5. 分组或采样反事实。

正文必须报告采用的近似。

### 12.4 LLM 输出不稳定

应对：

- temperature 设为 0 或低值；
- JSON schema；
- cache；
- parser/projector；
- 多次采样一致性只作为可选特征；
- 将失败输出视为无专家，而不是默认“全云”专家。

### 12.5 Letter 篇幅不足

应对：

- 核心只保留一个创新 CV-CED；
- 语义任务、混合动作和环境修正作为建模；
- 正文保留主结果和关键消融；
- 更多实验放补充材料；
- 若目标 venue 不接受补充材料，改投长文 venue。

## 13. 里程碑与预计周期

| 周期 | 目标 | 退出条件 |
|---|---|---|
| Week 1 | Phase 0-1，基线可运行 | Gate 1 |
| Week 2 | Phase 2，可信环境 | Gate 2 |
| Week 3 | Phase 3-4，混合动作与 LLM 链路 | Gate 3-4 |
| Week 4 | Phase 5，CV-CED | Gate 5 |
| Week 5 | Phase 6，小规模机制验证 | Gate 6 |
| Week 6-7 | Phase 7，完整实验 | 主矩阵完成 |
| Week 8 | 统计、图表、论文初稿 | 所有 claim 有实验对应 |
| Week 9 | 合作者修改和投稿材料 | 可投稿版本 |

这是理想排期。当前基线问题比旧论文描述更严重，建议预留 20% 至 30% 缓冲。

## 14. 下一迭代的具体任务

下一迭代只做 Phase 0 和 Phase 1，不同时修改完整系统模型。

### Sprint 1 Backlog

- [ ] 建立 Git 或代码快照；
- [ ] 新增依赖清单和运行环境说明；
- [ ] 修复包结构和 `main.py` 入口；
- [ ] 建立 `configs/smoke.yaml`；
- [ ] 新增统一 seed 工具；
- [ ] 重写 joint replay transition；
- [ ] 修复 MADDPG centralized critic；
- [ ] 移除 `num_edges = 5`；
- [ ] 恢复一个可运行的 Legacy MAPS 训练入口；
- [ ] 恢复 LLM parser 的最小可用版本；
- [ ] 新增 action、buffer、seed 和 smoke tests；
- [ ] 跑 2 seeds、20 episodes 的 baseline smoke comparison；
- [ ] 输出 `baseline_audit.json`。

### Sprint 1 不做

- 不跑 50/100 UE；
- 不实现 learned digital twin；
- 不接入四种 LLM；
- 不修改论文结果数字；
- 不实现 CV-CED；
- 不将 smoke 结果用于论文。

## 15. 项目完成定义

只有同时满足以下条件，才能认为论文改造完成：

### 代码

- 所有主算法可通过统一命令运行；
- 核心模块有单元和集成测试；
- 无硬编码 UE/ES 数量；
- 无连续 edge id 取整；
- verifier 无副作用且与环境一致；
- 测试部署不调用 LLM。

### 实验

- 主结果至少 5 seeds；
- 所有算法共享测试序列；
- 包含样本效率、可靠性、扩展性、鲁棒性和开销；
- 包含 verifier 与 counterfactual 消融；
- 包含 corrupted expert；
- 统计脚本可以从原始记录重新生成图表。

### 论文

- 摘要只声明实验真正支持的结论；
- 贡献点控制为三条；
- 不把技术修正包装成主要创新；
- 所有公式量纲一致；
- Algorithm 1 可复现；
- 最新相关工作覆盖至投稿前；
- 代码、配置、图表和论文参数一致。

## 16. 决策记录

### D1：代码改造还是完全重写

决定：保留现有领域目录和部分环境组件，重构训练主链路。  
理由：任务、设备和指标已有可复用结构，但训练入口和 MADDPG 联合训练不可信。

### D2：核心创新数量

决定：只保留 CV-CED 一个核心机制。  
理由：任务语义、混合动作和可靠性指标是必要设计，不再拆成独立创新。

### D3：LLM confidence

决定：不使用 LLM 自报 confidence 作为主门控。  
理由：缺少校准，且最新工作已有不确定性门控；本文强调可验证网络代价。

### D4：固定退火

决定：保留为 legacy baseline，不用于完整方法。  
理由：专家价值依赖状态和 agent，不应只依赖 episode。

### D5：理论目标

决定：证明安全接受性质，不承诺深度 MARL 全局收敛。  
理由：与方法机制一致，也更符合可证明范围。

### D6：实验扩展顺序

决定：先 10 UE 小规模证明机制，再扩展到 100 UE。  
理由：创新假设未成立前进行大规模实验只会增加成本。

## 17. 维护规则

- 每完成一个 Gate，更新本文档状态和实际日期。
- 算法定义、实验矩阵或论文 claim 发生变化时，先更新本文档再改代码。
- 新增实验必须登记配置路径、seed 集合和预期图表。
- 被放弃的路线写入决策记录，不从历史中删除。
- 论文中的最终数字只能来自带 manifest 的正式实验目录。

