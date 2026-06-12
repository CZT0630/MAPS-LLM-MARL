# MAPS EI 会议论文修改与实验实施方案

> 分支：`publication/ei-conference`  
> 制定日期：2026-06-13  
> 目标：在保留原始 MAPS 技术主线的前提下，以较低开发成本修复技术硬伤、补足关键证据，并形成可投稿 EI 会议的完整论文。  
> 非目标：本分支不实现 Letter 路线中的 CV-CED、反事实信用分配、约束门控、多 LLM 大矩阵或完整 Phase 0-9 重构。

## 1. 总体结论

EI 版本不应把创新定位为“首次结合 LLM 与 MARL”，也不应把“LLM 提高 RL 样本效率”单独作为原创发现。2023-2026 年的相关研究已经覆盖 LLM policy teacher、LLM-guided RL regularization、LLM-enhanced MARL offloading、LLM 直接生成卸载策略等方向。

本论文适合采用以下克制、可辩护的定位：

> This paper develops an LLM-guided multi-agent reinforcement learning framework for fine-grained task partitioning and parallel scheduling in terminal-edge-cloud networks. The LLM provides offline expert priors during training, while the learned decentralized policies make real-time decisions without invoking the LLM at deployment.

中文概括：

> 面向端-边-云三级异构网络，开发一种 LLM 指导的多智能体细粒度任务划分与并行调度框架；LLM 仅在训练阶段提供专家先验，部署阶段由轻量 MARL 策略独立实时决策。

该定位的差异点不是“LLM + MARL”本身，而是以下组合：

1. 任务不是二元整体卸载，而是在本地、边缘和云端之间进行连续比例划分，并按并行分支的最大完成时间评价任务时延。
2. 每个 UE 同时决定连续划分比例和离散 ES 选择，构成混合动作的多智能体联合决策。
3. LLM 专家动作通过混合动作蒸馏进入 MADDPG 训练，并通过退火逐步退出；部署时不需要 LLM。
4. 在带队列、deadline、无线链路和有线回传的 TEC 动态任务流中评价时延、能耗、可靠性和扩展性。

这是一篇“系统建模 + 框架实现 + 实证评价”型 EI 会议论文，而不是强调单点算法突破的 Letter。

## 2. 建议标题与贡献

### 2.1 推荐标题

首选：

> MAPS: LLM-Guided Multi-Agent Task Partitioning and Parallel Scheduling in Terminal-Edge-Cloud Networks

备选：

> LLM-Guided Multi-Agent Learning for Fine-Grained Task Partitioning in Terminal-Edge-Cloud Networks

不再使用：

- `innovative framework`
- `AGI technology is changing our lives`
- `insufficient generalization capabilities`
- `can be extended to other areas`
- `better adapting to the 6G network`
- 没有实验支持的 `robust`, `generalizable`, `scalable` 或 `near-optimal`

### 2.2 一句话贡献

> MAPS transfers offline LLM-generated scheduling priors into decentralized MARL policies through mixed-action distillation and annealed guidance, enabling real-time fine-grained terminal-edge-cloud task partitioning without online LLM inference.

### 2.3 论文贡献控制为三条

1. **问题建模：** 建立动态 TEC 任务流下的细粒度并行任务划分问题，显式考虑 UE/ES/CS 队列、无线传输、有线回传、结果返回、能耗和 deadline，并将其表述为 Dec-POMDP。
2. **方法设计：** 开发 MAPS，将 LLM 生成的状态条件专家先验蒸馏到 MADDPG actor；连续划分比例使用 MSE，离散 ES 选择使用交叉熵，并以退火系数逐步降低专家影响。
3. **实验评价：** 通过消融、可靠性和 UE 规模实验，比较 MAPS、无 LLM 的 MADDPG、无退火版本和 LLM-only，报告收敛效率、平均/P95 时延、能耗、DVR 和 TCR。

任务语义、正确的物理模型、混合动作损失和实验基础设施属于必要设计，不单独包装成创新点。

## 3. 最新研究现状与论文边界

### 3.1 已被相关工作覆盖的内容

- LLM 作为 RL policy teacher，并由小型策略在部署阶段替代 LLM，已由 LLM4Teach 等工作研究。
- 将 LLM prior 作为 RL regularization 以提升样本效率，已由 LINVIT 等工作研究。
- LLM 与 MARL/任务卸载的结合已出现在 UAV-assisted MEC 和 multi-cloud scheduling 中。
- 2026 年的 LeDRL 已在 collaborative edge computing 中结合语义任务信息、LLM 高层先验、MAPPO、注意力融合、反思和规模实验。
- 2026 年的 COMLLM 已让 LLM 直接学习考虑队列演化的多步 MEC 卸载策略，并强调拓扑零样本扩展。

因此，EI 版不能声称：

- 首次使用 LLM 指导 MARL；
- 首次使用 LLM 解决边缘卸载；
- 提出了通用的样本高效 RL 方法；
- 具有跨拓扑或跨场景泛化能力；
- 在理论上保证 MARL 收敛。

### 3.2 MAPS 可以保留的差异

相较于上述邻近工作，EI 版只强调当前系统确实实现并实验验证的差异：

- 三级 TEC，而非单层 MEC 或 UAV 二元卸载；
- 同一任务在 local/edge/cloud 三条分支上的比例划分与并行完成；
- 连续比例和离散节点选择的混合动作；
- LLM 作为离线训练先验，而不是在线控制器；
- 以可靠性指标评价 deadline-sensitive 动态任务流。

这些差异应写为“focuses on”或“is designed for”，不轻易写成“the first”。

## 4. 当前代码审计结论

### 4.1 已完成，可直接复用

Phase 0-2 已经完成以下基础工作：

- 旧代码和原始 LaTeX 已冻结在 `legacy/`；
- MADDPG 使用真实 joint replay 和 centralized critic；
- 支持固定随机种子、运行 manifest、独立结果目录和回归测试；
- Phase 2 使用带 `delta_t` 的仿真时间和持续队列；
- 同一 UE 的多任务到达不会丢失；
- UE-ES 使用含路径损耗和 `N0 B` 的 Shannon 速率；
- ES-CS 使用独立有线回传时延和每 bit 能耗；
- local/edge/cloud 计算能耗均可统计；
- 下行结果返回进入任务时延；
- 支持四类任务语义、deadline、priority 和 output ratio；
- 已提供平均时延、系统/用户能耗和任务完成统计入口；
- `evaluate_action()` 与环境队列模型共享确定性计算逻辑。

上述修复解决了原审稿意见中的大部分 system model 硬伤，但原始 LaTeX 尚未同步。

### 4.2 尚未满足论文实验要求

1. `algos/maddpg/maddpg_actor_critic.py` 仍将 ES 选择输出为 `[0,1]` 连续标量。
2. `policy_to_env_action()` 仍通过乘以 ES 数量后取整得到 `edge_id`，离散动作梯度语义不成立。
3. `MADDPGAgent.update()` 对三个比例和 ES 选择统一使用 MSE。
4. `legacy_maps.distill_weight` 当前为固定值，没有实现论文中的退火。
5. `fixtures/legacy_expert_cache.json` 是合成工程缓存，不能作为论文证据。
6. 现有 runner 没有独立的 `LLM-only`、`MAPS-w/o-Annealing` 实验模式。
7. episode 输出缺少论文所需的 P95 latency、DVR、TCR、Reward AUC 和 steps-to-threshold。
8. 没有 10/20/30/50 UE 的正式配置和统一评测矩阵。
9. 原始 LaTeX 仍使用错误的 Eq. (1)、Eq. (5)、wired energy、统一动作 MSE 和含糊 Algorithm 1。
10. 当前 MAPPO/HAPPO 也使用连续 ES selector，未修复前不能作为正式公平基线。

结论：Phase 2 是可信环境基础，但当前 `legacy_maps` 只能用于 smoke test，不能直接生成 EI 论文结果。

## 5. 代码修改方案

## E0：建立 EI 论文工作区

新增：

```text
paper/ei/
  main.tex
  references.bib
  figures/
configs/ei/
experiments/ei/
artifacts/ei/
docs/EI_CONFERENCE_REVISION_PLAN.md
```

实施：

- 从 `legacy/MAPS_original_latex_source.zip` 解压内容到 `paper/ei/`，旧归档不修改。
- 先保证原稿在本机可编译，再逐节修改。
- `docs/MAPS_REDEVELOPMENT_PLAN.md` 明确归属于 Letter 路线，不作为 EI 分支执行计划。

验收：

- EI LaTeX 独立编译；
- 所有论文图表仅从带 manifest 的 `artifacts/ei/` 生成；
- 不引用 Phase 1/2 smoke 指标作为论文结果。

## E1：修复混合动作表示

### 动作定义

每个 UE 的动作定义为：

```text
a_u = (phi_u, p_u)
phi_u = [phi_local, phi_edge, phi_cloud]
p_u   = categorical probabilities over E edge servers
```

actor 输出：

- 三个 partition logits，经 softmax 得到 `phi_u`；
- `E` 个 edge logits，经 softmax 得到 `p_u`；
- 执行时 `edge_id = argmax(p_u)`；
- 训练 actor 时保留 soft probability，使 centralized critic 对 edge head 可微。

MADDPG 的 replay action 统一保存 `[phi_local, phi_edge, phi_cloud, edge_one_hot...]`，不再保存归一化 edge 标量。环境边界使用统一 codec 转换为 `[phi_local, phi_edge, phi_cloud, edge_id]`。

推荐新增：

```text
algos/common/hybrid_action.py
```

职责：

- `HybridActionSpec(num_edges)`
- actor 输出解码；
- one-hot/probability 与环境 `edge_id` 互转；
- expert action 编码；
- shape、mask 和范围校验。

不建议在 EI 分支同时引入复杂的 Gumbel-Softmax 温度调度。MADDPG actor 更新可直接把 soft edge probabilities送入 critic，环境执行使用 argmax；必要时再增加 straight-through Gumbel-Softmax 作为实现细节。

验收：

- 不存在连续 edge scalar 乘 ES 数量后取整；
- 不硬编码 ES 数量；
- 3/5/10 个 ES 下 action shape 正确；
- policy、replay、expert 和 environment 使用同一 codec；
- 增加 actor gradient、codec round-trip、invalid action 单元测试。

## E2：实现 EI 版 MAPS 蒸馏和退火

### 混合蒸馏损失

连续任务比例：

```text
L_part = mean_u ||phi_u^RL - phi_u^LLM||_2^2
```

离散 ES 选择：

```text
L_edge = mean_u CE(p_u^RL, e_u^LLM)
```

总蒸馏损失：

```text
L_distill = L_part + eta_edge * L_edge
L_actor_total = L_actor_RL + lambda(k) * L_distill
```

其中 `k` 是 episode 或 environment step。论文和代码只选一种自变量，推荐使用 environment step 比例，以免不同 episode 长度破坏比较。

### 退火策略

为尽量保留原始 MAPS，采用可配置的三阶段退火：

```text
lambda(k) = lambda_high,  progress < 0.3
            lambda_low,   0.3 <= progress < 0.7
            0,            progress >= 0.7
```

默认值可从原稿的 `0.8/0.15/0` 开始，但必须通过小规模 pilot 检查梯度量级。若蒸馏损失与 policy loss 相差过大，应先归一化或调整权重，不能仅复制原数字。

正式实验模式：

- `maddpg`：`lambda = 0`，即 MAPS-w/o-LLM；
- `maps_no_annealing`：固定 `lambda = lambda_high`；
- `maps`：三阶段退火；
- `llm_only`：直接执行同一 expert cache，不训练 MARL。

`MAPS-w/o-Distill` 与 `MAPS-w/o-LLM` 在当前方法下等价，不重复画成两个看似独立的方法。

### 关于 LLM confidence

EI 最低成本版本不把 LLM 自报 confidence 直接乘入损失：

- 自报 confidence 未校准，容易引入新的审稿问题；
- 原稿创新主线是蒸馏和退火，不需要再增加机制；
- parser validity、缺失字段和 fallback 只用于生成 `expert_mask`。

如果后续确实加入 confidence，必须先做 calibration 或至少分桶验证“高 confidence 对应更低 expert cost”，否则只记录，不参与训练。

验收：

- partition 和 edge loss 可分别观察；
- `lambda=0` 时蒸馏项不产生 actor 梯度；
- fixed 与 annealed schedule 可由配置复现；
- 相同 expert cache 和 seed 下结果可复现；
- checkpoint 和 manifest 记录 loss 权重与 schedule。

## E3：建立可用于论文的 LLM 专家缓存

当前合成缓存必须替换为真实、可审计的离线专家数据。

### Prompt 输入

- 当前 UE 任务：data size、cycles、deadline slack、semantic type、priority、output ratio；
- UE CPU 和 UE-ES 速率；
- 各 ES CPU、队列负载；
- CS 负载和 backhaul 状态；
- 明确 local/edge/cloud 三比例约束和 ES id 范围；
- 明确只返回 JSON。

### 输出

```json
{
  "schema_version": "ei-v1",
  "actions": [
    {
      "ue_id": 0,
      "partition": {"local": 0.2, "edge": 0.6, "cloud": 0.2},
      "edge_id": 1,
      "reason_code": "latency_sensitive"
    }
  ]
}
```

### 缓存要求

- 正式训练不在线重复请求 LLM；
- 相同状态 id 始终读取相同响应；
- 保存 model name、prompt version、temperature、生成日期、原始响应 hash；
- 记录 parser success、fallback 和修复比例；
- API key 不进入配置、manifest 或 Git；
- 论文至少报告模型名、量化方式/服务方式、temperature、平均 token 数、平均离线查询时延和部署阶段 LLM 调用次数为 0。

为了控制成本，EI 版只使用一个 LLM。不要声称跨 LLM 泛化。第二个 LLM 仅作为有余力时的附加实验。

验收：

- cache 标记 `paper_evidence: true`；
- parser success rate 可统计；
- 缓存覆盖正式训练所需状态或采用可解释的稀疏查询策略；
- `llm_only`、`maps_no_annealing` 和 `maps` 共享同一缓存。

## E4：补充论文指标与评测协议

### 每任务指标

- Mean latency；
- P95 latency；
- device energy；
- system energy；
- deadline violated；
- completed；
- completed on time。

### 明确定义

设生成任务数为 `N_gen`，在评测窗口结束前完成的任务数为 `N_done`，按时完成数为 `N_on_time`：

```text
TCR = N_done / N_gen
DVR = (N_gen - N_on_time) / N_gen
On-time completion rate = N_on_time / N_gen
```

这样 DVR 包含超时完成和窗口结束时仍未完成的任务。若使用不同定义，代码和论文必须一致。

episode 结束后应设置固定 drain horizon，让已进入系统的任务继续执行但不再产生新任务；超过 drain horizon 的任务计为未完成。否则 TCR 会受 episode 截断方式影响。

### 收敛效率

- Reward AUC：对 environment steps 归一后的 reward 曲线积分；
- Steps to threshold：首次达到并连续保持目标阈值 `K` 个 evaluation points 的环境步数；
- Final reward：最后 10% 评测点均值；
- 不再仅凭曲线目测“第 2000 episode 收敛”。

### 统计

- 正式结果至少 5 个 seeds；
- 同一 seed 下所有方法共享任务到达、信道和初始状态序列；
- 报告 mean、standard deviation 和 95% confidence interval；
- 曲线画均值与置信带；
- 结果表保留原始物理单位，不只给 normalized reward。

验收：

- 可从 raw JSON/CSV 一键重新生成表格和图片；
- 指标定义有单元测试；
- 所有方法使用相同 environment-step budget。

## E5：正式实验矩阵

### A. 主实验与消融

固定 `U=10, E=5, C=1`：

| 方法 | LLM prior | Distillation | Annealing |
|---|---:|---:|---:|
| LLM-only | yes | no | no |
| MADDPG / MAPS-w/o-LLM | no | no | no |
| MAPS-w/o-Annealing | yes | mixed | fixed |
| MAPS | yes | mixed | yes |

回答：

- LLM prior 是否加速早期学习？
- 固定信任 LLM 是否损害后期策略？
- 退火是否保留早期收益并避免后期受限？

### B. 可靠性实验

在相同主场景报告：

- Mean/P95 latency；
- DVR；
- TCR；
- device/system energy。

可增加三个 deadline 强度档位作为高性价比附加实验：

```text
loose / medium / strict
```

### C. 扩展性实验

最低要求：

```text
U = 10, 20, 30, 50
E = 5
C = 1
```

固定 ES 数量用于形成逐渐拥塞的压力测试，论文必须明确这不是“保持负载不变”的线性扩展实验。若资源允许，再补一组比例扩展：

```text
(U,E) = (10,5), (20,5), (30,10), (50,10)
```

主比较至少使用 MAPS 和 MADDPG；若运行预算允许，四种方法全部覆盖。

### D. 推荐但非最低要求的增强基线

在核心实验稳定后，优先修复 MAPPO 的 categorical edge head，并增加 MAPPO 作为异构 MARL 基线。这比实现 QMIX、VDN、QPLEX 等离散值分解算法更符合当前连续划分问题，也能回应“只有一个 DRL 算法”的质疑。

该项属于推荐增强项，不阻塞 EI 最低投稿包。未修复混合动作前，现有 MAPPO/HAPPO 结果不得进入论文。

### 不做

- 不跑 100/500 UE 主实验；
- 不实现 CV-CED、verifier gating 或 counterfactual credit；
- 不比较四种 LLM；
- 不宣称 topology generalization；
- 不证明深度 MADDPG 的全局收敛；
- 不把 smoke run 当正式实验；
- 不保留旧版不可信曲线或手工生成结果。

## 6. 论文逐节修改方案

## 6.1 Abstract

按五句结构重写：

1. TEC 中动态任务流需要细粒度划分与并行调度；
2. MARL 能适应动态环境，但早期探索需要大量交互，在线 LLM 又存在时延开销；
3. 本文开发 MAPS，由 LLM 离线生成专家先验，通过 mixed-action distillation 和 annealing 指导 MADDPG；
4. 描述修正后的 TEC 仿真和实验维度；
5. 仅填入正式实验支持的数字。

删除 AGI、`innovative`、跨领域推广和未经验证的 generalization 表述。

## 6.2 Introduction

推荐五段：

1. TEC 任务分割与并行执行的应用需求；
2. 现有 DRL/MARL offloading 与 fine-grained partitioning 工作；
3. LLM-guided RL 的机会和局限：可提供先验，但在线调用慢、建议并非总是最优；
4. MAPS 的设计：offline prior、mixed-action distillation、annealed transfer、deployment without LLM；
5. 三条贡献。

“DRL sample inefficiency/high exploration cost”必须引用经过核验的 LLM4Teach、LINVIT 或 RL 文献，不能作为无引用常识句。

## 6.3 Related Work

EI 长文应单设 Related Work，分三组：

1. TEC/MEC task offloading and task partitioning；
2. MARL for end-edge-cloud scheduling；
3. LLM-guided RL and LLM-enhanced edge scheduling。

必须加入并准确区分：

- LLM4Teach：policy teacher；
- LINVIT：value-based regularization；
- LLM-assisted multi-cloud scheduling；
- LLM-QTRAN：UAV region decomposition；
- LeDRL：semantic prior + MAPPO for collaborative edge computing；
- COMLLM：LLM directly trained for multi-step MEC offloading。

最后用一段边界说明：MAPS 不试图提出通用 LLM-RL 算法，而聚焦 TEC 三级细粒度并行划分和训练期先验迁移。

## 6.4 System Model

必须整体重写公式，保持与 Phase 2 代码一致。

### Task model

任务写为：

```text
pi_u(t) = (d_u, l_u, D_u, rho_u, omega_u, p_u, a_u)
```

分别表示数据量、cycles、relative deadline、output ratio、semantic type、priority 和 arrival time。

允许 Poisson 多任务到达；每个 UE 每时隙最多为队首任务作一次调度决策，但后续任务保留在 admission queue。明确这不等于“每个 UE 只产生一个任务”。

### Rate model

替换原 Eq. (1)：

```text
R_ue(t) = B_ue log2(1 + P_u h_ue(t) / (N0 B_ue + I_ue(t)))
```

EI 主实验采用 orthogonal access 时写 `I_ue(t)=0`，并说明路径增益模型。

### Queue model

队列主量纲统一为 cycles：

```text
Q_n(t+1) = [Q_n(t) - f_n Delta t]^+ + A_n(t)
```

同时解释实现实际保存 task objects，以便计算 release time、deadline 和 FCFS completion。

### Backhaul and downlink

- UE-ES 是无线链路；
- ES-CS 是有线回传，时延为 data/rate + propagation；
- 回传能耗为 `epsilon_bh * bits`，不使用虚构的 cloud RF power；
- 返回数据量为 `rho_u d_u`；
- edge 返回路径为 ES-UE，cloud 返回路径为 CS-ES-UE。

### Computation energy

```text
E_n^cmp = kappa_n L_n f_n^2
```

给出 `kappa` 单位、频率单位，并加入经核验的 MEC/DVFS 文献。论文同时区分 device energy 和 system energy。

### Parallel latency

```text
T_u = max(T_u^local, T_u^edge, T_u^cloud)
```

每条分支包含 admission wait、communication release、queue wait、computation 和 result return。

### Objective

先分别用尺度常数归一化：

```text
J = w_T T_norm + w_E E_norm + w_D DVR
```

权重使用 `w_T, w_E, w_D`。不要再让 objective 和 reward 同时使用含义不清的 `alpha`。

## 6.5 Dec-POMDP and Method

明确：

- 每个 UE 是 agent；
- local observation 包含自身任务、队列、CPU、各 ES rate/load 摘要、cloud/backhaul 摘要；
- centralized critic 在训练时接收 joint state/action；
- actor 在部署时只使用 local observation；
- target networks 用于稳定 Bellman target；
- exploration noise 只加在训练动作，写明 OU noise 参数和衰减/关闭时机。

### LLM 与 MDP 的连接

写清楚：

```text
a_u^LLM ~ pi_LLM(. | prompt(o_u, g))
```

LLM action 是由当前 observation/global summary 条件化得到的 training prior，不是环境转移函数的一部分，也不在部署时参与 policy inference。它仅通过 replay 中保存的 expert label 进入 actor loss。

### Algorithm 1

算法应至少包括：

1. 初始化 actor、critic、target networks、replay buffer 和 expert cache；
2. 重置环境并观察 state；
3. 按 state id 读取或生成 LLM expert prior；
4. 各 agent 生成 partition probabilities 和 edge probabilities；
5. 添加探索并通过 hybrid codec 得到环境动作；
6. 执行动作并保存 joint transition 与 expert label；
7. 采样 mini-batch；
8. 使用 Bellman loss 更新 critic；
9. 计算 `L_part`、`L_edge` 和 `lambda(k)`；
10. 使用 RL loss + annealed mixed distillation 更新 actor；
11. soft-update target networks；
12. 训练结束后仅部署 actor。

删除“Use noise strategies”和“Synchronize network parameters between cloud and edge agents”等未定义表述。

## 6.6 Convergence discussion

不尝试证明深度非凸 MADDPG 的全局收敛。改为：

- 给出 actor/critic 更新公式；
- 说明 target network、replay buffer 和 bounded reward 的稳定化作用；
- 用 5 seeds、Reward AUC、steps-to-threshold 和置信区间做经验收敛评价；
- 在 limitations 中承认不提供全局收敛保证。

这比给出不成立的数学证明更严谨。

## 6.7 Experiments

实验节按研究问题组织：

- RQ1：MAPS 是否减少达到目标性能所需的环境交互？
- RQ2：退火是否优于固定蒸馏？
- RQ3：MAPS 是否改善 deadline reliability？
- RQ4：UE 数量增长时性能如何变化？
- RQ5（可选）：相较 MAPPO 是否仍有早期学习优势？

推荐图表：

| 编号 | 内容 |
|---|---|
| Fig. 1 | TEC system and parallel task branches |
| Fig. 2 | MAPS training and deployment workflow |
| Fig. 3 | Reward vs environment steps with 95% CI |
| Fig. 4 | Latency/energy training curves or steps-to-threshold |
| Fig. 5 | Scalability for U=10/20/30/50 |
| Table I | System and training parameters |
| Table II | Mean/P95 latency, energy, DVR, TCR |
| Table III | Ablation and LLM overhead |

Normalized Reward 必须给出公式、归一化范围和参考值。正文结果优先解释物理指标，不以 reward 代替系统性能。

## 6.8 Conclusion and Limitations

结论只总结：

- MAPS 是一种面向 TEC 混合动作调度的 LLM-guided MARL framework；
- LLM 在训练阶段提供先验，部署阶段不调用；
- 正式实验支持的收敛、时延、能耗、可靠性和规模结论。

限制：

- 单一 LLM；
- 仿真环境；
- 正交接入和静态路径损耗；
- 规模只验证到 50 UE；
- 无全局收敛保证；
- 未证明跨场景泛化。

删除“can be extended to other areas”和未经实验支持的未来效果承诺。

## 7. 审稿意见到修改动作的映射

| 审稿意见 | EI 分支动作 |
|---|---|
| LLM + RL 新颖性有限 | 降低创新措辞，定位为 TEC fine-grained mixed-action framework |
| 场景仅 10 UE/5 ES | 增加 U=10/20/30/50 扩展实验 |
| 只有 MADDPG 和一个 LLM | 最低包通过消融增强证据；推荐增加修复后的 MAPPO，不声称跨 LLM |
| Eq. (1) 缺 channel gain/N0B | 论文同步 Phase 2 Shannon/path-loss 模型 |
| Eq. (5) 量纲错误 | 引入 Delta t，queue 统一为 cycles |
| wired energy 错误 | 使用独立 backhaul rate/propagation/energy-per-bit |
| downlink 被忽略 | 使用 output ratio 和 edge/cloud return path |
| mixed action 统一 MSE | partition MSE + edge CE |
| alpha 重名 | objective/reward 使用 `w_T,w_E,w_D` |
| 每 UE 最多一个任务 | Poisson 多到达 + pending admission queue |
| Algorithm 1 太抽象 | 写出 query/cache、action、replay、critic、mixed loss、annealing 和 target update |
| LLM 与 MDP 连接不清 | 将 LLM action 定义为 observation-conditioned training prior |
| actor/critic/target 未解释 | 增加 CTDE 和更新公式 |
| 要求数学收敛证明 | 不过度承诺，改用经验收敛协议并声明理论边界 |
| Normalized Reward 未定义 | 给出归一化公式，主表同时报告物理指标 |
| LLM prompt/latency/overhead 缺失 | 报告 prompt schema、tokens、offline query latency、cache 和 zero deployment calls |
| 语法和符号错误 | 重建 notation table，逐式核对，不做局部补丁式修词 |

## 8. 推荐实施顺序与停止条件

### Sprint EI-1：方法正确性

- 建立 `paper/ei/`；
- 实现 hybrid action codec；
- 修改 MADDPG actor/critic/replay；
- 实现 mixed distillation；
- 实现 fixed/annealed/no-LLM 三种配置；
- 增加单元测试。

停止条件：四种 EI 方法可在 Phase 2 环境完成短训练，loss 有限且同 seed 可复现。

### Sprint EI-2：论文证据链

- 生成真实 LLM expert cache；
- 实现 LLM-only；
- 增加 P95、DVR、TCR、AUC、threshold metrics；
- 增加 drain horizon；
- 建立 10/20/30/50 UE 配置；
- 跑 2 seeds pilot，确定 episode/step budget 和蒸馏权重。

停止条件：pilot 能回答 RQ1-RQ4；如果 MAPS 没有早期优势，先检查 expert quality 和 loss scale，不直接跑大矩阵。

### Sprint EI-3：正式实验

- 5 seeds 主实验；
- 5 seeds 消融；
- 5 seeds 规模实验；
- 可选 MAPPO；
- 统计分析和 publication figures。

停止条件：每个论文 claim 都能映射到 raw artifact、配置和图表。

### Sprint EI-4：论文修改

- 重写 Abstract/Introduction/Related Work；
- 用 Phase 2 公式重写 System Model；
- 重写 Dec-POMDP、MAPS 和 Algorithm 1；
- 写 Results by RQ；
- 写 Limitations；
- 统一符号、单位、标题、图注和引用；
- 完成匿名化与模板检查。

## 9. 预计成本

按单人开发、不计 GPU 和 LLM API 排队：

| 工作 | 预计时间 |
|---|---:|
| Hybrid action + mixed loss + tests | 3-5 天 |
| EI runner/metrics/configs | 2-4 天 |
| LLM cache 与 prompt 审计 | 1-3 天 |
| Pilot 与参数修正 | 2-4 天 |
| 正式实验 | 3-7 天 |
| LaTeX 重写与图表 | 4-7 天 |

总计约 3-4 周。若省略 MAPPO 和第二个 deadline sensitivity 实验，可控制在约 2-3 周。

## 10. 最终投稿最低完成定义

### 代码

- mixed action 不再通过连续 edge scalar 取整；
- partition MSE + edge CE；
- no-LLM/fixed/annealed/LLM-only 四种方法可统一运行；
- 正式 expert cache 可审计；
- 关键逻辑有测试；
- 所有运行带 seed、config、commit 和 manifest。

### 实验

- 至少 5 seeds；
- 消融包含 MADDPG、MAPS-w/o-Annealing、MAPS、LLM-only；
- 报告 mean/P95 latency、device/system energy、DVR、TCR；
- U=10/20/30/50；
- 收敛按 environment steps、AUC 和 threshold 定义；
- 所有方法共享 workload 和 expert cache。

### 论文

- 不再过度声称创新、泛化或理论收敛；
- 所有系统公式与代码一致；
- Algorithm 1 可复现；
- LLM 与 Dec-POMDP 的连接明确；
- 讨论 LLM prompt、离线开销和部署零调用；
- 每个数值结论来自正式 artifact；
- 相关工作更新到投稿前，且引用元数据和支持的 claim 均经核验。

## 11. 决策记录

### D-EI-1：保留原始 MAPS，还是采用 Letter 的 CV-CED？

决定：保留原始 MAPS 的 expert distillation + annealing 主线。

理由：EI 路线目标是低成本形成技术严谨、证据完整的会议论文。CV-CED 会改变研究问题、算法和实验矩阵，属于 `publication/letter-revision`。

### D-EI-2：是否加入 LLM confidence weighting？

决定：最低包不加入。

理由：未经校准的自报 confidence 不能证明可靠性，并会增加新的消融负担。EI 版只使用 parser validity mask；confidence 可记录但不参与损失。

### D-EI-3：是否必须实现多个 MARL baseline？

决定：最低包不阻塞于多个 MARL；MAPPO 是优先增强项。

理由：现有 MAPPO 的 edge action 同样不正确，直接使用会造成不公平比较。先完成 MAPS 技术正确性和四种核心消融，再决定是否投入 2-4 天修复 MAPPO。

### D-EI-4：是否保留旧实验曲线？

决定：不保留为论文证据。

理由：旧曲线来自旧物理模型、旧 MADDPG 实现和不完整指标，无法与 Phase 2 可信环境对应。可作为历史记录保留在 `legacy/`，但正式论文图表全部重跑。

## 12. 调研来源与引用状态

以下来源用于确定研究边界。正式写入 `references.bib` 前仍需按标题、作者、年份、venue、DOI/arXiv ID 和具体支持句逐项核验，不能根据本计划直接手写 BibTeX。

### 已发表或已有 DOI

1. Y. Cao et al., “Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods,” *IEEE TNNLS*, 2025. DOI: <https://doi.org/10.1109/TNNLS.2024.3497992>.
2. X. Tang et al., “LLM-Assisted Reinforcement Learning: Leveraging Lightweight Large Language Model Capabilities for Efficient Task Scheduling in Multi-Cloud Environment,” *IEEE Transactions on Consumer Electronics*, 2024. DOI: <https://doi.org/10.1109/TCE.2024.3524612>.
3. “Task Offloading With LLM-Enhanced Multi-Agent Reinforcement Learning in UAV-Assisted Edge Computing,” *Sensors*, 2025. DOI: <https://doi.org/10.3390/s25010175>.
4. H. She et al., “Efficient End-Edge-Cloud Task Offloading in 6G Networks Based on Multiagent Deep Reinforcement Learning,” *IEEE Internet of Things Journal*, 2024. DOI: <https://doi.org/10.1109/JIOT.2024.3372614>.
5. Y. Li et al., “Joint Task Partitioning and Parallel Scheduling in Device-Assisted Mobile Edge Networks,” *IEEE Internet of Things Journal*, 2024. DOI: <https://doi.org/10.1109/JIOT.2023.3341062>.

### 邻近预印本

1. Y. Hao et al., “LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing,” arXiv:2605.05727, submitted May 8, 2026: <https://arxiv.org/abs/2605.05727>.
2. “Multi-Turn Reasoning LLMs for Task Offloading in Mobile Edge Computing,” arXiv:2604.07148, 2026: <https://arxiv.org/abs/2604.07148>.
3. J. Hu et al., “Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents,” arXiv:2311.13373: <https://arxiv.org/abs/2311.13373>.
4. S. Zhang et al., “How Can LLM Guide RL? A Value-Based Approach,” arXiv:2402.16181: <https://arxiv.org/abs/2402.16181>.

2026 年文献目前多为近期预印本。论文中可以用它们说明最新研究趋势，但应明确其预印本状态，并在实际投稿前重新检查是否已有正式发表版本。
