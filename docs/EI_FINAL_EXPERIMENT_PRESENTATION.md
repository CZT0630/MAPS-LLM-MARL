# MAPS EI 论文最终实验与结果呈现方案

> 适用分支：`publication/ei-conference`
> 制定日期：2026-06-13
> 状态：EI 实验的最终执行与论文呈现依据

## 1. 实验叙事

实验只回答五个问题：

1. **RQ1：** LLM 专家先验能否减少 MAPS 达到目标性能所需的环境交互？
2. **RQ2：** 退火机制是否优于始终保持固定专家权重？
3. **RQ3：** 训练后的 MAPS 是否改善 TEC 任务的尾部时延、能耗和 deadline 可靠性？
4. **RQ4：** 当 UE 数量增加、资源竞争增强时，MAPS 的性能是否仍稳定优于无 LLM 的学习基线？
5. **RQ5：** LLM 专家动作是否具有可用质量，其离线生成成本与部署成本分别是多少？

正文中的每张图和每张表必须对应其中一个问题。不能回答这些问题的训练日志不进入论文。

## 2. 最终实验场景

### 2.1 网络拓扑

基础场景固定为：

```text
10 UEs + 5 ESs + 1 CS
```

每个 UE 是一个 agent。任务可被划分为：

```text
local / edge / cloud
```

每个动作同时包含：

- 连续任务比例 `phi = [phi_local, phi_edge, phi_cloud]`；
- 离散 ES 选择 `edge_id`。

总任务时延是三条并行执行分支完成时间的最大值。环境使用 Phase 2 的持续队列、无线链路、有线回传、计算能耗和下行结果返回模型。

### 2.2 任务流

正式实验使用：

- Poisson 动态到达；
- `video_analytics`、`ar_vr`、`ai_inference`、`control` 四类任务；
- data size、CPU cycles、deadline、priority 和 output ratio；
- 每个 UE 可在同一时隙产生多个任务，未调度任务保留在 admission queue；
- episode 结束后进入 drain horizon，不再生成任务，但继续处理队列。

测试集使用预先生成并冻结的 scenario bank。所有方法在相同 test seed 下共享：

- 到达序列；
- 任务属性；
- UE-ES 距离和信道状态；
- 初始队列；
- deadline 分布。

### 2.3 三类实验场景

#### S1：基础场景

```text
U = 10, E = 5, C = 1
deadline = medium
```

用于收敛、消融、主性能和专家质量实验。

#### S2：拥塞扩展场景

```text
U = {10, 20, 30, 50}
E = 5
C = 1
```

固定 ES 数量，让 UE 增加形成逐渐增强的资源竞争。该实验是 congestion stress test，不称为保持资源比例不变的线性扩展。

所有规模下报告每任务指标，不能比较未经归一化的总系统能耗。

#### S3：deadline 压力场景

```text
deadline factor = {loose, medium, strict}
U = 10, E = 5, C = 1
```

主策略在 medium deadline 下训练，再在三个冻结测试集上评估。该实验只称为 deadline sensitivity 或 stress test，不宣称跨分布泛化。

若目标会议页数严格，S3 放入附录或补充材料，不删除原始结果。

## 3. 最终 baseline 设计

### 3.1 正文方法集合

| 方法 | 类别 | LLM | 学习 | 作用 |
|---|---|---:|---:|---|
| Greedy-MinCost | 模型启发式 | no | no | 说明是否需要长期学习 |
| LLM-only | 语言模型决策 | yes | no | 说明 LLM 是否能独立实时调度 |
| MADDPG | MARL backbone | no | yes | 隔离 LLM 指导的价值 |
| MAPPO | 异构 MARL | no | yes | 避免只和同一算法家族比较 |
| MAPS-w/o-Annealing | 消融 | yes | yes | 检验固定专家权重的影响 |
| MAPS | 完整方法 | yes | yes | 混合动作蒸馏 + 退火 |

### 3.2 Greedy-MinCost 定义

Greedy-MinCost 不是最优解，也不称为 upper bound。它是使用完整环境快照的一步模型启发式。

实施方式：

1. 按 deadline slack 从小到大处理 active UE；
2. 对每个 UE 枚举有限的 partition template 和所有 ES；
3. 保持其他已选择动作不变，调用无副作用的 `evaluate_action()`；
4. 选择预测综合成本最低的候选动作；
5. 报告其决策 wall-clock，体现模型搜索开销。

综合成本与论文目标一致：

```text
J = w_T * T_norm + w_E * E_norm + w_D * violation
```

候选 partition template 在所有实验中固定并写入配置。不能为不同场景手工调整候选集。

### 3.3 LLM-only 定义

LLM-only 对每个状态直接采用同一 prompt 和同一 LLM 生成的动作，不训练 MARL。

- temperature 固定为 0 或可复现低值；
- 首次查询保存 state-keyed cache；
- 性能来自对应状态的真实 LLM 输出，不使用固定的通用动作；
- 论文同时报告 uncached query latency 和 cached replay latency；
- LLM-only 不出现在“训练收敛曲线”中，因为它没有训练过程。

### 3.4 MADDPG

MADDPG 与 MAPS 共享：

- actor/critic 网络；
- hybrid action codec；
- replay buffer；
- reward；
- optimizer；
- environment-step budget；
- scenario bank。

唯一差别是：

```text
lambda_distill = 0
```

因此：

```text
MADDPG = MAPS-w/o-LLM = MAPS-w/o-Distillation
```

论文只保留名称 `MADDPG`，不能把三个等价名称重复画成三条曲线。

### 3.5 MAPPO

MAPPO 是正式必需的第二类 MARL baseline，但必须先：

- 使用与 MAPS 相同的 partition softmax；
- 使用 categorical ES head；
- 使用相同 local observation 和 global state；
- 使用相同 environment-step budget；
- 调整网络规模，使参数量与 MADDPG/MAPS 同数量级；
- 单独报告 on-policy 与 off-policy 的 wall-clock 差异，不把它解释为样本效率。

当前连续 edge selector 的 MAPPO 不得进入论文。

### 3.6 MAPS-w/o-Annealing

该方法保留 LLM mixed-action distillation，但：

```text
lambda(k) = lambda_fixed
```

它只回答“退火是否必要”。`lambda_fixed` 应由 pilot 确定并锁定，不能根据正式结果反复选择。

### 3.7 MAPS

完整 MAPS 使用：

```text
L_distill = L_partition_MSE + eta_edge * L_edge_CE
L_actor = L_RL + lambda(k) * L_distill
```

其中 `lambda(k)` 按 environment-step progress 退火。部署评估时：

- 关闭 exploration noise；
- 不调用 LLM；
- 只运行每个 UE 的 actor。

## 4. 不进入正文的基线与原因

### 保留为工程 sanity check，不作为主 baseline

- `All-Local`：验证环境和本地计算公式；
- `Random`：验证 LLM 专家质量是否优于随机动作；
- `All-Edge`：可用于调试 edge queue，但缺乏合理调度能力。

它们可以保存在原始结果中，正文不画曲线。若 Greedy 或学习方法出现异常，才在附录披露用于诊断。

### EI 分支不新增

- HAPPO；
- QMIX、VDN、QPLEX、QTRAN；
- 多个 LLM；
- CV-CED、verifier gating、counterfactual credit；
- 100/500 UE 主实验。

原因：

- HAPPO 与 MAPPO 对当前 EI 叙事提供的信息高度重复；
- value decomposition 方法需要额外离散化连续 partition，比较不自然；
- 多 LLM 和 CV-CED 属于 Letter 路线；
- 过多 baseline 会增加调参不公平和实验成本，却不回答新的核心问题。

## 5. 最终指标体系

## 5.1 Primary metrics

论文把以下四项设为 primary metrics：

1. `Reward AUC`：训练期样本效率，越高越好；
2. `P95 task latency`：部署期尾部时延，越低越好；
3. `DVR`：deadline violation ratio，越低越好；
4. `system energy per completed task`：每个完成任务的系统能耗，越低越好。

这四项分别覆盖学习效率、时延可靠性、任务可靠性和资源成本。

## 5.2 Secondary metrics

- environment steps to threshold；
- final evaluation reward；
- mean task latency；
- TCR；
- device energy per completed task；
- actor inference latency；
- training wall-clock；
- LLM query latency、token count、parser success 和 fallback rate。

## 5.3 指标定义

设：

- `N_gen`：评测窗口内生成的任务；
- `N_done`：在 drain horizon 结束前完成的任务；
- `N_on_time`：在 deadline 前完成的任务。

```text
TCR = N_done / N_gen
DVR = (N_gen - N_on_time) / N_gen
```

`1-DVR` 与 on-time completion rate 完全等价，因此正文不再同时展示二者。

系统能耗使用：

```text
system energy per completed task
= total UE/ES/CS/backhaul energy / max(N_done, 1)
```

设备能耗仅作为 secondary metric。正文不能把 device energy 和 system energy 混写成同一个 “energy consumption”。

## 5.4 收敛定义

训练曲线横轴只使用：

```text
environment interactions
```

不以 episode 数量作为主要样本效率单位。

threshold 由 MADDPG 最后 10% evaluation points 的平均 reward 确定。某方法首次达到该 threshold，并连续保持 `K` 个 evaluation points，记为 steps to threshold。

若某方法未达到 threshold，表格写 `not reached`，不能用最大训练步数替代。

Reward AUC 在相同 environment-step 区间上计算，并按横轴长度归一化。

## 6. 最终实验矩阵

## E1：训练收敛与机制消融

场景：S1。

方法：

```text
MADDPG
MAPPO
MAPS-w/o-Annealing
MAPS
```

指标：

- evaluation reward curve；
- Reward AUC；
- steps to threshold；
- final reward。

回答 RQ1 和 RQ2。

Greedy 和 LLM-only 没有训练过程，不放入收敛曲线。

## E2：独立测试集主性能

场景：S1，训练完成后在冻结 scenario bank 上评估。

方法：

```text
Greedy-MinCost
LLM-only
MADDPG
MAPPO
MAPS-w/o-Annealing
MAPS
```

指标：

- mean latency；
- P95 latency；
- system energy per completed task；
- device energy per completed task；
- DVR；
- TCR；
- action decision latency。

回答 RQ3。

## E3：UE 规模与拥塞

场景：S2。

正文方法：

```text
Greedy-MinCost
MADDPG
MAPPO
MAPS
```

不再重复 `MAPS-w/o-Annealing` 和 `LLM-only`：

- 退火作用已由 E1/E2 回答；
- LLM-only 在大规模场景查询成本高，且不能体现部署策略扩展性。

指标：

- P95 latency；
- DVR；
- system energy per completed task；
- TCR；
- actor/decision latency。

回答 RQ4。

每个规模分别训练对应策略，不能把 10-UE critic 直接当作 50-UE 训练模型。若只迁移 actor 做测试，必须另列为 cross-scale stress test，不能与重新训练结果混合。

## E4：deadline sensitivity

场景：S3。

方法：

```text
Greedy-MinCost
MADDPG
MAPPO
MAPS
```

指标：

- P95 latency；
- DVR；
- TCR。

该实验使用 medium 场景训练好的策略，在 loose/medium/strict test banks 上评估。结果描述为 stress sensitivity，不使用 “generalization improvement”。

篇幅不足时放附录。

## E5：LLM 专家质量与系统开销

场景：从 S1 的冻结 state bank 采样至少 500 个状态。

动作来源：

```text
Random
All-Local
Greedy-MinCost
LLM expert
```

指标：

- JSON/parser success rate；
- hard-constraint valid rate；
- one-step normalized cost；
- cost improvement over Random；
- cost gap to Greedy；
- average input/output tokens；
- uncached query latency；
- cache hit rate；
- expert cache generation time；
- deployed MAPS LLM calls，固定为 0；
- actor inference latency。

回答 RQ5。

LLM expert 不必优于 Greedy，但必须：

- 显著优于 Random/All-Local 中至少一个弱参考；
- 具有足够高的解析和约束有效率；
- 在 pilot 中实际带来早期 Reward AUC 收益。

若这些条件不成立，应改 prompt 或降低论文对 LLM expert quality 的表述，不能直接进入正式大实验。

## 7. 重复次数与统计单位

### 7.1 训练

- E1/E2：至少 5 个独立 training seeds；
- E3：每个规模至少 5 个 training seeds；
- E4：复用 E2 的 5 组训练 checkpoint；
- E5：使用冻结 state bank，不把同一状态上的多个候选动作当独立训练重复。

### 7.2 测试

每个训练 checkpoint 在相同的 20 个 test scenario seeds 上评估，并保证
每个 checkpoint 的测试集中至少生成 1000 个任务。若 P95 在 bootstrap
重采样下仍明显不稳定，则增加 test scenarios，而不是增加训练 seeds。

P95 latency 只对已完成任务计算，并与 TCR、DVR 同时报告；不能只报告
“成功完成任务的低时延”而隐藏未完成任务。

统计时：

1. 先对每个 training seed 的所有测试场景求均值；
2. 再以 training seed 为独立统计单位；
3. 不能把每个 task、episode 或 timestep 伪装成独立样本扩大样本量。

Greedy 和 LLM-only 没有 training seed，报告 test-scenario bootstrap CI；与学习方法的主要显著性结论只围绕 MAPS、MADDPG、MAPPO 和消融展开。

### 7.3 报告

- mean；
- standard deviation；
- 95% confidence interval；
- MAPS 相对 MADDPG 的 paired effect；
- MAPS 相对 MAPS-w/o-Annealing 的 paired effect。

5 seeds 的统计功效有限，因此正文以 CI 和 effect size 为主，不把单个 `p<0.05` 作为核心证据。若希望做正式非参数显著性检验，核心 E1/E2 增加到至少 8-10 seeds。

### 7.4 预计运行数量

不计 pilot：

```text
E1/S1 learning runs:
4 methods * 5 seeds = 20 runs

E3 additional scale runs:
3 learning methods * 3 new scales * 5 seeds = 45 runs

Total formal training runs:
65 runs
```

其中：

- S1 的 MADDPG、MAPPO 和 MAPS checkpoint 复用于 E2、E3 的 `U=10` 和 E4；
- MAPS-w/o-Annealing 只在 S1 做机制消融，不重复跑全部规模；
- Greedy-MinCost 和 LLM-only 只评估，不产生训练 run；
- E4 复用 medium 场景训练 checkpoint，不额外训练；
- pilot 预计 `4 methods * 2 seeds = 8` 个短 run。

若 50-UE 训练成本不可接受，先减少训练步数做容量验证，不能在正式结果中
只保留运行成功的方法。

## 8. 公平比较约束

所有学习方法必须共享：

- 相同 training scenario bank；
- 相同 test scenario bank；
- 相同 environment-step budget；
- 相同 observation；
- 相同 reward；
- 相同 hybrid action semantics；
- 相同 evaluation interval；
- 相近 actor/critic 参数量；
- 相同硬件和精度；
- 相同随机种子集合。

MAPS 和 MAPS-w/o-Annealing 共享：

- 同一 LLM model；
- 同一 prompt version；
- 同一 state-keyed cache；
- 同一 parser 和 fallback；
- 同一 distillation loss；
- 只改变 annealing schedule。

不能通过以下做法制造优势：

- MAPS 使用更多环境交互；
- MAPS 使用更多 hidden units；
- 为不同方法分别选择最有利 reward；
- 只删除 MAPS 表现差的 seeds；
- 用训练期带 exploration noise 的指标作为部署性能；
- 用不同任务序列比较算法。

## 9. 论文最终图表

## Fig. 1：TEC system model

内容：

- UE/ES/CS；
- local/edge/cloud 三条并行分支；
- wireless uplink/downlink；
- wired backhaul；
- queues 和 deadline。

这是系统模型图，不承载实验结果。

## Fig. 2：MAPS training and deployment workflow

训练：

```text
state -> LLM prior -> replay
state -> actor -> environment
RL loss + mixed distillation -> actor update
annealing -> reduce expert weight
```

部署：

```text
local observation -> actor -> hybrid scheduling action
```

明确标注 deployment 不调用 LLM。

## Fig. 3：Learning efficiency and ablation

### Fig. 3(a)

- x：environment interactions；
- y：evaluation reward；
- curves：MADDPG、MAPPO、MAPS-w/o-Annealing、MAPS；
- 5 seeds mean + 95% CI。

### Fig. 3(b)

- Reward AUC 或 steps to threshold；
- 使用点图/条形图加 CI；
- 不使用双纵轴。

该图回答 RQ1/RQ2。

## Fig. 4：Independent-test reliability

推荐两个子图：

- Fig. 4(a)：P95 latency；
- Fig. 4(b)：DVR。

方法：

```text
Greedy, LLM-only, MADDPG, MAPPO, no-annealing, MAPS
```

mean latency、energy 和 TCR 的精确值放 Table II，避免图表重复。

## Fig. 5：Scalability under increasing UEs

三个子图：

- Fig. 5(a)：P95 latency vs UE；
- Fig. 5(b)：DVR vs UE；
- Fig. 5(c)：system energy per completed task vs UE。

方法：

```text
Greedy, MADDPG, MAPPO, MAPS
```

该图回答 RQ4。

## Fig. 6：Deadline sensitivity

两个子图：

- DVR vs deadline factor；
- TCR vs deadline factor。

若正文篇幅紧张，整图移至附录。

## Table I：Simulation and training parameters

包括：

- topology；
- task distribution；
- wireless/backhaul；
- computation and energy coefficients；
- deadline factors；
- reward weights；
- actor/critic architecture；
- learning rate、batch、gamma、tau；
- annealing schedule；
- seeds、training steps、test scenarios。

## Table II：Independent-test main results

列：

```text
Method
Mean latency
P95 latency
System energy/completed task
DVR
TCR
Decision latency
```

device energy 放附录，避免主表过宽。

## Table III：Sample efficiency and LLM overhead

上半部分：

```text
MADDPG / MAPPO / no-annealing / MAPS
Reward AUC
Steps to threshold
Final reward
Training wall-clock
```

下半部分或独立小表：

```text
Parser success
Valid action rate
Tokens/query
Uncached LLM latency
Actor latency
Deployment LLM calls
```

如果目标模板不适合上下分区，则把 LLM overhead 放入附录 Table IV。

## 10. 从旧论文删除的实验展示

旧 `Reward.pdf`、`Latency.pdf` 和 `Energy.pdf` 不复制到 EI 论文目录，不作为新结果复用。

删除或停止展示：

1. 旧版手工判断“2000/5000 episodes 收敛”的曲线解释；
2. 未定义归一化方式的 Normalized Reward；
3. latency 随训练 episode 的曲线；
4. energy 随训练 episode 的曲线；
5. actor loss、critic loss、distillation loss 曲线；
6. LLM-only 的伪收敛曲线；
7. 同时展示 TCR、on-time completion rate 和 `1-DVR`；
8. 同时在正文展示 device energy 与 system energy；
9. 重复 baseline 名称：`MADDPG`、`MAPS-w/o-LLM`、`MAPS-w/o-Distill`；
10. HAPPO、QMIX、VDN、QPLEX 等未完成或动作空间不匹配的方法；
11. 只有单 seed 或没有置信区间的柱状图；
12. 旧模型下的 “60% fewer episodes” 和 “40% lower energy” 数字。

需要继续记录但不进入正文：

- actor/critic/distillation loss；
- queue backlog；
- per-node utilization；
- device energy；
- policy-expert similarity；
- parser 原始错误；
- 每任务明细。

这些数据用于诊断、复现和附录，而不是主结论。

## 11. 现有代码需要补充和修改

## 11.1 Hybrid action

新增：

```text
algos/common/hybrid_action.py
```

修改：

```text
algos/maddpg/maddpg_actor_critic.py
algos/maddpg/maddpg_agent.py
algos/maddpg/replay_buffer.py
algos/mappo/mappo_policy.py
algos/mappo/mappo_agent.py
experiments/runner.py
```

要求：

- partition 使用 softmax；
- ES 使用 categorical probabilities；
- replay 保存 edge one-hot/probability；
- environment boundary 才转换为 `edge_id`；
- 删除连续 edge scalar 乘 ES 数量后取整。

## 11.2 MAPS variants

统一配置：

```text
method: maddpg
method: mappo
method: maps_no_annealing
method: maps
method: llm_only
method: greedy_min_cost
```

`maps_no_llm` 和 `maps_no_distill` 不再建立独立 runner。

## 11.3 Expert cache

修改：

```text
llm_assistant/prompt_builder.py
llm_assistant/response_parser.py
llm_assistant/expert_provider.py
llm_assistant/llm_client.py
```

新增：

- canonical state serialization；
- state hash；
- state-keyed response cache；
- prompt/model metadata；
- token/query latency；
- parser validity；
- paper evidence flag。

固定通用专家动作的 `fixtures/legacy_expert_cache.json` 只保留 smoke test 用途。

## 11.4 Scenario bank and evaluation

建议新增：

```text
experiments/ei/scenario_bank.py
experiments/ei/train.py
experiments/ei/evaluate.py
experiments/ei/analyze.py
experiments/ei/baselines.py
configs/ei/base.yaml
configs/ei/methods/
configs/ei/scales/
configs/ei/deadlines/
```

scenario bank 负责冻结任务、信道和初始状态。训练与测试 bank 分离。

## 11.5 Metrics

重写或替换现有 `utils/metrics.py` 中含义模糊的字段：

```text
completion_rate -> task_completion_rate
timeout_rate -> deadline_violation_ratio
energy -> system_energy_j / device_energy_j
```

新增：

- per-task latency records；
- P95；
- energy per completed task；
- drain-horizon unfinished tasks；
- Reward AUC；
- steps to threshold；
- hierarchical seed aggregation；
- 95% CI。

## 11.6 Environment evaluation mode

修改 `environment/cloud_edge_env.py`：

- 支持 `evaluation=True`；
- 关闭 exploration 由 agent 控制；
- 支持 stop-arrival + drain；
- 输出完整 task completion records；
- 记录 decision latency 时不混入 environment simulation time；
- 对所有方法保持相同 reward 和物理模型。

## 12. 配置与运行产物

每次正式运行保存：

```text
run_manifest.json
resolved_config.yaml
scenario_bank_id
seed
git_commit
episode_metrics.json
task_records.parquet/csv
training_losses.json
evaluation_metrics.json
checkpoint
```

最终图表只能从正式 artifact 自动生成，不能手工填写数值。

## 13. Pilot gate

正式大实验前只跑：

```text
U=10, E=5
2 seeds
short budget
```

必须同时满足：

- hybrid action 的梯度和执行语义正确；
- MAPPO 和 MADDPG 均可稳定训练；
- LLM parser/constraint validity 足够高；
- MAPS 早期 Reward AUC 不低于 MADDPG；
- no-annealing 与 MAPS 确实形成可测差异；
- DVR/TCR 不是长期全 0 或全 1；
- 50 UE 单步运行时间可接受。

若 DVR 全 0 或全 1，应调整 deadline distribution 或 offered load，而不是在画图阶段筛选场景。

## 14. 最终 claim 与证据

只有正式实验完成后才允许填写具体数字。

### Claim A：样本效率

证据：

- Fig. 3；
- Table III 的 Reward AUC 和 steps to threshold。

允许：

> MAPS reached the reference performance with fewer environment interactions than MADDPG under the evaluated TEC setting.

禁止：

> MAPS generally solves MARL sample inefficiency.

### Claim B：退火作用

证据：

- MAPS vs MAPS-w/o-Annealing；
- Fig. 3 和 Table II。

允许：

> Annealed guidance preserved the early benefit of the expert prior while avoiding the fixed-guidance performance ceiling observed in this experiment.

禁止：

> Annealing eliminates LLM hallucination.

### Claim C：部署性能

证据：

- Fig. 4；
- Table II。

允许：

> The trained MAPS policies reduced P95 latency and deadline violations without online LLM inference in the tested scenarios.

禁止：

> MAPS guarantees reliable real-time scheduling.

### Claim D：规模压力

证据：

- Fig. 5。

允许：

> MAPS maintained lower P95 latency/DVR than the selected baselines as the number of UEs increased from 10 to 50 under a fixed five-ES topology.

禁止：

> MAPS is scalable to massive 6G networks.

## 15. 最终完成条件

EI 实验完成必须满足：

- 六个正文方法定义清晰且可运行；
- MAPPO 使用正确混合动作；
- Greedy 与 LLM-only 有可复现实现；
- E1-E5 均有冻结配置和 raw artifact；
- E1-E3 至少 5 training seeds；
- 正式测试关闭 exploration 和 LLM；
- Fig. 3-5 与 Table I-III 可自动生成；
- Fig. 6 可放附录；
- 不复用旧实验数字；
- 每个论文结论都能定位到 artifact、seed 和统计脚本。
