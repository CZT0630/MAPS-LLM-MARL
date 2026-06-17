# MAPS EI 投稿唯一实施方案

> 适用分支：`publication/ei-conference`
>
> 文档角色：本文件是 EI 代码修改、正式实验和论文重写的唯一执行依据。
>
> 更新日期：2026-06-13

## 1. 使用规则

后续实施只参考本文件，不再分别参考旧的“会议修改方案”和“实验呈现方案”。

其他保留文档的用途如下：

| 文档 | 用途 | 是否作为后续方案 |
|---|---|---:|
| `EI_SUBMISSION_MASTER_PLAN.md` | EI 代码、实验、论文的唯一主计划 | yes |
| `PHASE1_BASELINE_AUDIT.md` | Phase 0-1 已完成工作的审计记录 | no |
| `PHASE2_IMPLEMENTATION.md` | Phase 2 环境修复与边界记录 | no |
| `MIMO_API_SETUP.md` | MiMo API、`.env` 和探测命令操作手册 | no |
| `legacy/PHASE0_MANIFEST.md` | 旧版本冻结归档校验 | no |

若代码实现与本文件冲突，先更新本文件中的决策记录，再修改代码。正式实验
不得使用 Phase 1/2 smoke 结果、旧论文曲线或手工构造数值。

## 2. 投稿目标与研究定位

EI 版本不重构为 Letter 路线，也不追求新的通用 LLM-RL 理论。目标是在保留
原始 MAPS 主线的基础上，以可控成本形成一篇技术严谨、证据完整的会议论文。

推荐标题：

> MAPS: LLM-Guided Multi-Agent Task Partitioning and Parallel Scheduling in Terminal-Edge-Cloud Networks

核心定位：

> This paper develops an LLM-guided multi-agent reinforcement learning
> framework for fine-grained task partitioning and parallel scheduling in
> terminal-edge-cloud networks. The LLM provides offline expert priors during
> training, while the learned decentralized policies make real-time decisions
> without online LLM inference.

创新边界不是“首次结合 LLM 与 MARL”，而是以下组合在 TEC 问题中的完整实现：

1. 同一任务在 local、edge、cloud 三条并行分支上的细粒度比例划分。
2. 连续任务比例与离散 ES 选择组成的多智能体混合动作。
3. LLM 专家先验通过连续 MSE、离散 CE 的混合蒸馏进入 MADDPG。
4. 专家权重随训练进度退火，部署阶段只运行 actor，不调用 LLM。
5. 在包含队列、deadline、无线链路、有线回传和结果返回的动态任务流中评价。

论文只使用以下克制表述：

- `develops`
- `is designed for`
- `focuses on`
- `under the evaluated settings`

禁止无证据使用：

- `the first`
- `innovative`
- `AGI`
- `generalizable`
- `near-optimal`
- `scalable to massive 6G networks`
- `guarantees convergence/reliability`
- `can be extended to other areas`

## 3. 最终贡献点

论文贡献控制为三条：

1. **系统建模。** 建立动态 TEC 任务流下的细粒度并行任务划分问题，显式考虑
   UE/ES/CS 队列、无线传输、有线回传、结果返回、计算与通信能耗以及 deadline，
   并表述为 Dec-POMDP。
2. **方法设计。** 开发 MAPS，将 observation-conditioned LLM 专家先验蒸馏到
   MADDPG actor；连续比例使用 MSE，离散 ES 选择使用交叉熵，专家权重按训练进度
   退火。
3. **实验评价。** 通过收敛、消融、独立测试、可靠性和 UE 压力实验，比较
   MAPS、MADDPG、MAPPO、固定蒸馏、Greedy-MinCost 和 LLM-only。

任务语义、正确物理模型、指标基础设施和真实 API 接入是可信性要求，不单独包装
成算法创新。

## 4. 当前代码状态

### 4.1 已完成，可直接复用

- Phase 0：旧代码、原始 LaTeX 和旧配置已冻结在 `legacy/`。
- Phase 1：MADDPG 使用 joint replay 和 centralized critic，运行产物包含 seed、
  config、commit、manifest、metrics、loss 和 checkpoint。
- Phase 2：已修复仿真时间、持续队列、多任务到达、无线速率、有线回传、下行返回、
  UE/ES/CS 能耗、任务语义和联合动作评估。
- 小米 MiMo OpenAI-compatible client 已接入，实际 API 模型为 `mimo-v2.5`。
- API key 从项目根目录 `.env` 的 `MIMO_API_KEY` 读取，不进入 Git 或 manifest。
- Greedy-MinCost 已加入 `baselines/greedy_min_cost.py` 和统一 runner，并有单元测试。

### 4.2 已实现但不能直接作为论文结果

- `legacy_maps` 仍使用合成固定专家缓存。
- C1 已统一为 `3+E` 策略动作，但正式实验和冻结场景尚未开始。
- C2 已实现 mixed distillation、分项记录和 environment-step 三阶段退火；
  smoke 仍使用合成固定缓存，正式 state-keyed cache artifact 尚未生成，
  `0.8/0.15/0` 仍只作为 pilot 初值。
- C3 的 prompt、联合状态哈希、严格 parser、可审计 cache、共享 provider、
  LLM-only runner、真实 MiMo cache 与配套冻结场景产物已完成。
- C4 的 drain horizon、逐任务记录、正式 evaluation artifact、P95/DVR/TCR、
  decision latency、energy-per-completed-task 和 AUC/threshold summary 已完成。
- 当前训练与测试的大规模 scenario bank 编排仍未完成，不能开始 EI 正式矩阵。

### 4.3 尚待完成

| 工作 | 状态 | 正式实验前是否必须 |
|---|---|---:|
| Hybrid action codec | **done** | yes |
| MADDPG mixed-action actor/critic/replay | **done** | yes |
| MAPPO categorical ES head | **done** | yes |
| Partition squared L2 + edge CE | **done** | yes |
| Fixed/annealed/no-LLM 三种 MAPS 配置 | **done** | yes |
| state-keyed cache 代码与审计校验 | **done** | yes |
| 真实 MiMo cache artifact | **done** | yes |
| LLM-only evaluator 代码 | **done** | yes |
| LLM-only 冻结场景评估证据 | **done** | yes |
| Drain horizon 和逐任务记录 | **done** | yes |
| P95、DVR、TCR、AUC、threshold | **done** | yes |
| 10/20/30/50 UE scenario bank | pending | yes |
| 论文 LaTeX 独立工作区 | pending | yes |
| 自动图表和统计脚本 | pending | yes |

## 5. 方法与命名统一

### 5.1 论文中的方法名

正文固定使用六种方法：

| 方法 | 类型 | 训练使用 LLM | 部署使用 LLM |
|---|---|---:|---:|
| Greedy-MinCost | 一步模型启发式 | no | no |
| LLM-only | 直接语言模型决策 | no training | yes |
| MADDPG | 无专家 MARL | no | no |
| MAPPO | 第二类 MARL baseline | no | no |
| MAPS-w/o-Annealing | 固定专家权重消融 | yes | no |
| MAPS | 完整方法 | yes | no |

图表方法名使用 `LLM-only`，不能写成错误的 `LMM-only`。实验设置和论文参数表必须
披露实际模型：

```text
Provider: Xiaomi MiMo
Paper model name: MiMo-V2.5
API model ID: mimo-v2.5
```

不得把 MiMo 生成的结果标成 Qwen、GPT、GLM 或其他模型。方法名可以保持通用，
但模型身份不能隐瞒，否则实验无法复现。

### 5.2 等价名称去重

在当前方法定义下：

```text
MADDPG = MAPS-w/o-LLM = MAPS-w/o-Distillation
```

正文只保留 `MADDPG`，不把三个等价名称画成三条曲线。

### 5.3 不新增的主 baseline

EI 正文不新增 HAPPO、QMIX、VDN、QPLEX、QTRAN、多 LLM 或 CV-CED：

- HAPPO 与 MAPPO 对当前问题提供的信息重复。
- value-decomposition 方法需要额外离散化连续 partition，公平性较差。
- 多 LLM 和 CV-CED 属于高成本 Letter 路线。
- All-Local、All-Edge、Random 只保留为 sanity check 或专家质量弱参照。

## 6. 代码修改方案

按 C1-C6 顺序执行。每一阶段达到停止条件后再进入下一阶段。

### C1：统一混合动作

新增：

```text
algos/common/hybrid_action.py
```

每个 UE 的策略动作定义为：

```text
a_u = (phi_u, p_u)
phi_u = softmax(partition_logits)       # 3 dimensions
p_u   = softmax(edge_logits)            # E dimensions
```

要求：

- actor 输出 3 个 partition logits 和 `E` 个 edge logits。
- 训练时 critic 接收 `[phi_local, phi_edge, phi_cloud, edge_probs...]`。
- replay 保存 edge one-hot/probability，不保存归一化 edge scalar。
- 环境边界才执行 `edge_id = argmax(p_u)`。
- expert、MADDPG、MAPPO、replay、critic 和环境共用同一 codec。
- ES 数量完全由配置决定，至少测试 3/5/10 ES。

需要修改：

```text
algos/maddpg/maddpg_actor_critic.py
algos/maddpg/maddpg_agent.py
algos/maddpg/replay_buffer.py
algos/mappo/mappo_policy.py
algos/mappo/mappo_agent.py
experiments/runner.py
```

停止条件：

- 不再出现连续 edge scalar 乘 ES 数量后取整。
- codec round-trip、shape、mask、invalid action 和 actor gradient 测试通过。
- MADDPG 与 MAPPO 可在 Phase 2 环境完成短训练。

**C1 完成记录（2026-06-13）：**

已修改文件：

```text
algos/common/hybrid_action.py          (新建: HybridActionCodec)
algos/maddpg/maddpg_actor_critic.py    (Actor 输出 3+E, Critic 接收 3+E)
algos/maddpg/maddpg_agent.py           (logit 噪声, codec 边界, 混合蒸馏)
algos/mappo/mappo_policy.py            (Dirichlet simplex + Categorical edge)
algos/mappo/mappo_agent.py             (输出 [3+E], PPO 混合 log-prob)
algos/happo/happo_policy.py            (同步使用 simplex + categorical)
algos/happo/happo_agent.py             (同步更新)
algos/common/trajectory_buffer.py      (存储 partition 参数与 edge logits)
baselines/greedy_min_cost.py           (输出 [N, 3+E] one-hot)
experiments/runner.py                  (统一 codec 转换)
llm_assistant/expert_provider.py        (专家动作经统一 codec 转换)
tests/test_hybrid_action.py            (codec、3/5/10 ES、梯度和回归测试)
tests/test_greedy_min_cost.py          (适配新 action 格式)
tests/test_maddpg_agent.py             (适配新 action 格式)
tests/test_replay_buffer.py            (适配新 action 格式)
```

验证结果：

- 135/135 测试通过。
- Phase 1 smoke（5 算法）全部通过。
- Phase 2 短训练（MADDPG、MAPPO）通过；同步 HAPPO/Legacy MAPS smoke 也通过。
- 无连续 edge scalar 取整；环境边界只使用 `argmax(edge_probs)`。
- codec round-trip、shape、3/5/10 ES 真实训练更新、mask、invalid action、
  replay 数据完整性、极端 logits 和 actor gradient 测试通过。

### C2：实现 mixed distillation 与退火

损失定义：

```text
L_part = mean_u ||phi_u^RL - phi_u^LLM||_2^2
L_edge = mean_u CE(p_u^RL, e_u^LLM)
L_distill = L_part + eta_edge * L_edge
L_actor = L_RL + lambda(k) * L_distill
```

`k` 统一使用 environment-step progress，不使用 episode 编号。

正式模式：

```text
maddpg:              lambda(k) = 0
maps_no_annealing:   lambda(k) = lambda_fixed
maps:                lambda(k) = scheduled value
```

最低成本退火采用三阶段配置：

```text
lambda_high, progress < 0.3
lambda_low,  0.3 <= progress < 0.7
0,           progress >= 0.7
```

原稿的 `0.8/0.15/0` 只能作为 pilot 初始值，必须检查 RL loss 与 distillation loss
的量级，不能未经验证直接冻结。

本 EI 版本不把 LLM 自报 confidence 直接乘入损失。parser validity 和缺失字段只生成
`expert_mask`；confidence 可以记录，但除非完成校准实验，否则不参与训练。

停止条件：

- `L_part`、`L_edge`、`L_distill` 可分别记录。
- `lambda=0` 时专家项不产生 actor 梯度。
- fixed 和 annealed schedule 可由配置复现。
- checkpoint/manifest 保存 schedule、权重和 prompt/cache 版本。

### C3：建立真实 LLM 专家证据链

实际模型固定为 `MiMo-V2.5 (mimo-v2.5)`，配置见
`configs/ei/llm_mimo.yaml`，密钥操作见 `MIMO_API_SETUP.md`。

Prompt 输入至少包含：

- task data size、cycles、deadline slack、semantic type、priority、output ratio；
- UE CPU、pending queue 和 UE-ES rates；
- 每个 ES 的 CPU、队列负载；
- cloud/backhaul 摘要；
- partition 约束和合法 ES id；
- JSON-only 输出约束。

建议输出 schema：

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

正式 cache 必须保存：

```text
state_hash
prompt_version
provider
requested_model
returned_model
temperature
raw_response_hash
parsed_action
parser_status
fallback_status
request_id
token_usage
query_latency_ms
generation_time
paper_evidence=true
```

要求：

- 正式训练不在线重复请求 API。
- `LLM-only`、`MAPS-w/o-Annealing` 和 `MAPS` 共享同一 state-keyed cache。
- `fixtures/legacy_expert_cache.json` 继续只用于 smoke test。
- API key 不写入 config、artifact、日志或 Git。
- 论文不声称跨 LLM 泛化。

停止条件：

- cache 可审计且状态哈希稳定。
- parser success、valid action 和 fallback rate 可统计。
- LLM-only 可在冻结 scenario bank 上独立评估。

**C3 完成记录（2026-06-17）：**

已新增/修改文件：

```text
llm_assistant/ei_prompt_builder.py        (新建: EI-v1 prompt builder)
llm_assistant/ei_state.py                 (新建: 联合决策状态提取)
llm_assistant/expert_cache.py             (新建: ExpertCache, CacheEntry, hash_state)
llm_assistant/cached_expert_provider.py   (新建: CachedExpertProvider)
llm_assistant/__init__.py                 (导出新类)
baselines/llm_only.py                     (新建: LLMOnlyAgent)
scripts/generate_expert_cache.py          (新建: API cache 生成脚本)
experiments/runner.py                     (新增 llm_only 算法支持)
config.yaml                               (新增 llm_only 配置段)
configs/ei/formal_s1.yaml                 (新增: C3 正式 S1 配置)
tests/test_expert_cache.py                (新增 C3 单元与集成测试)
tests/test_ei_formal_config.py            (新增: 正式配置检查)
```

验证结果：

- C3 定向测试覆盖 cache、prompt、严格 parser、共享 provider 和 runner 集成。
- ExpertCache 支持原子 save/load，并在加载时验证 raw response hash、模型、
  temperature、状态键和 parser/fallback 一致性。
- hash_state 使用完整 SHA-256、稳定规范化，并保留 `1e-9` 等小量纲物理参数。
- CachedExpertProvider 使用联合状态键，区分 cache miss、parser fallback、非决策 UE
  与无任务时隙。
- LLMOnlyAgent 输出 [3+E] policy action，fallback 为 all-local。
- LLM-only 部署评估直接执行 cache 中的原始 env action，避免 float32 往返造成
  严格状态哈希漂移。
- add_from_api_response 严格检查 partition 和 ES id，按实际待决策 UE 统计
  valid/partial/failed。
- generate_expert_cache.py 脚本可通过 `python -m scripts.generate_expert_cache` 运行。
- 生成脚本同时输出与 cache 轨迹配套的冻结场景清单。
- llm_only 可通过统一 runner 接口运行，记录场景文件 SHA-256，并可强制零 cache miss。
- API key 不写入 cache、config 或 manifest。

正式产物与统计：

- `artifacts/ei/expert_cache.json`
  - entries: 500
  - provider/model: Xiaomi MiMo / `mimo-v2.5`
  - cache SHA-256: `7485c17fc1ea2c0e14f2312f472997ee1b2e271aee75ad8761885a66c1c83439`
  - parser success rate: 100%
  - valid action rate: 100%
  - fallback rate: 0%
  - mean uncached latency: 4988.11 ms
  - max uncached latency: 62512.86 ms
- `artifacts/ei/expert_scenario_bank.json`
  - scenario bank id: `ei-expert-29c3d494edf5b103`
  - scenario count: 3
  - construction seed: 42
- LLM-only 冻结评估：
  - `online_api_calls=0`
  - cache lookups/hits/misses: 500 / 500 / 0
  - decision actions: 2581
  - runtime fallback actions/rate: 0 / 0%
- `MAPS-w/o-Annealing` 和 `MAPS` 1-step cache 验证均命中同一
  `artifacts/ei/expert_cache.json` 与同一 cache SHA-256。

### C4：正式指标与评估协议

逐任务记录至少包含：

```text
task_id
scenario_id
arrival_time
deadline
completion_time
latency
completed
completed_on_time
device_energy_j
system_energy_j
decision_latency_ms
```

定义：

```text
N_gen     = evaluation window 内生成的任务数
N_done    = drain horizon 结束前完成的任务数
N_on_time = deadline 前完成的任务数

TCR = N_done / N_gen
DVR = (N_gen - N_on_time) / N_gen
```

`DVR` 包含超时完成和 drain horizon 后仍未完成的任务。P95 latency 只对已完成任务
计算，但必须与 TCR 和 DVR 同时报告。

Primary metrics：

1. Reward AUC。
2. P95 task latency。
3. DVR。
4. System energy per completed task。

Secondary metrics：

- steps to threshold；
- final evaluation reward；
- mean latency；
- TCR；
- device energy per completed task；
- actor/Greedy/LLM decision latency；
- training wall-clock；
- LLM tokens、query latency、parser success 和 fallback rate。

收敛横轴统一为 environment interactions：

```text
Reward AUC = 相同 interaction 区间上的归一化曲线面积
Steps to threshold = 首次达到并连续 K 次保持参考阈值的交互步数
Final reward = 最后 10% evaluation points 的均值
```

若某方法未达到 threshold，表格写 `not reached`。

环境需要支持：

- `evaluation=True`；
- 停止新任务到达后进入固定 drain horizon；
- 完整 task records；
- decision wall-clock 不混入仿真时延；
- 正式评估关闭 exploration；
- MAPS 正式评估关闭 LLM。

**C4 完成记录（2026-06-17）：**

已新增/修改文件：

```text
environment/cloud_edge_env.py       (新增 evaluation/drain 控制和逐任务 ledger)
experiments/runner.py               (新增 run_evaluation 与 C4 metrics/artifacts)
main.py                             (新增 --mode eval)
configs/ei/formal_s1.yaml           (新增 evaluation 协议参数)
tests/test_c4_evaluation.py         (新增 C4 协议测试)
tests/test_ei_formal_config.py      (扩展正式配置检查)
README.md                           (更新状态与 eval smoke 命令)
```

实现结果：

- `env.reset(options={"evaluation": True, "scenario_id": ...})` 可开启评估语义。
- runner 在 evaluation window 最后一个 step 停止新任务到达，然后执行固定
  `drain_horizon_steps`。
- 每个任务记录至少包含 `task_id`、`scenario_id`、`arrival_time`、`deadline`、
  `completion_time`、`latency`、`completed`、`completed_on_time`、
  `device_energy_j`、`system_energy_j` 和 `decision_latency_ms`。
- `task_records.csv` 使用 C4 drain 语义重算 `completed`，drain horizon 后仍未完成的
  任务计入 DVR。
- `evaluation_metrics.json` 输出 `N_gen`、`N_done`、`N_on_time`、`TCR`、`DVR`、
  `p95_task_latency`、`mean_latency`、system/device energy per completed task
  以及 mean/P95 decision latency。
- 训练 run summary 新增按 environment interactions 计算的 `reward_auc`、
  `final_reward`、`steps_to_threshold` 和 threshold 状态。
- `--mode eval` 默认关闭 exploration；学习型算法没有 `evaluation.checkpoint_dir`
  时会拒绝正式评估，除非显式设置 `evaluation.allow_untrained=true`。
- `MAPS` 与 `MAPS-w/o-Annealing` 的正式 eval 不使用 LLM；`LLM-only` 仍只使用
  frozen cache，`online_api_calls=0`。

### C5：Scenario bank、配置和运行产物

建议新增：

```text
experiments/ei/scenario_bank.py
experiments/ei/train.py
experiments/ei/evaluate.py
experiments/ei/analyze.py
configs/ei/base.yaml
configs/ei/methods/
configs/ei/scales/
configs/ei/deadlines/
artifacts/ei/
```

训练 bank 与测试 bank 必须分离。相同 test seed 下所有方法共享：

- 任务到达与属性；
- UE-ES 距离和信道；
- 初始队列；
- deadline 分布。

每次正式运行保存：

```text
run_manifest.json
resolved_config.yaml
scenario_bank_id
seed
git_commit
episode_metrics.json
task_records.csv/parquet
training_losses.json
evaluation_metrics.json
checkpoint
```

所有图表从正式 artifact 自动生成，禁止手工填数字。

### C6：Pilot gate 与正式运行

先运行：

```text
U=10, E=5, C=1
2 training seeds
short environment-step budget
```

进入正式实验前必须同时满足：

- MADDPG、MAPPO、MAPS fixed 和 MAPS annealed 稳定训练。
- hybrid action 的训练和执行语义一致。
- parser/constraint validity 足够高。
- MAPS 的早期 Reward AUC 不低于 MADDPG。
- fixed 与 annealed 形成可测差异。
- DVR/TCR 不长期全 0 或全 1。
- 50 UE 单步和训练成本可接受。

若专家先验不能改善早期 AUC，先检查 prompt、cache 覆盖、loss scale 和 mask，不直接
扩大实验矩阵。若 DVR 退化为全 0/1，调整 workload/deadline 分布并重新冻结场景，
不能在画图阶段筛选结果。

## 7. 最终实验方案

### 7.1 场景

S1 基础场景：

```text
U = 10, E = 5, C = 1
deadline = medium
```

用于收敛、消融、主性能和专家质量。

S2 UE 拥塞压力：

```text
U = {10, 20, 30, 50}
E = 5, C = 1
```

固定 ES 数量，随着 UE 增长增加资源竞争。该实验称为 congestion/scalability stress
test，不声称线性扩展或 massive-scale scalability。

S3 deadline sensitivity：

```text
U = 10, E = 5, C = 1
deadline factor = {loose, medium, strict}
```

使用 medium 场景训练策略，在三个冻结测试 bank 上评估。该实验是 sensitivity/stress
test，不称为跨分布泛化。

正式任务流采用 Poisson 到达和四类任务：

```text
video_analytics
ar_vr
ai_inference
control
```

### 7.2 实验矩阵

E1 训练收敛与退火消融：

```text
Scenario: S1
Methods: MADDPG, MAPPO, MAPS-w/o-Annealing, MAPS
Metrics: reward curve, Reward AUC, steps to threshold, final reward
Answers: RQ1, RQ2
```

Greedy 和 LLM-only 没有训练过程，不进入收敛曲线。

E2 独立测试集主性能：

```text
Scenario: S1 frozen test bank
Methods: Greedy-MinCost, LLM-only, MADDPG, MAPPO,
         MAPS-w/o-Annealing, MAPS
Metrics: mean/P95 latency, system energy/completed task,
         device energy/completed task, DVR, TCR, decision latency
Answers: RQ3
```

E3 UE 规模与拥塞：

```text
Scenario: S2
Methods: Greedy-MinCost, MADDPG, MAPPO, MAPS
Metrics: P95 latency, DVR, system energy/completed task, TCR,
         decision latency
Answers: RQ4
```

每个规模分别训练对应学习策略。不能把 10-UE centralized critic 当作 50-UE 正式模型。

E4 deadline sensitivity：

```text
Scenario: S3
Methods: Greedy-MinCost, MADDPG, MAPPO, MAPS
Metrics: P95 latency, DVR, TCR
```

篇幅不足时整组放附录，不删除 raw results。

E5 专家质量与开销：

```text
States: at least 500 frozen S1 states
Actions: Random, All-Local, Greedy-MinCost, MiMo-V2.5 expert
Metrics: parser success, valid rate, one-step cost,
         improvement over weak references, gap to Greedy,
         tokens, uncached latency, cache hit rate,
         cache generation time, actor latency,
         deployment LLM calls (=0)
Answers: RQ5
```

LLM expert 不必优于 Greedy，但必须具备足够解析/约束有效率，并在 pilot 中带来实际
早期训练收益；否则降低对专家质量的 claim。

### 7.3 统计与公平性

- E1-E3 至少 5 个独立 training seeds。
- 每个 checkpoint 使用相同的至少 20 个 test scenario seeds。
- 每个 checkpoint 的测试集合至少产生 1000 个任务。
- 先对每个 training seed 的测试场景聚合，再以 training seed 作为独立统计单位。
- 报告 mean、standard deviation、95% CI 和 paired effect。
- 5 seeds 时以 CI 和 effect size 为主，不以单个 `p<0.05` 作为核心证据。
- Greedy 和 LLM-only 使用 test-scenario bootstrap CI。

所有学习方法必须共享：

- scenario banks；
- environment-step budget；
- observation、reward 和 hybrid action semantics；
- evaluation interval；
- 同数量级网络参数；
- 硬件、精度和 seed 集合。

MAPS 与 MAPS-w/o-Annealing 还必须共享同一模型、prompt、cache、parser、fallback 和
distillation loss，只改变 annealing schedule。

预计正式训练量：

```text
E1/S1: 4 learning methods * 5 seeds = 20 runs
E3:    3 learning methods * 3 new scales * 5 seeds = 45 runs
Total: 65 formal training runs
```

E2、E4 复用 checkpoint；Greedy 与 LLM-only 只评估。

## 8. 论文修改方案

### 8.1 Abstract

按五句结构重写：

1. TEC 动态任务流需要细粒度划分与并行调度。
2. MARL 早期探索需要大量环境交互，在线 LLM 又有明显开销。
3. MAPS 使用离线 LLM 先验、mixed-action distillation 和 annealing。
4. 简述系统模型、baseline、场景和指标。
5. 只写正式实验支持的数字。

删除 AGI、跨领域推广和未验证 generalization。

### 8.2 Introduction

采用五段：

1. TEC 应用与 fine-grained partitioning 需求。
2. DRL/MARL task offloading 和 partitioning 的现状与局限。
3. LLM-guided RL 的机会与在线推理成本。
4. MAPS 的 offline prior、mixed distillation、annealing 和 zero-call deployment。
5. 三条贡献。

“DRL sample inefficiency/high exploration cost”必须配可靠引用。

### 8.3 Related Work

独立成节，分为：

1. TEC/MEC task offloading and partitioning。
2. MARL for terminal-edge-cloud scheduling。
3. LLM-guided RL and LLM-enhanced edge scheduling。

必须明确：已有工作已经覆盖 LLM policy teacher、LLM regularization、LLM-enhanced
MARL offloading 和 LLM direct scheduling。MAPS 不声称提出通用 LLM-RL 算法。

### 8.4 System Model

公式必须与 Phase 2 代码一致。

任务：

```text
pi_u(t) = (d_u, l_u, D_u, rho_u, omega_u, priority_u, arrival_u)
```

允许 Poisson 多任务到达，每个 UE 每时隙最多为 admission queue 队首任务做一次决策，
不再写“每个 UE 只产生一个任务”。

无线速率：

```text
R_u(t) = B_u log2(1 + P_u h_u(t) / (N0 B_u + I_u(t)))
```

主实验采用 orthogonal access 时明确 `I_u(t)=0`，并定义 path gain。

队列：

```text
Q_n(t+1) = [Q_n(t) - f_n Delta t]^+ + A_n(t)
```

解释实现中保存 task objects，以计算 release、FCFS、deadline 和 completion。

回传与下行：

- UE-ES 是无线链路。
- ES-CS 使用 `data/rate + propagation`。
- 回传能耗为 energy-per-bit，不使用虚构 cloud RF power。
- 返回数据量由 output ratio 决定。
- cloud 返回路径为 CS-ES-UE。

计算能耗：

```text
E_n^cmp = kappa_n L_n f_n^2
```

给出 `kappa` 和频率单位，并引用经过核验的 MEC/DVFS 文献。

并行时延：

```text
T_u = max(T_u^local, T_u^edge, T_u^cloud)
```

每条分支包含 admission wait、communication release、queue wait、computation 和 return。

目标函数：

```text
J = w_T T_norm + w_E E_norm + w_D DVR
```

不再让 objective 和 reward 使用含义不清且重名的 `alpha`。

### 8.5 Dec-POMDP 与 MAPS

明确：

- 每个 UE 是一个 agent。
- actor 部署时只使用 local observation。
- centralized critic 训练时使用 joint state/action。
- target networks 构造 Bellman target。
- exploration noise 只用于训练。

LLM 与 MDP 的关系写为：

```text
a_u^LLM ~ pi_LLM(. | prompt(o_u, global_summary))
```

LLM action 是由当前状态条件化得到的训练标签，不属于环境转移函数，也不在部署时参与
策略推理。

Algorithm 1 必须包含：

1. 初始化 actor、critic、target、replay 和 expert cache。
2. 观察状态并按 state hash 读取专家先验。
3. actor 输出 partition 和 edge probabilities。
4. 加探索并通过 codec 执行环境动作。
5. 保存 joint transition、expert action 和 mask。
6. Bellman loss 更新 critic。
7. 计算 partition MSE、edge CE 和 `lambda(k)`。
8. RL loss 加 mixed distillation 更新 actor。
9. soft-update target networks。
10. 训练结束只部署 actor。

删除没有定义的“Use noise strategies”和“Synchronize cloud/edge agents”等句子。

### 8.6 Convergence Discussion

不尝试证明深度非凸 MADDPG 的全局收敛。只做：

- 给出 actor/critic 更新公式；
- 解释 target network、replay 和 bounded reward 的稳定化作用；
- 用 5 seeds、AUC、steps-to-threshold 和 CI 做经验收敛评价；
- 在 limitations 明确没有全局收敛保证。

### 8.7 Experiments and Results

按研究问题 RQ1-RQ5 组织，不按日志类型组织。每个小节必须包含：

- 问题；
- 场景与方法；
- 指标定义；
- 结果与 CI；
- 物理含义；
- 限制。

不得使用旧 `Reward.pdf`、`Latency.pdf`、`Energy.pdf` 或旧版“60% fewer episodes”、
“40% lower energy”数字。

### 8.8 Conclusion and Limitations

结论只总结正式实验支持的样本效率、时延、能耗、可靠性和压力测试结果。

明确限制：

- 单一 LLM；
- 仿真环境；
- orthogonal access 和静态 path-loss 假设；
- 最多测试 50 UE；
- 无全局收敛保证；
- 未证明跨拓扑或跨分布泛化。

## 9. 最终图表

正文目标为：

| 编号 | 内容 | 回答问题 |
|---|---|---|
| Fig. 1 | TEC system、三条并行分支、queues、links | 系统模型 |
| Fig. 2 | MAPS training/deployment workflow | 方法 |
| Fig. 3 | Reward curve + AUC/threshold 消融 | RQ1/RQ2 |
| Fig. 4 | Independent-test P95 latency + DVR | RQ3 |
| Fig. 5 | P95/DVR/energy vs UE count | RQ4 |
| Fig. 6 | DVR/TCR vs deadline factor，可放附录 | sensitivity |
| Table I | 仿真与训练参数 | reproducibility |
| Table II | 主测试 latency/energy/DVR/TCR/decision latency | RQ3 |
| Table III | AUC/threshold/training time/LLM overhead | RQ1/RQ5 |

不进入正文：

- actor/critic/distillation loss 曲线；
- latency/energy 随训练 episode 曲线；
- LLM-only 伪收敛曲线；
- 同时展示 `TCR`、`1-DVR` 和 on-time completion；
- 同时在主表展示 device 与 system energy；
- 单 seed 柱状图；
- 未定义的 normalized reward；
- HAPPO/QMIX/VDN/QPLEX 等未完成或不匹配方法。

loss、queue backlog、per-node utilization、device energy、expert similarity 和 parser
错误仍需保存，用于诊断和附录。

## 10. 审稿意见对应关系

| 审稿问题 | 本方案动作 |
|---|---|
| LLM+RL 创新有限 | 降低 novelty claim，强调 TEC fine-grained mixed action |
| 场景只有 10 UE/5 ES | 增加 U=10/20/30/50 压力实验 |
| baseline 太少 | 增加 Greedy-MinCost、MAPPO、LLM-only 和退火消融 |
| rate 缺 channel gain/N0B | 使用 Phase 2 Shannon/path-loss 模型 |
| queue 量纲错误 | 引入 `Delta t`，队列统一为 cycles |
| wired energy 错误 | 使用独立 backhaul rate/propagation/energy-per-bit |
| 忽略 downlink | 引入 output ratio 和 edge/cloud return path |
| 混合动作统一 MSE | partition MSE + edge CE |
| objective/reward alpha 重名 | 使用 `w_T,w_E,w_D` |
| 每 UE 只有一个任务 | Poisson 多到达 + admission queue |
| Algorithm 1 抽象 | 展开 cache、replay、loss、annealing、target update |
| LLM 与 MDP 连接不清 | 定义 observation-conditioned training prior |
| actor/critic/target 不清 | 增加 CTDE 和完整更新公式 |
| 要求数学收敛证明 | 不过度承诺，改为经验收敛协议 |
| Normalized Reward 未定义 | 给公式并同时报告物理指标 |
| prompt/latency/overhead 缺失 | 报告 schema、tokens、latency、cache、zero deployment calls |

## 11. 执行顺序

1. 完成 C1 hybrid action。
2. 完成 C2 mixed distillation 和 annealing。
3. 完成 C3 真实 expert cache 与 LLM-only。
4. 完成 C4 指标、drain horizon 和正式 evaluation。
5. 完成 C5 scenario bank、配置与 artifact pipeline。
6. 执行 C6 两 seed pilot。
7. Pilot 通过后运行 5-seed 正式矩阵。
8. 自动生成图表与统计表。
9. 建立 `paper/ei/`，按第 8 节重写论文。
10. 做符号、单位、引用、claim 和匿名化审计。

预计成本约 3-4 周，不包含 API/GPU 排队时间。

## 12. 最低投稿完成标准

### 代码

- Hybrid action 不再使用连续 edge scalar 取整。
- Partition MSE + edge CE 已实现并有测试。
- 六种正文方法可通过统一接口运行。
- 正式 expert cache 可审计。
- 关键逻辑有测试。
- 所有运行带 seed、config、commit、scenario bank 和 manifest。

### 实验

- E1-E3 至少 5 training seeds。
- 主比较包含六种正文方法。
- 报告 mean/P95 latency、system energy、DVR、TCR。
- 完成 U=10/20/30/50。
- 收敛按 interactions、AUC 和 threshold 定义。
- 所有方法共享 workload、budget 和公平配置。
- 每个论文 claim 可定位到 raw artifact。

### 论文

- 系统公式与代码一致。
- LLM 与 Dec-POMDP 的连接明确。
- Algorithm 1 可复现。
- 实际 LLM 模型和 API 设置如实披露。
- 不复用旧结果。
- 不过度声称 novelty、泛化或理论收敛。
- 投稿前逐条核验引用元数据及其支持的具体 claim。

## 13. 当前下一步

当前最先执行的不是继续增加 baseline，也不是开始跑正式大实验，而是：

> 完成 C5 scenario bank、配置和正式 artifact pipeline。

在 C5 完成前，现有 `legacy_maps`、Greedy smoke、Phase 2 smoke、C3 cache 和
C4 evaluation smoke 证据只能证明工程路径、专家证据链与评估协议可运行，
不能进入 EI 论文最终结果。
