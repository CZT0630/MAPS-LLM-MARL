# LLM+MADDPG 云边端计算卸载系统

> 当前项目正在按新版 MAPS 研究路线重构。开始开发或运行正式实验前，请先阅读：
>
> - [MAPS 论文与实验系统完整改造方案](docs/MAPS_REDEVELOPMENT_PLAN.md)
> - [MAPS 创新定位与最新研究调研](MAPS_novelty_positioning_report_2026.md)
>
> 现有 README 描述的是旧版原型。当前主入口、LLM 训练链路和 MADDPG 联合训练仍在整改，旧结果不得直接作为新版论文证据。

## 🌟 项目概述

本项目实现了基于大语言模型(LLM)和多智能体深度确定性策略梯度(MADDPG)的云边端三层架构计算卸载决策系统。通过LLM专家知识指导和知识蒸馏技术，提升MADDPG智能体的决策能力。

### 核心特性

- **LLM+MADDPG混合架构**: 结合LLM的推理能力和MADDPG的学习能力
- **知识蒸馏技术**: Agent通过知识蒸馏学习LLM的决策模式
- **纯Agent模式**: 训练后的Agent可在无LLM指导下独立决策
- **云边端三层架构**: 支持10个端设备、5个边缘服务器、1个云服务器的异构环境
- **三元分割决策**: 支持任务在本地、边缘、云端的灵活分配
- **统一路径管理**: 标准化的目录结构和文件保存系统

## 🏗️ 系统架构

### 训练阶段
```
LLM专家指导 → MADDPG决策 → 环境交互 → 经验存储 → 网络训练(含知识蒸馏)
```

### 测试阶段
```
纯Agent决策 → 环境交互 → 性能评估
```

**注意**: 测试阶段不再需要LLM指导，Agent通过知识蒸馏已经内化了LLM的决策能力。

## 📁 项目结构

```
LLM4RL/
├── algos/                      # 算法实现
│   ├── maddpg_agent.py        # MADDPG智能体(支持知识蒸馏)
│   ├── maddpg_actor_critic.py # Actor-Critic网络
│   ├── replay_buffer.py       # 经验回放缓冲区
│   └── noise.py              # 噪声模型
├── environment/               # 环境模型
│   ├── cloud_edge_env.py     # 云边端环境
│   ├── device_models.py      # 设备模型
│   └── task_generator.py     # 任务生成器
├── llm_assistant/            # LLM助手模块
│   ├── llm_client.py         # LLM客户端
│   ├── prompt_builder.py     # 提示词构建
│   └── response_parser.py    # 响应解析
├── experiments/              # 训练测试脚本
│   ├── train_llm_maddpg_complete.py  # LLM+MADDPG训练
│   ├── test_llm_maddpg.py           # 纯Agent模式测试
│   ├── train_maddpg.py              # 纯MADDPG训练
│   ├── test_maddpg.py               # 纯MADDPG测试
│   ├── train_llm.py                 # 纯LLM训练
│   └── test_llm.py                  # 纯LLM测试
├── utils/                    # 工具函数
│   ├── path_manager.py       # 统一路径管理器
│   ├── csv_saver.py          # CSV数据保存
│   ├── config.py            # 配置管理
│   ├── metrics.py           # 性能指标
│   └── plotting.py          # 图表绘制
├── results/                  # 结果目录(自动生成)
│   └── experiment_{timestamp}/
│       ├── maddpg/models/           # 纯MADDPG模型
│       │   ├── actor_agent_0_final.pth
│       │   ├── critic_agent_0_final.pth
│       │   └── ...
├── llm_maddpg/                 # LLM+MADDPG算法结果
│   ├── models/                 # 模型文件
│   │   ├── agent_0_final.pth
│   │   ├── agent_1_final.pth
│   │   └── ...
├── llm/                        # 纯LLM算法结果
│   └── models/                 # 模型文件(如有)
├── data/                       # 数据文件
│   ├── csv/                    # CSV格式训练指标
│   │   ├── MADDPG_training_metrics_*.csv
│   │   ├── LLM_MADDPG_training_metrics_*.csv
│   │   └── LLM_training_metrics_*.csv
│   └── json/                   # JSON格式详细数据
├── plots/                      # 图表文件
├── logs/                       # 日志文件
├── test_results/               # 测试结果
└── comparison/                 # 算法对比结果
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch numpy gymnasium matplotlib pyyaml pandas requests

# 配置LLM服务器(可选，仅训练时需要)
# 编辑 config.yaml 中的 llm 配置项
```

### 2. 使用统一入口(推荐)

```bash
# 完整训练和测试流程
python main.py --mode all

# 仅训练所有算法
python main.py --mode train_only

# 仅测试所有算法
python main.py --mode test_only

# 训练特定算法
python main.py --mode llm_maddpg_only
python main.py --mode maddpg_only
python main.py --mode llm_only

# 测试特定算法
python main.py --mode test_maddpg_only --model-path results/experiment_YYYYMMDD_HHMMSS/maddpg/models
python main.py --mode test_llm_maddpg_only --model-path results/experiment_YYYYMMDD_HHMMSS/llm_maddpg/models
python main.py --mode test_llm_only --model-path results/experiment_YYYYMMDD_HHMMSS/llm/models

```

### 3. 🎮 服务器GPU训练（新增）

#### 环境检查
```bash
# 检查GPU环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

#### 指定GPU训练
```bash
# 使用GPU 0进行完整训练+测试
python main.py --gpu 0 --mode all --server-mode

# 使用GPU 1，自定义训练轮数
python main.py --gpu 1 --episodes 1000 --mode all --server-mode

# 批量训练模式（按顺序训练所有算法）
python main.py --gpu 0 --batch-train --episodes 500 --server-mode
```

#### 服务器模式参数说明
- `--gpu`: 指定GPU ID（如：0, 1, 2...）
- `--episodes`: 自定义训练轮数（覆盖配置文件设置）
- `--server-mode`: 显示详细信息和GPU内存使用
- `--batch-train`: 批量训练模式，按顺序执行所有算法
- `--seed`: 设置随机种子确保可复现性

#### 批量训练流程
```bash
# 完整的服务器批量训练（推荐）
python main.py --gpu 0 --batch-train --episodes 1000 --server-mode --seed 42
```

**执行顺序：**
1. 🔥 训练MADDPG → 😴 休息10秒
2. 🔥 训练LLM+MADDPG → 😴 休息10秒  
3. 🔥 训练LLM
4. 🧪 测试MADDPG
5. 🧪 测试LLM+MADDPG
6. 🧪 测试LLM
7. 📊 生成对比报告

#### 单算法GPU训练
```bash
# 仅在GPU上训练MADDPG
python main.py --gpu 0 --mode maddpg_only --episodes 500 --server-mode

# 仅在GPU上训练LLM+MADDPG
python main.py --gpu 1 --mode llm_maddpg_only --episodes 800 --server-mode

# 仅在GPU上训练LLM
python main.py --gpu 0 --mode llm_only --episodes 300 --server-mode
```

#### 多GPU并行训练（高级用法）
```bash
# 终端1：GPU 0训练MADDPG
python main.py --gpu 0 --mode maddpg_only --episodes 1000 --server-mode

# 终端2：GPU 1训练LLM+MADDPG
python main.py --gpu 1 --mode llm_maddpg_only --episodes 1000 --server-mode

# 终端3：GPU 2训练LLM（如果有第三个GPU）
python main.py --gpu 2 --mode llm_only --episodes 500 --server-mode
```

## 📊 统一路径管理系统

### 自动目录结构
每次实验都会创建带时间戳的独立目录：
```
results/experiment_20250703_195030/
├── maddpg/                     # 纯MADDPG算法结果
│   ├── models/                 # 模型文件
│   │   ├── actor_agent_0_final.pth
│   │   ├── critic_agent_0_final.pth
│   │   └── ...
├── llm_maddpg/                 # LLM+MADDPG算法结果
│   ├── models/                 # 模型文件
│   │   ├── agent_0_final.pth
│   │   ├── agent_1_final.pth
│   │   └── ...
├── llm/                        # 纯LLM算法结果
│   └── models/                 # 模型文件(如有)
├── data/                       # 数据文件
│   ├── csv/                    # CSV格式训练指标
│   │   ├── MADDPG_training_metrics_*.csv
│   │   ├── LLM_MADDPG_training_metrics_*.csv
│   │   └── LLM_training_metrics_*.csv
│   └── json/                   # JSON格式详细数据
├── plots/                      # 图表文件
├── logs/                       # 日志文件
├── test_results/               # 测试结果
└── comparison/                 # 算法对比结果
```

### 模型文件格式
- **纯MADDPG**: 分离格式 - `actor_agent_{i}_final.pth`, `critic_agent_{i}_final.pth`
- **LLM+MADDPG**: 完整格式 - `agent_{i}_final.pth`
- **纯LLM**: 无需保存模型文件

## 🧪 算法特性

### 知识蒸馏机制

1. **训练阶段**: LLM提供专家动作，Agent学习模仿
2. **蒸馏损失**: `L_distill = MSE(agent_action, llm_action)`
3. **总损失**: `L_total = L_policy + α * L_distill`
4. **内化效果**: Agent逐渐学会独立决策


### 动作空间

```python
# 三元分割决策
action = [α1, α2, α3, edge_id]
# α1: 本地执行比例
# α2: 边缘执行比例  
# α3: 云端执行比例 (α1 + α2 + α3 = 1)
# edge_id: 目标边缘服务器ID (0-4)
```

## 📊 性能指标

- **能耗效率**: 总能耗消耗 (J)
- **时延性能**: 平均任务完成时延 (s)
- **资源利用**: 系统资源利用率 (%)
- **完成率**: 任务截止时间满足率 (%)

## 🎯 核心优势

### 1. 统一路径管理
- 自动创建标准化目录结构
- 时间戳标识的独立实验
- 算法特定的文件组织
- 支持训练和测试结果分离

### 2. 知识蒸馏效果
- Agent通过训练学习LLM决策模式
- 测试时无需LLM，降低推理成本
- 保持决策质量，提升响应速度

### 3. 部署优势
- **低延迟**: 无需等待LLM推理
- **高可靠**: 不依赖外部LLM服务
- **低成本**: 减少计算资源消耗

### 4. 数据管理
- CSV格式的标准化训练指标
- JSON格式的详细实验数据
- 自动算法对比报告
- 完整的实验追踪记录

## 🔍 实验验证

### 使用统一入口验证
```bash
# 完整验证流程
python main.py --mode all --seed 42

# 查看结果
ls results/experiment_*/
```

### 预期结果
- LLM+MADDPG纯Agent模式应接近有LLM指导的性能
- 相比纯MADDPG应有显著性能提升
- 响应时间大幅降低(无LLM推理延迟)
- 所有结果保存在标准化目录结构中

## ⚠️ 注意事项

1. **训练依赖**: 训练阶段需要LLM服务器支持
2. **测试独立**: 测试阶段完全不依赖LLM
3. **路径管理**: 系统自动管理所有文件路径
4. **模型兼容**: 支持多种模型文件格式自动检测
5. **配置匹配**: 测试配置应与训练环境配置一致

## 🔄 开发迭代

### 当前版本特性
- ✅ 完成LLM+MADDPG训练框架
- ✅ 实现知识蒸馏机制
- ✅ 支持纯Agent模式测试
- ✅ 统一路径管理系统
- ✅ 标准化数据保存格式
- ✅ 完整性能对比分析

### 后续计划
- 🔲 优化知识蒸馏算法
- 🔲 增加更多评估指标
- 🔲 支持在线学习模式
- 🔲 模型压缩与加速

## 📧 联系方式

如有问题，请通过Issue或邮件联系开发团队。

