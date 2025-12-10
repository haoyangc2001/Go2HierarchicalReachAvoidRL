# Go2 Hierarchical Reach-Avoid RL

## 项目概述

Go2 Hierarchical Reach-Avoid RL 是一个基于强化学习的机器人导航系统，专为 Unitree Go2 机器人设计。该项目实现了一个分层控制架构，结合了预训练的低层级运动策略和可训练的高层级导航策略，使机器人能够在复杂环境中安全导航并到达目标位置。

## 核心功能

### 1. 分层控制架构

#### 低层级（Locomotion）
- **功能**：将速度指令（线速度和角速度）转换为机器人的关节动作
- **实现**：预训练的运动控制策略，基于 PPO 算法训练
- **输入**：机器人状态（关节角度、速度、IMU 数据等）和速度指令
- **输出**：关节目标位置或力矩
- **文件位置**：
  - 环境封装：`legged_gym/envs/go2/go2_env.py`
  - 低层级策略加载：`legged_gym/envs/go2/hierarchical_go2_env.py` 中的 `_load_low_level_policy` 方法

#### 高层级（Navigation）
- **功能**：将环境观测转换为速度指令，实现避障导航
- **实现**：可训练的导航策略，基于 Reach-Avoid PPO 算法
- **输入**：环境观测（机器人位置、障碍物信息、目标位置、速度等）
- **输出**：速度指令（线速度和角速度）
- **文件位置**：
  - 导航环境：`legged_gym/envs/go2/high_level_navigation_env.py`
  - 高层级配置：`legged_gym/envs/go2/high_level_navigation_env.py` 中的 `HighLevelNavigationConfig` 类

#### 层级连接机制
- **动作重复机制**：高层级输出的速度指令在低层级重复执行多次
- **参数配置**：通过 `high_level_action_repeat` 参数控制，默认为 1
- **文件位置**：`legged_gym/envs/go2/hierarchical_go2_env.py` 中的 `low_level_action_repeat` 属性

#### 分层环境封装
- **统一接口**：`HierarchicalVecEnv` 类封装了分层环境，提供标准的 RL 环境接口
- **文件位置**：`legged_gym/scripts/train_reach_avoid.py` 中的 `HierarchicalVecEnv` 类
- **核心方法**：`reset()`、`step()`、`close()`

### 2. Reach-Avoid 强化学习算法

Reach-Avoid 强化学习算法是专为避障导航任务设计的 PPO（Proximal Policy Optimization）扩展版本，实现了到达目标位置（reach）和避免障碍物（avoid）的双重目标。

#### 2.1 核心问题定义

- **Reach 目标**：机器人需要到达指定目标位置，由 `g_values` 表示（负值表示成功到达）
- **Avoid 约束**：机器人必须避免与障碍物碰撞，由 `h_values` 表示（非负值表示碰撞）
- **状态空间**：机器人位置、速度、障碍物信息、目标位置等环境观测
- **动作空间**：速度指令（线速度和角速度）

#### 2.2 算法核心组件

##### ReachAvoidPPO 类
- **功能**：实现完整的 Reach-Avoid PPO 算法
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py`
- **核心方法**：
  - `act()`：根据观测生成动作
  - `update()`：更新策略网络
  - `init_storage()`：初始化经验缓冲区

##### 自定义优势函数计算
- **功能**：计算适用于 Reach-Avoid 任务的广义优势估计（GAE）
- **实现**：`_calculate_reach_gae` 函数
- **特点**：
  - 考虑了 reach 和 avoid 双重目标
  - 基于 JAX 参考实现的 PyTorch 移植
  - 支持多环境并行计算
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py` 中的 `_calculate_reach_gae` 函数

##### 经验回放缓冲区
- **功能**：存储和管理训练数据
- **实现**：`ReachAvoidBuffer` 类
- **特点**：
  - 存储 `g_values` 和 `h_values` 用于避障任务
  - 支持批量采样和多轮训练
  - 实现了自定义的优势函数计算
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py` 中的 `ReachAvoidBuffer` 类

##### 数据批次处理
- **功能**：将经验数据转换为训练批次
- **实现**：`ReachAvoidBatch` 数据类
- **特点**：
  - 封装了训练所需的所有数据
  - 支持扁平化数据视图
  - 便于批量采样和训练
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py` 中的 `ReachAvoidBatch` 类

#### 2.3 算法更新流程

1. **经验收集**：通过与环境交互收集轨迹数据
2. **完成标志计算**：
   - 环境完成：`env_dones = self.dones`
   - 安全违规：`safety_dones = self.h_values[:-1] >= 0`
   - 最终完成标志：`done_seq = torch.logical_or(env_dones, safety_dones)`
3. **优势函数计算**：调用 `_calculate_reach_gae` 计算优势和目标值
4. **优势归一化**：对优势函数进行归一化处理
5. **策略更新**：
   - 计算新的动作概率和价值估计
   - 计算策略损失（裁剪目标）
   - 计算价值损失（裁剪目标）
   - 计算熵正则化项
   - 总损失：`loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy`
   - 反向传播和梯度裁剪
   - 优化器更新

#### 2.4 关键参数配置

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `learning_rate` | 学习率 | 3e-4 |
| `gamma` | 折扣因子 | 0.999 |
| `lam` | GAE 参数 | 0.95 |
| `num_learning_epochs` | 每次迭代的训练轮数 | 4 |
| `num_mini_batches` | 每次迭代的 mini-batch 数量 | 4 |
| `clip_param` | PPO 裁剪参数 | 0.2 |
| `value_loss_coef` | 价值损失权重 | 1.0 |
| `entropy_coef` | 熵正则化权重 | 0.0 |
| `max_grad_norm` | 梯度裁剪阈值 | 1.0 |

#### 2.5 成功指标计算

- **功能**：评估机器人在轨迹中的避障导航成功率
- **实现**：`compute_reach_avoid_success_rate` 函数
- **计算逻辑**：
  1. 检查是否成功到达目标（`g_values < 0`）
  2. 记录首次成功到达的时间步
  3. 检查在成功前是否发生碰撞（`h_values >= 0`）
  4. 成功率 = （成功到达目标且未碰撞的环境数）/ 总环境数
- **文件位置**：`legged_gym/scripts/train_reach_avoid.py` 中的 `compute_reach_avoid_success_rate` 函数

#### 2.6 与标准 PPO 的区别

1. **目标函数**：
   - 标准 PPO：单一奖励信号
   - ReachAvoidPPO：双重目标（reach + avoid）

2. **优势函数**：
   - 标准 PPO：基于奖励信号的 GAE
   - ReachAvoidPPO：基于 `g_values` 和 `h_values` 的自定义 GAE

3. **完成标志**：
   - 标准 PPO：仅环境完成
   - ReachAvoidPPO：环境完成或安全违规

4. **经验缓冲区**：
   - 标准 PPO：存储奖励和完成标志
   - ReachAvoidPPO：额外存储 `g_values` 和 `h_values`

5. **训练流程**：
   - 标准 PPO：基于奖励的策略更新
   - ReachAvoidPPO：考虑双重目标的策略更新

### 3. 环境与仿真

- 基于 Isaac Gym 物理引擎的高性能仿真环境
- 支持多环境并行训练
- 提供丰富的环境观测和奖励信号

### 4. 训练流程

执行 `train_reach_avoid.py` 后，项目将启动完整的训练流程：

1. **环境初始化**：创建 Go2 机器人仿真环境
2. **低层级策略加载**：加载预训练的运动控制策略
3. **高层级策略初始化**：初始化导航策略网络
4. **经验收集**：通过与环境交互收集轨迹数据
5. **策略更新**：使用 PPO 算法更新高层级导航策略
6. **性能评估**：计算并监控避障任务成功率
7. **模型保存**：定期保存训练好的模型检查点

## 技术架构

### 主要组件

| 组件 | 描述 | 位置 |
|------|------|------|
| HierarchicalGO2Env | 分层环境封装 | `legged_gym/envs/go2/hierarchical_go2_env.py` |
| ReachAvoidPPO | 避障强化学习算法 | `rsl_rl/algorithms/reach_avoid_ppo.py` |
| ActorCritic | 策略网络架构 | `rsl_rl/modules/actor_critic.py` |
| HighLevelNavigationEnv | 高层级导航封装 | `legged_gym/envs/go2/high_level_navigation_env.py` |

### 网络架构

- **Actor 网络**：4 层全连接网络，每层 512 个单元
- **Critic 网络**：4 层全连接网络，每层 512 个单元
- 激活函数：ReLU
- 初始化噪声：标准差为 0.1 的高斯噪声

## 环境配置

### 安装步骤

#### 系统要求
- **操作系统**：推荐 Ubuntu 18.04 或更高版本
- **GPU**：NVIDIA GPU
- **驱动版本**：推荐 525 或更高版本

#### 详细安装步骤

1. **克隆项目仓库**
   ```bash
   git clone https://github.com/haoyangc2001/Go2HierarchicalReachAvoidRL.git
   cd Go2HierarchicalReachAvoidRL
   ```

2. **创建虚拟环境（推荐使用 Conda）**
   ```bash
   # 安装 Miniconda（如果未安装）
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   
   # 初始化 Conda
   ~/miniconda3/bin/conda init --all
   source ~/.bashrc
   
   # 创建并激活虚拟环境
   conda create -n go2-rl python=3.8
   conda activate go2-rl
   ```

3. **安装依赖项**
   
   - **安装 PyTorch**
     ```bash
     conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
     ```
   
   - **安装 Isaac Gym**
     ```bash
     # 1. 从 NVIDIA 官网下载 Isaac Gym：https://developer.nvidia.com/isaac-gym
     # 2. 解压到当前项目目录下的 isaacgym 文件夹
     
     # 安装 Isaac Gym Python 绑定
     cd isaacgym/python
     pip install -e .
     
     # 验证安装（可选）
     cd examples
     python 1080_balls_of_solitude.py
     ```
   
   - **安装项目依赖**
     ```bash
     # 返回项目根目录
     cd ../..
     cd legged_gym_go2
     pip install -e .
     ```
   
   - **安装其他依赖**
     ```bash
     pip install numpy matplotlib torchvision
     ```


## 使用说明

### 运行训练脚本

确保已经激活虚拟环境并配置好环境变量后，运行以下命令：

```bash
# 从项目根目录运行
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py
```

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| --headless | 是否在无头模式下运行 | True |
| --rl_device | RL 计算设备 | cuda:1 |
| --sim_device | 仿真设备 | cuda:1 |
| --compute_device_id | 计算设备 ID | 1 |
| --sim_device_id | 仿真设备 ID | 1 |
| --task | 任务名称（固定为 go2） | go2 |
| --resume | 是否从 checkpoint 恢复训练 | False |
| --experiment_name | 实验名称 | high_level_go2 |
| --run_name | 运行名称 | 自动生成时间戳 |
| --num_envs | 并行训练环境数量 | 32 |
| --seed | 随机种子 | 42 |
| --max_iterations | 最大训练迭代次数 | 10000 |

### 示例命令

#### 基本训练
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py
```

#### 无头模式训练（更高效率）
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --headless=true
```

#### 使用不同 GPU
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --rl_device=cuda:0 --sim_device=cuda:0 --compute_device_id=0 --sim_device_id=0
```

#### 恢复训练
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --resume=true --experiment_name=high_level_go2
```

### 配置文件

- 环境配置：`GO2HighLevelCfg`
- 训练配置：`GO2HighLevelCfgPPO`
- 可通过命令行参数覆盖配置值

## 训练结果

### 输出信息

训练过程中，脚本会输出以下关键信息：

```
iter 00001 | success 0.000 | policy_loss -0.00123 | value_loss 0.12345 | Vmean 0.567 | Rmean 0.890 | Vrmse 0.123 | VexpVar 0.456 | adv_std 0.789 | elapsed 1.23s
```

### 生成文件

- **模型检查点**：`logs/<experiment_name>/<timestamp>/model_<iteration>.pt`
- **训练日志**：控制台输出，可重定向到文件
- **GH 快照**：（可选）定期保存的状态快照

## 项目结构

```
Go2HierarchicalReachAvoidRL/
├── legged_gym_go2/
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   └── go2/
│   │   │       ├── hierarchical_go2_env.py    # 分层环境封装
│   │   │       ├── high_level_navigation_env.py # 高层级导航环境
│   │   │       └── go2_env.py                   # 基础 Go2 环境
│   │   ├── scripts/
│   │   │   └── train_reach_avoid.py            # 训练脚本
│   │   └── utils/
│   └── rsl_rl/
│       ├── algorithms/
│       │   └── reach_avoid_ppo.py              # 避障 PPO 算法
│       └── modules/
│           └── actor_critic.py                 # 策略网络
└── logs/                                       # 训练日志和模型保存目录
```

## 训练流程详解

### 1. 环境创建

`create_env` 函数创建分层环境：

```python
def create_env(env_cfg, train_cfg, args, device) -> HierarchicalVecEnv:
    base_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )
    return HierarchicalVecEnv(base_env)
```

### 2. 经验收集

训练循环中，脚本收集固定长度的轨迹数据：

```python
for step in range(horizon):
    actions, log_probs, values = alg.act(rollout_obs[step])
    next_obs, next_g, next_h, dones, _ = env.step(actions)
    # 存储轨迹数据
```

### 3. 成功率计算

`compute_reach_avoid_success_rate` 函数评估避障任务成功率：

```python
def compute_reach_avoid_success_rate(g_sequence, h_sequence) -> float:
    # g_sequence: 目标达成指标序列
    # h_sequence: 障碍物规避指标序列
    # 返回成功率
```

### 4. 策略更新

使用 PPO 算法更新高层级策略：

```python
policy_loss, value_loss = alg.update()
```

## 扩展与自定义

### 1. 修改网络架构

在 `train_reach_avoid` 函数中修改网络尺寸：

```python
train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]  # Actor 网络
 train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]  # Critic 网络
```

### 2. 调整训练参数

修改 `GO2HighLevelCfgPPO` 配置：

- 学习率
- 折扣因子
- gae 参数
- 批量大小
- 训练迭代次数

### 3. 自定义环境

扩展 `HierarchicalGO2Env` 类或修改 `HighLevelNavigationConfig` 配置，自定义环境特性：

- 障碍物类型和分布
- 奖励函数
- 观测空间
- 动作空间

## 性能指标

训练过程中监控的关键指标：

- **成功率（success）**：成功到达目标且未碰撞的环境比例
- **策略损失（policy_loss）**：Actor 网络的损失值
- **价值损失（value_loss）**：Critic 网络的损失值
- **价值均值（Vmean）**：价值函数的均值
- **回报均值（Rmean）**：轨迹回报的均值
- **价值 RMSE（Vrmse）**：价值函数预测误差
- **解释方差（VexpVar）**：价值函数解释回报变化的能力
- **优势标准差（adv_std）**：优势函数的标准差

## 示例应用场景

1. **室内导航**：在办公室或家庭环境中自主导航
2. **仓储物流**：在仓库环境中搬运物品
3. **搜索救援**：在复杂环境中搜索目标
4. **环境监测**：在特定区域内进行环境监测

