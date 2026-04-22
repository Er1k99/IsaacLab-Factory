# IsaacLab-Factory

这是一个独立的 IsaacLab 任务仓库，当前只包含一个本地注册的 `PegInsert` 直接强化学习环境：

- task id: `Isaac-Factory-PegInsert-Local-Direct-v0`
- Python 包名: `isaaclab-factory-tasks`
- import 名称: `isaaclab_factory_tasks`

仓库目标不是复刻整个上游 Factory 任务集，而是把 `PegInsert` 从原始多任务环境中单独抽出来，整理成一个更容易训练、回放和评估的本地任务包。

## 当前能力

- 注册本地 Gym 环境 `Isaac-Factory-PegInsert-Local-Direct-v0`
- 提供 RL-Games 训练脚本、回放脚本和评估脚本
- 支持以下 agent 配置
  - `PPO_GRU`
  - `PPO_LSTM`
  - `PPO_MLP`
  - `PPO_TRANSFORMER`
  - `PPO_TRANSFORMER_GRU`
  - `SAC`
- 提供一个简单的 scripted baseline 用于离线评估
- 为 RL-Games 增加两类本地扩展
  - Transformer PPO 的时序观测堆叠封装
  - SAC 的 asymmetric actor/critic 输入与终止观测修复

## 环境概览

当前任务使用 Franka 机械臂执行 8mm peg-hole 插销任务，环境配置来自 `source/isaaclab_factory_tasks/direct/peg_insert/`。

默认配置要点：

- 环境类：`PegInsertEnv`
- 配置类：`PegInsertEnvCfg`
- 默认并行环境数：`128`
- 动作维度：`6`
- 默认 episode 时长：`10.0 s`
- 默认仿真设备：`cuda:0`

任务资源依赖 IsaacLab / Nucleus 中的 Factory 资产，例如：

- `Factory/factory_peg_8mm.usd`
- `Factory/factory_hole_8mm.usd`
- `Props/Mounts/SeattleLabTable/table_instanceable.usd`

因此推荐始终在一个正常可运行的 IsaacLab / Isaac Sim 环境内启动本仓库脚本。

## 依赖与前提

最小前提不是单纯的 Python 环境，而是一个可运行的 IsaacLab 运行时。训练、回放和评估脚本都会直接导入：

- `isaaclab`
- `isaaclab_rl`
- `isaaclab_assets`
- `rl_games`

仓库自身的 `pyproject.toml` 还声明了这些基础 Python 依赖：

- `gymnasium`
- `hydra-core`
- `numpy<2`
- `prettytable`
- `PyYAML`
- `torch>=2.7`

推荐做法：

1. 先准备好 IsaacLab / Isaac Sim 运行环境。
2. 进入该运行环境后，再安装本仓库或直接用 `isaaclab.sh -p` 运行脚本。

## 路径解析约定

本仓库脚本会在启动时自动补充 Python 搜索路径，按下面顺序尝试发现 IsaacLab 源码：

1. 当前仓库的 `source/`
2. 环境变量 `ISAACLAB_ROOT`
3. 环境变量 `ISAACLAB_PATH`
4. `~/Isaaclab2.3.2`
5. `~/IsaacLab`

如果你的 IsaacLab 不在这些位置，先显式设置：

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
```

## 安装

如果你希望在交互式 Python、notebook 或其他自定义脚本里直接 `import isaaclab_factory_tasks`，建议在 IsaacLab 环境中执行：

```bash
pip install -e .
```

仅使用本仓库自带脚本时，即使不执行 `pip install -e .`，通常也能运行，因为脚本会自动把仓库 `source/` 加入 `sys.path`。不过为了避免环境差异，仍然建议安装一次。

## 快速开始

下面所有命令默认在仓库根目录执行。

### 1. 列出已注册环境

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/environments/list_envs.py --keyword PegInsert
```

如果注册成功，你会看到 `Isaac-Factory-PegInsert-Local-Direct-v0`。

### 2. 训练

训练脚本：

- `scripts/reinforcement_learning/rl_games/train.py`

基础命令：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --headless
```

启用 Weights & Biases：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --track \
  --wandb-entity <your_wandb_entity> \
  --headless
```

训练 SAC：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm SAC \
  --headless
```

训练 Transformer PPO：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_TRANSFORMER \
  --headless
```

### 3. 回放 checkpoint

回放脚本：

- `scripts/reinforcement_learning/rl_games/play.py`

使用默认 best checkpoint 回放：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --num_envs 1
```

使用最新 checkpoint 回放：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --use_last_checkpoint \
  --num_envs 1
```

指定 checkpoint 回放：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm SAC \
  --checkpoint /path/to/checkpoint.pth \
  --num_envs 1
```

常用可选参数：

- `--video`：录制回放视频，输出到对应 run 目录下的 `videos/play/`
- `--real-time`：尽量按真实时间步长运行
- `--print_peg_metrics`：定期打印 peg 高度 / lift 高度诊断信息
- `--disable_fabric`：调试时可关闭 fabric

### 4. 评估 scripted baseline 或 RL checkpoint

评估脚本：

- `scripts/reinforcement_learning/rl_games/eval.py`

这个脚本会强制 `headless=True`，主要输出成功率、成功步数和成功时间统计，不负责可视化回放。


```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode scripted \
  --checkpoint /path/to/checkpoint.pth
```

评估 PPO_GRU：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode ppo_gru \
  --checkpoint /path/to/checkpoint.pth
```

评估 SAC：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode sac \
  --checkpoint /path/to/checkpoint.pth
```

## 算法与默认实验名

训练 / 回放时的 `--algorithm`，以及评估时的 `--eval_mode`，与默认实验目录名的对应关系如下：

| 类型 | train/play 参数 | eval 参数 | 默认实验名 |
| --- | --- | --- | --- |
| PPO GRU | `PPO_GRU` | `ppo_gru` | `FactoryPegInsertGRU` |
| PPO LSTM | `PPO_LSTM` | `ppo_lstm` | `FactoryPegInsertLSTM` |
| PPO MLP | `PPO_MLP` | `ppo_mlp` | `FactoryPegInsertMLP` |
| PPO Transformer | `PPO_TRANSFORMER` | `ppo_transformer` | `FactoryPegInsertTransformer` |
| PPO Transformer + GRU | `PPO_TRANSFORMER_GRU` | `ppo_transformer_gru` | `FactoryPegInsertTransformerGRU` |
| SAC | `SAC` | `sac` | `FactoryPegInsertSAC` |

补充说明：

- `PPO` 是 `PPO_GRU` 的兼容别名
- `play.py` 默认加载 best checkpoint，即 `<默认实验名>.pth`
- 加上 `--use_last_checkpoint` 后，`play.py` / `eval.py` 会改为加载最近一次保存的 checkpoint

## Hydra 覆盖

`train.py` 和 `play.py` 通过 Hydra 读取环境和 agent 配置，所以可以在命令末尾直接附加配置覆盖项。

例如：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --headless \
  env.scene.num_envs=256 \
  agent.params.config.max_epochs=400
```

脚本会自动把 Hydra 输出目录固定到当前仓库下的 `outputs/`，避免输出散落到外部目录。

## 输出目录

训练和评估时，和本仓库直接相关的输出主要落在以下位置：

```text
logs/
└── rl_games/
    └── <ExperimentName>/
        └── <RunDir>/
            ├── nn/
            ├── params/
            └── videos/

outputs/
└── <Hydra 输出目录>

wandb/
├── runs/
├── artifacts/
├── data/
└── .cache/
```

其中：

- `params/env.yaml` 和 `params/agent.yaml` 会在训练启动时自动保存
- `nn/` 中保存 RL-Games checkpoint
- 训练视频输出到 `videos/train/`
- 回放视频输出到 `videos/play/`
- `wandb/` 下的本地缓存、artifact 和 run 数据会被固定在仓库内

仓库根目录下还包含一个 `best_model/` 目录，里面是已保存的参考 checkpoint；默认脚本不会自动从这里检索模型，如果要使用其中某个模型，请显式传 `--checkpoint`。

## 目录结构

```text
source/isaaclab_factory_tasks/
├── __init__.py
├── direct/
│   └── peg_insert/
│       ├── __init__.py
│       ├── agents/
│       │   ├── rl_games_ppo_gru_cfg.yaml
│       │   ├── rl_games_ppo_lstm_cfg.yaml
│       │   ├── rl_games_ppo_mlp_cfg.yaml
│       │   ├── rl_games_ppo_transformer_cfg.yaml
│       │   ├── rl_games_ppo_transformer_gru_cfg.yaml
│       │   └── rl_games_sac_cfg.yaml
│       ├── control.py
│       ├── env.py
│       ├── env_cfg.py
│       ├── task_cfg.py
│       └── utils.py
└── utils/
    ├── hydra.py
    ├── importer.py
    ├── parse_cfg.py
    ├── rl_games_sac.py
    └── rl_games_transformer.py

scripts/
├── environments/
│   └── list_envs.py
└── reinforcement_learning/
    └── rl_games/
        ├── train.py
        ├── play.py
        └── eval.py
```

几个关键文件的作用：

- `source/isaaclab_factory_tasks/direct/peg_insert/__init__.py`
  - 注册 `Isaac-Factory-PegInsert-Local-Direct-v0`
- `source/isaaclab_factory_tasks/direct/peg_insert/agents/*.yaml`
  - 存放各类 RL-Games agent 配置
- `source/isaaclab_factory_tasks/utils/rl_games_transformer.py`
  - 实现本地 Transformer PPO 组件和时序观测 wrapper
- `source/isaaclab_factory_tasks/utils/rl_games_sac.py`
  - 实现本地 SAC 扩展


## 常见问题

### 1. `ModuleNotFoundError: No module named 'isaaclab_factory_tasks'`

优先检查：

- 是否从仓库脚本启动，而不是从其他目录直接运行零散代码
- 是否已经在 IsaacLab 环境中执行 `pip install -e .`
- 是否把 `ISAACLAB_ROOT` 指到了正确的 IsaacLab 目录

### 2. `Could not import IsaacLab runtime modules`

这说明当前 Python 环境里没有可用的 IsaacLab / Isaac Sim 运行时。请改用 IsaacLab 的启动器执行，例如：

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --help
```

### 3. 找不到 checkpoint

`play.py` 和 `eval.py` 在未显式传 `--checkpoint` 时，会去：

```text
logs/rl_games/<默认实验名>/<run_dir>/nn/
```

下查找模型。

如果你要加载旧实验或其他目录中的模型，请显式传入：

```bash
--checkpoint /path/to/model.pth
```

### 4. 评估脚本为什么没有画面

`scripts/reinforcement_learning/rl_games/eval.py` 会强制以 `headless` 模式运行。它的目的不是可视化，而是统计成功率和成功时间。需要看可视化请使用 `play.py`。

## 当前仓库范围说明

当前代码实际注册和维护的是本地 `PegInsert` 任务。仓库中虽然还存在少量其他调试脚本，但它们不属于当前主流程，也不代表这里已经完整提供了对应任务包。
