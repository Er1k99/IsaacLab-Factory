# IsaacLab-Factory

这个仓库把上游 `~/Isaaclab2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/factory` 里的
`Isaac-Factory-PegInsert-Direct-v0` 单独抽出来做了重构，只保留 `PegInsert` 任务需要的代码路径。

现在仓库里已经补齐了和 IsaacLab 常见任务仓库一致的最小运行链路：

- `source/isaaclab_factory_tasks/`：任务包、注册入口、Hydra/registry 工具
- `scripts/reinforcement_learning/rl_games/`：训练与回放脚本
- `scripts/environments/list_envs.py`：列出当前仓库注册出的环境
- `config/extension.toml`：IsaacLab 风格扩展元数据

## 目录

```text
source/isaaclab_factory_tasks/
├── __init__.py
└── direct/
    ├── __init__.py
    └── peg_insert/
        ├── __init__.py
        ├── agents/
        │   ├── __init__.py
        │   └── rl_games_ppo_cfg.yaml
        ├── control.py
        ├── env.py
        ├── env_cfg.py
        ├── task_cfg.py
        └── utils.py
```

## 重构目标

- 只保留 `PegInsert`，去掉 `gear_mesh` 和 `nut_thread` 的分支判断
- 把原来单个大 `FactoryEnv` 中的任务专属逻辑拆干净
- 保留原任务 ID，方便在干净进程里继续使用 `--task Isaac-Factory-PegInsert-Direct-v0`
- 额外注册一个别名 `Isaac-Factory-PegInsert-Refactored-Direct-v0`，避免和上游包同时导入时发生冲突

## 安装

```bash
pip install -e .
```

如果你是在 IsaacLab 自带 Python 环境里运行，建议先激活 IsaacLab 环境，再执行上面的安装。

## 使用

先确保当前包被导入，这样 Gym 环境会完成注册：

```python
import isaaclab_factory_tasks  # noqa: F401
```

如果当前 Python 进程里没有先导入上游 `isaaclab_tasks`，可以继续用原任务名：

```bash
--task Isaac-Factory-PegInsert-Direct-v0
```

如果上游同名任务已经先注册，使用这里额外提供的别名：

```bash
--task Isaac-Factory-PegInsert-Refactored-Direct-v0
```

当前仓库还额外提供了一个只学习抓取的基线任务：

```bash
--task Isaac-Factory-PegPick-Direct-v0
```

## 训练

推荐直接使用 IsaacLab 的启动器运行当前仓库里的脚本：

```bash
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Direct-v0 \
  --track \
  --wandb-entity <your_entity> \
  --headless
```

这些脚本会自动把当前仓库 `source/` 和常见的本地 IsaacLab 源码目录
`~/Isaaclab2.3.2/source/{isaaclab,isaaclab_rl,isaaclab_assets}` 加入 `sys.path`，
所以即使你还没有把这些源码包单独 `pip install -e`，通常也可以直接运行。

当前训练脚本的保存与日志逻辑是：

- 训练日志默认写到当前仓库下的 `logs/rl_games/FactoryPegInsert/<timestamp>/`
- Hydra 默认输出目录也固定在当前仓库下的 `outputs/`
- wandb 的本地运行目录、缓存和 artifact 暂存也固定在当前仓库下的 `wandb/`
- 每次运行都会自动附带时间戳，避免覆盖旧日志和旧模型
- `params/env.yaml` 和 `params/agent.yaml` 会自动保存到对应实验目录
- RL-Games 会每 `50` 个 epoch 做一次周期性保存
- 训练结束后，如果启用了 `--track`，会自动把 `nn/` 下的最佳和最新 checkpoint 上传到 wandb artifact

如果只想记录 wandb 标量，不上传模型，可以额外传：

```bash
--wandb-upload-model False
```

如果只想快速看环境是否注册成功：

```bash
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/environments/list_envs.py --keyword PegInsert
```

训练完成后回放：

```bash
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Direct-v0 \
  --use_last_checkpoint
```

插销训练命令：
```bash
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/train.py   --task Isaac-Factory-PegInsert-Direct-v0   --track   --wandb-entity jiadapeng4-sasfs   --headless
```

插销play命令：
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Refactored-Direct-v0 \
  --checkpoint /home/eric/IsaacLab-Factory/last_Factory_ep_200_rew_370.69455.pth \
  --num_envs 1



抓取训练命令：
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegPick-Direct-v0 \
  --track \
  --wandb-entity jiadapeng4-sasfs \
  --headless

抓取play命令：
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegPick-Direct-v0 \
  --checkpoint /home/eric/IsaacLab-Factory/logs/rl_games/FactoryPegPick/2026-04-14_22-08-08/nn/last_FactoryPegPick_ep_200_rew_128.242.pth \
  --num_envs 1


/home/eric/IsaacLab-Factory/logs/rl_games/FactoryPegPick/2026-04-14_18-03-47/nn/last_FactoryPegPick_ep_50_rew_74.41704.pth