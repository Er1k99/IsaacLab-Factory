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

## 训练

推荐直接使用 IsaacLab 的启动器运行当前仓库里的脚本：

```bash
~/Isaaclab2.3.2/isaaclab.sh -p /home/eric/IsaacLab-Factory/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Direct-v0 \
  --headless
```

这些脚本会自动把当前仓库 `source/` 和常见的本地 IsaacLab 源码目录
`~/Isaaclab2.3.2/source/{isaaclab,isaaclab_rl,isaaclab_assets}` 加入 `sys.path`，
所以即使你还没有把这些源码包单独 `pip install -e`，通常也可以直接运行。

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
