# IsaacLab-Factory

- `IsaacLab-Factory` is a standalone IsaacLab task repository.
- The repository is used to train and evaluate the Factory PegInsert task.
- The main maintained Gymnasium environment is:

```text
Isaac-Factory-PegInsert-Local-Direct-v0
```

- The code packages the IsaacLab Factory peg insertion task as an independent task.
- The task supports standalone registration, training, playback, and evaluation.
- The robot is a Franka manipulator.
- The objective is to insert an 8 mm peg into an 8 mm hole.
- The action is a 6D end-effector pose delta.
- GPU-parallel simulation is used by default.

## Feature Overview

- Available workflows:

| Category | Entry Point | Description |
| --- | --- | --- |
| Environment registration | `source/isaaclab_factory_tasks/direct/peg_insert/__init__.py` | Registers `Isaac-Factory-PegInsert-Local-Direct-v0` |
| RL-Games training | `scripts/reinforcement_learning/rl_games/train.py` | Trains PPO / SAC |
| RL-Games playback | `scripts/reinforcement_learning/rl_games/play.py` | Loads checkpoints for visualization or video recording |
| RL-Games evaluation | `scripts/reinforcement_learning/rl_games/eval.py` | Reports success rate, successful steps, and successful time |
| skrl TD3 training | `scripts/reinforcement_learning/skrl/train.py` | Trains TD3 |
| skrl TD3 playback/evaluation | `scripts/reinforcement_learning/skrl/play.py` / `eval_td3.py` | Loads TD3 checkpoints for playback or success-rate evaluation |
| Classic methods | `scripts/reinforcement_learning/classic_method/` | Scripted baseline and two RRT baselines |
| Environment check | `scripts/environments/list_envs.py` | Lists registered environments |

- Supported reinforcement learning agents:

| Framework | Argument | Default Experiment Name | Config File |
| --- | --- | --- | --- |
| RL-Games | `PPO_GRU` | `FactoryPegInsertGRU` | `rl_games_ppo_gru_cfg.yaml` |
| RL-Games | `PPO_LSTM` | `FactoryPegInsertLSTM` | `rl_games_ppo_lstm_cfg.yaml` |
| RL-Games | `PPO_MLP` | `FactoryPegInsertMLP` | `rl_games_ppo_mlp_cfg.yaml` |
| RL-Games | `PPO_TRANSFORMER` | `FactoryPegInsertTransformer` | `rl_games_ppo_transformer_cfg.yaml` |
| RL-Games | `PPO_TRANSFORMER_GRU` | `FactoryPegInsertTransformerGRU` | `rl_games_ppo_transformer_gru_cfg.yaml` |
| RL-Games | `SAC` | `FactoryPegInsertSAC` | `rl_games_sac_cfg.yaml` |
| skrl | `TD3` | `FactoryPegInsertTD3` | `skrl_td3_cfg.yaml` |

- `best_model/` contains reference checkpoints.
- PPO / SAC reference models use the `.pth` format.
- TD3 reference models use the `.pt` format.

## Repository Structure

```text
IsaacLab-Factory/
├── source/isaaclab_factory_tasks/          # Installable Python package for tasks, configs, and algorithm extensions
│   ├── direct/peg_insert/                  # Main PegInsert DirectRLEnv task
│   │   ├── __init__.py                     # Gymnasium task registration entry point
│   │   ├── env.py                          # Environment step/reset/obs/reward/success logic
│   │   ├── env_cfg.py                      # Simulation, robot, action space, observation space, and PhysX config
│   │   ├── task_cfg.py                     # Peg/hole assets, randomization, reward parameters, and success threshold
│   │   ├── control.py                      # OSC / IK control utilities
│   │   ├── utils.py                        # Pose, keypoint, observation concatenation, and physics utilities
│   │   └── agents/                         # RL-Games / skrl agent YAML configs
│   └── utils/                              # Training-framework adapter layer
│       ├── hydra.py                        # IsaacLab registry + Hydra overrides
│       ├── parse_cfg.py                    # Config loading and checkpoint discovery
│       ├── rl_games_sac.py                 # RL-Games SAC extension
│       └── rl_games_transformer.py         # Transformer PPO network and sequence wrapper
├── scripts/                                # Command-line entry points
│   ├── environments/list_envs.py           # Environment registration check
│   └── reinforcement_learning/
│       ├── rl_games/                       # PPO/SAC training, playback, and evaluation
│       ├── skrl/                           # TD3/skrl experiment workflow
│       └── classic_method/                 # Scripted / RRT baselines
├── best_model/                             # Reference checkpoints
├── logs/                                   # Training logs and checkpoint outputs
├── runs/                                   # Historical TensorBoard fallback logs
├── config/extension.toml                   # IsaacLab extension metadata
├── pyproject.toml                          # Package installation and Python dependency metadata
├── requirements.txt                        # Pinned Python dependencies for the current environment
└── README.md
```

## Dependencies And Versions

- Use the runtime launcher provided by IsaacLab.
- Do not run training or evaluation scripts directly with the system Python.
- Confirmed versions in the current environment:

| Software / Library | Confirmed Version | Notes |
| --- | --- | --- |
| Python | `3.11.15` | Current conda environment version |
| Isaac Sim | `5.1.0.0` | Required simulation runtime |
| Isaac Lab | `0.54.2` or `2.3.2` | Required |
| `isaaclab_rl` | `0.4.7` | Required RL wrapper |
| `isaaclab_assets` | `0.2.4` | Required asset configs |
| PyTorch | `2.7.0+cu128` | Required; `pyproject.toml` requires `torch>=2.7` |
| Gymnasium | `1.2.1` | Required |
| Hydra Core | `1.3.2` | Required |
| NumPy | `1.26.0` | Required; `pyproject.toml` requires `numpy<2` |
| PyYAML | `6.0.2` | Required |
| rl-games | `1.6.1` | Required for RL-Games training/playback/evaluation |
| prettytable | `3.3.0` | Environment list output |
| TensorBoard | `2.20.0` | Training-curve visualization |
| Weights & Biases | `0.25.1` | Optional; used when `--track` is enabled |
| skrl | `2.0.0` | Required for TD3/skrl experiments only |

- Minimum dependency constraints declared by this repository are in `pyproject.toml`:

```text
python >= 3.10
torch >= 2.7
numpy < 2
```

- Pinned Python dependencies for the current environment are listed in `requirements.txt`.
- Simulation assets depend on the IsaacLab / Nucleus Factory resources.
- Main assets:

```text
Factory/factory_peg_8mm.usd
Factory/factory_hole_8mm.usd
Props/Mounts/SeattleLabTable/table_instanceable.usd
```

- An NVIDIA GPU is recommended for training and evaluation.
- The default simulation device is `cuda:0`.
- The default number of parallel environments is `128`.

## Installation

- Set Conda environment:
```bash
conda create -n ME5406 python=3.11 -y
conda activate ME5406
```
- Set the IsaacLab root directory:
```bash
export ISAACLAB_ROOT="your_isaaclab_path"
```

- Install this repository in the IsaacLab Python environment:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p -m pip install -e .
```

- Optionally install the pinned Python dependencies:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p -m pip install -r requirements.txt
```

- Repository scripts automatically add `source/` to `sys.path`.
- If you only run the scripts provided in this repository, installation is usually not required.
- If you want to `import isaaclab_factory_tasks` in an interactive Python shell, notebook, or custom script, editable installation is recommended.

## Quick Check

- Run from the repository root:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/environments/list_envs.py --keyword PegInsert
```

- Expected output includes:

```text
Isaac-Factory-PegInsert-Local-Direct-v0
```

## Training

- PPO GRU:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --headless
```

- Transformer + GRU PPO:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_TRANSFORMER_GRU \
  --headless
```

- SAC:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm SAC \
  --headless
```

- TD3:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm TD3 \
  --headless
```

- Common arguments:

| Argument | Description |
| --- | --- |
| `--num_envs` | Overrides the number of parallel environments |
| `--max_iterations` | Overrides the number of training iterations |
| `--checkpoint` | Resumes training from a specified checkpoint |
| `--track` | Enables W&B |
| `--wandb-entity` | W&B entity |
| `--sigma` | Initial PPO policy standard deviation |

## Playback

- Load the default best checkpoint:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --num_envs 32
```

- Load the latest checkpoint:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_GRU \
  --use_last_checkpoint \
  --num_envs 32
```

- Load a specified checkpoint:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --algorithm PPO_TRANSFORMER_GRU \
  --checkpoint best_model/FactoryPegInsertTransformerGRU/nn/FactoryPegInsertTransformerGRU.pth \
  --num_envs 32
```

## Evaluation

- Evaluate the scripted baseline:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode scripted \
  --num_envs 256 \
  --num_episodes 1000
```

- Evaluate PPO:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode ppo_transformer_gru \
  --checkpoint best_model/FactoryPegInsertTransformerGRU/nn/FactoryPegInsertTransformerGRU.pth \
  --num_envs 256 \
  --num_episodes 1000
```

- Evaluate SAC:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode sac \
  --checkpoint best_model/FactoryPegInsertSAC/nn/FactoryPegInsertSAC.pth \
  --num_envs 256 \
  --num_episodes 1000
```

- Evaluate TD3:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/skrl/eval_td3.py \
  --task Isaac-Factory-PegInsert-Local-Direct-v0 \
  --eval_mode skrl \
  --algorithm TD3 \
  --checkpoint logs/skrl/FactoryPegInsertTD3/<RunDir>/checkpoints/best_agent.pt \
  --num_envs 256 \
  --num_episodes 1000
```

- Evaluation scripts use `headless=True` by default.
- Evaluation scripts are intended for statistics, not visualization.
- Output includes episode count, success count, success rate, average successful steps, and average successful time.

## Classic Methods

- Classic method scripts are located at:

```text
scripts/reinforcement_learning/classic_method/
```

- Available scripts:

| File | Method |
| --- | --- |
| `scripted_baseline.py` | Hand-written two-stage policy: XY alignment followed by downward insertion |
| `rrt_baseline.py` | Simplified Cartesian RRT path tracking |
| `rrt2_baseline.py` | Cartesian RRT with simplified box collision avoidance |

- Run examples:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/classic_method/scripted_baseline.py \
  --num_envs 32 \
  --num_episodes 1000
```

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/classic_method/rrt2_baseline.py \
  --num_envs 32 \
  --num_episodes 1000
```

## Output Directories

- RL-Games training output:

```text
logs/rl_games/<ExperimentName>/<RunDir>/
├── nn/
├── params/
├── summaries/
└── videos/
```

- Contents:

| Path | Content |
| --- | --- |
| `nn/` | `.pth` checkpoints |
| `params/env.yaml` | Environment config used during training |
| `params/agent.yaml` | Agent config used during training |
| `summaries/` | TensorBoard event logs |

- Other directories:

| Path | Content |
| --- | --- |
| `best_model/` | Reference checkpoints |
| `logs/skrl/` | Historical TD3/skrl experiment logs |
| `outputs/` | Hydra outputs |
| `wandb/` | W&B local cache and artifacts |
| `runs/` | Historical TensorBoard fallback logs |

- Open TensorBoard:

```bash
tensorboard --logdir logs
```

## FAQ

### IsaacLab Modules Cannot Be Found

- Use the IsaacLab launcher:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --help
```

- If IsaacLab is not in the default location, set:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
```

### Checkpoint Cannot Be Found

- The default checkpoint search directory is:

```text
logs/rl_games/<ExperimentName>/<RunDir>/nn/
```

- If the model is in `best_model/` or another directory, pass it explicitly:

```bash
--checkpoint /absolute/or/relative/path/to/model.pth
```
