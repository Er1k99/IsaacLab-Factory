# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the PegInsert task with RL-Games."""

import argparse
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

try:
    from distutils.util import strtobool
except ImportError:
    from setuptools._distutils.util import strtobool

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source"
WANDB_ROOT = REPO_ROOT / "wandb"


def _bootstrap_pythonpath() -> None:
    candidate_roots = []
    for env_name in ("ISAACLAB_ROOT", "ISAACLAB_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate_roots.append(Path(env_value).expanduser())

    candidate_roots.extend(
        [
            Path.home() / "Isaaclab2.3.2",
            Path.home() / "IsaacLab",
        ]
    )

    search_paths = [SOURCE_ROOT]
    for root in candidate_roots:
        source_dir = root / "source"
        for pkg_name in ("isaaclab", "isaaclab_rl", "isaaclab_assets"):
            pkg_root = source_dir / pkg_name
            if pkg_root.exists():
                search_paths.append(pkg_root)

    for path in reversed(search_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_pythonpath()

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import IsaacLab runtime modules. Run this script with a working IsaacLab/Isaac Sim environment, "
        "for example: <isaaclab_root>/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py"
    ) from exc

parser = argparse.ArgumentParser(description="Train an RL-Games PPO or SAC agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Registry key for the RL agent configuration. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO_GRU",
    choices=["PPO", "PPO_GRU", "PPO_LSTM", "PPO_MLP", "PPO_TRANSFORMER", "PPO_TRANSFORMER_GRU", "SAC"],
    help="RL-Games algorithm/network preset to train. 'PPO' is kept as a backward-compatible alias for 'PPO_GRU'.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="Policy initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum RL training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="Weights and Biases project name.")
parser.add_argument("--wandb-entity", type=str, default=None, help="Weights and Biases entity.")
parser.add_argument("--wandb-name", type=str, default=None, help="Weights and Biases run name.")
parser.add_argument(
    "--wandb-upload-model",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="Upload the best and latest checkpoints to Weights and Biases as a model artifact.",
)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Enable Weights and Biases tracking.",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None, help="Reserved for Ray integration.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.utils.hydra import hydra_task_config
from isaaclab_factory_tasks.utils.rl_games_sac import (
    FactoryRlGamesVecEnvWrapper,
    register_factory_rl_games_sac,
    upgrade_factory_sac_agent_cfg,
)
from isaaclab_factory_tasks.utils.rl_games_transformer import (
    FactoryTemporalRlGamesVecEnvWrapper,
    get_factory_transformer_history_length,
    is_factory_transformer_agent_cfg,
    register_factory_rl_games_transformer,
)

logger = logging.getLogger(__name__)


PPO_GRU = "PPO_GRU"
PPO_LSTM = "PPO_LSTM"
PPO_MLP = "PPO_MLP"
PPO_TRANSFORMER = "PPO_TRANSFORMER"
PPO_TRANSFORMER_GRU = "PPO_TRANSFORMER_GRU"
SAC = "SAC"
PPO_VARIANTS = {PPO_GRU, PPO_LSTM, PPO_MLP, PPO_TRANSFORMER, PPO_TRANSFORMER_GRU}


def _normalize_rl_games_algorithm(algorithm_name: str | None) -> str | None:
    if algorithm_name is None:
        return None

    normalized = str(algorithm_name).strip().lower().replace("-", "_")
    if normalized in {"ppo", "ppo_gru"}:
        return PPO_GRU
    if normalized == "ppo_lstm":
        return PPO_LSTM
    if normalized == "ppo_mlp":
        return PPO_MLP
    if normalized == "ppo_transformer":
        return PPO_TRANSFORMER
    if normalized == "ppo_transformer_gru":
        return PPO_TRANSFORMER_GRU
    if normalized == "sac":
        return SAC
    raise ValueError(
        f"Unsupported RL-Games algorithm: '{algorithm_name}'. Expected one of: "
        "PPO_GRU, PPO_LSTM, PPO_MLP, PPO_TRANSFORMER, PPO_TRANSFORMER_GRU, SAC."
    )


def _get_algorithm_family(algorithm_name: str) -> str:
    algorithm_name = _normalize_rl_games_algorithm(algorithm_name)
    if algorithm_name == SAC:
        return SAC
    if algorithm_name in PPO_VARIANTS:
        return "PPO"
    raise ValueError(f"Unsupported RL-Games algorithm family for '{algorithm_name}'.")


def _resolve_default_agent_cfg_entry_point(algorithm_name: str) -> str:
    algorithm_name = _normalize_rl_games_algorithm(algorithm_name)
    entry_points = {
        PPO_GRU: "rl_games_ppo_gru_cfg_entry_point",
        PPO_LSTM: "rl_games_ppo_lstm_cfg_entry_point",
        PPO_MLP: "rl_games_ppo_mlp_cfg_entry_point",
        PPO_TRANSFORMER: "rl_games_ppo_transformer_cfg_entry_point",
        PPO_TRANSFORMER_GRU: "rl_games_ppo_transformer_gru_cfg_entry_point",
        SAC: "rl_games_sac_cfg_entry_point",
    }
    return entry_points[algorithm_name]


if args_cli.agent is None:
    algorithm = _normalize_rl_games_algorithm(args_cli.algorithm)
    agent_cfg_entry_point = _resolve_default_agent_cfg_entry_point(algorithm)
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = None


def _get_agent_cfg_recurrent_type(agent_cfg: dict) -> str | None:
    rnn_cfg = agent_cfg.get("params", {}).get("network", {}).get("rnn")
    if not isinstance(rnn_cfg, dict):
        return None
    recurrent_type = str(rnn_cfg.get("name", "")).strip().lower()
    if recurrent_type not in {"gru", "lstm"}:
        return None
    return recurrent_type


def _get_agent_cfg_algorithm(agent_cfg: dict) -> str:
    algo_name = agent_cfg.get("params", {}).get("algo", {}).get("name")
    normalized_algo_name = str(algo_name).strip().lower() if algo_name is not None else "a2c_continuous"
    if normalized_algo_name == "sac":
        return SAC
    if normalized_algo_name not in {"ppo", "a2c_continuous", "a2c"}:
        raise ValueError(f"Unsupported RL-Games algo in agent cfg: '{algo_name}'.")
    if is_factory_transformer_agent_cfg(agent_cfg):
        recurrent_type = _get_agent_cfg_recurrent_type(agent_cfg)
        if recurrent_type == "gru":
            return PPO_TRANSFORMER_GRU
        if recurrent_type is not None:
            raise ValueError(
                f"Unsupported recurrent type '{recurrent_type}' for the local transformer PPO network."
            )
        return PPO_TRANSFORMER

    recurrent_type = _get_agent_cfg_recurrent_type(agent_cfg)
    if recurrent_type == "gru":
        return PPO_GRU
    if recurrent_type == "lstm":
        return PPO_LSTM
    return PPO_MLP


def _load_saved_agent_cfg_from_checkpoint(checkpoint_path: str) -> dict | None:
    checkpoint_file = Path(checkpoint_path).resolve()
    params_file = checkpoint_file.parents[1] / "params" / "agent.yaml"
    if not params_file.is_file():
        return None

    with open(params_file, encoding="utf-8") as file_handle:
        return yaml.full_load(file_handle)


def _default_wandb_project_name(task_name: str, fallback_name: str | None = None) -> str:
    return "Factory"


def _default_wandb_run_name(base_name: str, log_dir: str) -> str:
    return f"{base_name}-{log_dir}"


def _resolve_log_dir(config_name: str, configured_name: str | None) -> str:
    """Resolve a timestamped experiment directory to avoid overwriting previous runs."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if configured_name is None:
        return timestamp

    configured_name = str(configured_name).strip()
    if configured_name == "" or configured_name.lower() in {"test", "default", "auto"}:
        return timestamp

    return f"{configured_name}_{timestamp}"


def _resolve_run_dir_pattern(configured_name: str | None) -> str:
    if configured_name is None:
        return ".*"
    configured_name = str(configured_name).strip()
    return configured_name if configured_name else ".*"


def _upload_wandb_checkpoints(wandb_run, run_dir: Path, config_name: str) -> None:
    """Upload the best and latest RL-Games checkpoints to Weights & Biases."""
    import wandb

    nn_dir = run_dir / "nn"
    if not nn_dir.is_dir():
        logger.warning("Skipping Weights & Biases model upload because checkpoint directory does not exist: %s", nn_dir)
        return

    checkpoint_paths = sorted(nn_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime)
    if len(checkpoint_paths) == 0:
        logger.warning("Skipping Weights & Biases model upload because no checkpoints were found in: %s", nn_dir)
        return

    best_checkpoint = nn_dir / f"{config_name}.pth"
    if not best_checkpoint.is_file():
        best_checkpoint = checkpoint_paths[-1]

    latest_checkpoint = checkpoint_paths[-1]

    artifact = wandb.Artifact(
        name=f"{config_name}-{wandb_run.id}-checkpoints",
        type="model",
        metadata={
            "config_name": config_name,
            "run_dir": str(run_dir),
            "best_checkpoint": best_checkpoint.name,
            "latest_checkpoint": latest_checkpoint.name,
            "checkpoint_count": len(checkpoint_paths),
        },
    )
    artifact.add_file(str(best_checkpoint), name=f"nn/{best_checkpoint.name}")
    if latest_checkpoint != best_checkpoint:
        artifact.add_file(str(latest_checkpoint), name=f"nn/{latest_checkpoint.name}")

    params_dir = run_dir / "params"
    if params_dir.is_dir():
        for cfg_path in sorted(params_dir.glob("*.yaml")):
            artifact.add_file(str(cfg_path), name=f"params/{cfg_path.name}")

    wandb_run.log_artifact(artifact, aliases=["latest", "best"])
    wandb_run.summary["best_checkpoint"] = str(best_checkpoint)
    wandb_run.summary["latest_checkpoint"] = str(latest_checkpoint)
    wandb_run.summary["checkpoint_count"] = len(checkpoint_paths)


def _prepare_wandb_dirs() -> Path:
    """Pin all local Weights & Biases files to the repository."""
    run_dir = WANDB_ROOT / "runs"
    cache_dir = WANDB_ROOT / ".cache"
    artifact_dir = WANDB_ROOT / "artifacts"
    data_dir = WANDB_ROOT / "data"
    config_dir = WANDB_ROOT / "config"

    for path in (run_dir, cache_dir, artifact_dir, data_dir, config_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(run_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    os.environ["WANDB_CONFIG_DIR"] = str(config_dir)

    return run_dir


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    """Train an RL-Games agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on CPU.")

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    saved_agent_cfg = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        saved_agent_cfg = _load_saved_agent_cfg_from_checkpoint(resume_path)
        if saved_agent_cfg is not None:
            agent_cfg = saved_agent_cfg

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    effective_algorithm = _get_agent_cfg_algorithm(agent_cfg)
    effective_algorithm_family = _get_algorithm_family(effective_algorithm)
    if effective_algorithm_family == SAC:
        upgrade_factory_sac_agent_cfg(agent_cfg)
    print(f"[INFO] Using RL-Games algorithm: {effective_algorithm}")

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )

    if args_cli.sigma is not None and effective_algorithm_family != "PPO":
        raise ValueError("'--sigma' is only supported for RL-Games PPO.")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env_cfg.seed = agent_cfg["params"]["seed"]

    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = str((REPO_ROOT / "logs" / "rl_games" / config_name).resolve())
    log_dir = _resolve_log_dir(config_name, agent_cfg["params"]["config"].get("full_experiment_name"))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    wandb_project = (
        _default_wandb_project_name(args_cli.task, config_name)
        if args_cli.wandb_project_name is None
        else args_cli.wandb_project_name
    )
    experiment_name = (
        _default_wandb_run_name(config_name, log_dir) if args_cli.wandb_name is None else args_cli.wandb_name
    )

    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = os.path.join(log_root_path, log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    if effective_algorithm_family == SAC:
        env = FactoryRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    elif effective_algorithm in {PPO_TRANSFORMER, PPO_TRANSFORMER_GRU}:
        env = FactoryTemporalRlGamesVecEnvWrapper(
            env,
            rl_device,
            clip_obs,
            clip_actions,
            obs_groups,
            concate_obs_groups,
            history_length=get_factory_transformer_history_length(agent_cfg),
        )
    else:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    agent_cfg["params"]["config"]["max_env_steps"] = int(env.unwrapped.max_episode_length)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(IsaacAlgoObserver())

    if effective_algorithm in {PPO_TRANSFORMER, PPO_TRANSFORMER_GRU}:
        register_factory_rl_games_transformer()
    if effective_algorithm_family == SAC:
        register_factory_rl_games_sac(runner)

    runner.load(agent_cfg)
    runner.reset()

    global_rank = int(os.getenv("RANK", "0"))
    wandb_run = None
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        wandb_run_dir = _prepare_wandb_dirs()
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            dir=str(wandb_run_dir),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb_run.config.update({"env_cfg": env_cfg.to_dict()})
            wandb_run.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    if wandb_run is not None:
        if args_cli.wandb_upload_model:
            _upload_wandb_checkpoints(wandb_run, Path(env_cfg.log_dir), config_name)
        wandb_run.finish()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
