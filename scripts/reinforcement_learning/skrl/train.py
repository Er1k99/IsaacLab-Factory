# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the refactored Factory tasks with skrl (SAC by default)."""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

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
        "for example: <isaaclab_root>/isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py"
    ) from exc

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-Direct-v0", help="Task name.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Registry key for the skrl agent config. Defaults to skrl_<algorithm>_cfg_entry_point.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="SAC",
    choices=["SAC", "PPO", "TD3"],
    help="Which skrl algorithm config to load.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume from.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help=(
        "Override training budget. Semantics differ by algorithm: "
        "for SAC/off-policy -> total vector-env timesteps (each timestep advances all "
        "num_envs envs once, producing num_envs transitions); "
        "for PPO/on-policy -> maximum training epochs. "
        "For fair sample-budget comparison across algorithms, compare timesteps * num_envs "
        "vs max_epochs * horizon_length * num_envs."
    ),
)
parser.add_argument("--wandb-project-name", type=str, default=None, help="W&B project name.")
parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity.")
parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
parser.add_argument(
    "--wandb-upload-model",
    type=lambda x: bool(strtobool(x)),
    default=True,
    nargs="?",
    const=True,
    help="Upload the latest checkpoint to Weights & Biases as a model artifact.",
)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Enable Weights & Biases tracking.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import skrl
import torch
from packaging import version
from skrl.utils.runner.torch import Runner
import importlib

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

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. Install 'pip install skrl>={SKRL_VERSION}'."
    )

algorithm = args_cli.algorithm.lower()
if args_cli.agent is None:
    agent_cfg_entry_point = f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent


def _maybe_patch_skrl_sac() -> None:
    """Monkey-patch skrl SAC critic loss: Huber + optional Q-target clipping.

    Both are gated by env vars (defaults: Huber on, Q clip ±50).
    Target clipping bounds critic outputs, which indirectly bounds policy gradient
    and breaks the policy-exploits-optimistic-Q feedback loop seen on Factory tasks.
    """

    if args_cli.algorithm.upper() != "SAC":
        return

    use_huber = os.environ.get("SKRL_SAC_USE_HUBER", "1") not in ("0", "false", "False", "no", "NO")

    beta_env = os.environ.get("SKRL_SAC_HUBER_BETA", "1.0")
    try:
        beta = float(beta_env)
    except ValueError:
        beta = 1.0

    q_clip_env = os.environ.get("SKRL_SAC_Q_CLIP", "50.0")
    try:
        q_clip = float(q_clip_env)
    except ValueError:
        q_clip = 50.0
    use_q_clip = q_clip > 0.0

    if not (use_huber or use_q_clip):
        return

    try:
        import torch
        import torch.nn.functional as torch_F

        sac_module = importlib.import_module("skrl.agents.torch.sac.sac")
        sac_F = getattr(sac_module, "F", None)
        if sac_F is None:
            print("[WARN] SAC patch skipped: skrl SAC module has no attribute 'F'")
            return

        def _patched_mse_loss(input, target, *args, **kwargs):
            reduction = kwargs.get("reduction", "mean")
            if use_q_clip:
                target = torch.clamp(target, -q_clip, q_clip)
            if use_huber:
                return torch_F.smooth_l1_loss(input, target, reduction=reduction, beta=beta)
            return torch_F.mse_loss(input, target, reduction=reduction)

        sac_F.mse_loss = _patched_mse_loss
        huber_str = f"on (beta={beta})" if use_huber else "off"
        clip_str = f"±{q_clip}" if use_q_clip else "off"
        print(
            f"[INFO] Patched skrl SAC critic loss: huber={huber_str}, q_target_clip={clip_str} "
            f"(module={getattr(sac_module, '__file__', 'unknown')})"
        )
    except Exception as exc:
        print(f"[WARN] SAC patch failed: {exc}")


def _prepare_wandb_dirs() -> Path:
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


def _get_experiment_group_name(agent_cfg: dict) -> str:
    configured_dir = agent_cfg["agent"]["experiment"].get("directory", "")
    return Path(str(configured_dir)).expanduser().name if configured_dir else args_cli.task


def _make_action_bound(agent, value: float) -> torch.Tensor:
    action_space = getattr(agent, "action_space", None)
    shape = getattr(action_space, "shape", None) or (getattr(agent, "num_actions", 1),)
    return torch.full(tuple(shape), value, device=agent.device)


def _patch_skrl_unbounded_action_bounds(agent) -> None:
    min_actions = _make_action_bound(agent, -1.0)
    max_actions = _make_action_bound(agent, 1.0)

    for owner in [agent, *getattr(agent, "models", {}).values()]:
        if owner is None:
            continue
        for min_attr, max_attr in (
            ("_min_actions", "_max_actions"),
            ("_d_min_actions", "_d_max_actions"),
            ("_g_min_actions", "_g_max_actions"),
            ("_mg_min_actions", "_mg_max_actions"),
        ):
            if hasattr(owner, min_attr) and (
                getattr(owner, min_attr, None) is None or getattr(owner, max_attr, None) is None
            ):
                setattr(owner, min_attr, min_actions)
                setattr(owner, max_attr, max_actions)


def _upload_wandb_checkpoints(wandb_run, run_dir: Path, config_name: str) -> None:
    import wandb

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        logger.warning("Skipping W&B model upload: checkpoint dir not found: %s", ckpt_dir)
        return

    checkpoint_paths = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoint_paths:
        logger.warning("Skipping W&B model upload: no checkpoints in %s", ckpt_dir)
        return

    latest_ckpt = checkpoint_paths[-1]
    best_ckpt = ckpt_dir / "best_agent.pt"
    if not best_ckpt.is_file():
        best_ckpt = latest_ckpt

    artifact = wandb.Artifact(
        name=f"{config_name}-{wandb_run.id}-checkpoints",
        type="model",
        metadata={
            "config_name": config_name,
            "run_dir": str(run_dir),
            "best_checkpoint": best_ckpt.name,
            "latest_checkpoint": latest_ckpt.name,
            "checkpoint_count": len(checkpoint_paths),
        },
    )
    artifact.add_file(str(best_ckpt), name=f"checkpoints/{best_ckpt.name}")
    if latest_ckpt != best_ckpt:
        artifact.add_file(str(latest_ckpt), name=f"checkpoints/{latest_ckpt.name}")

    params_dir = run_dir / "params"
    if params_dir.is_dir():
        for cfg_path in sorted(params_dir.glob("*.yaml")):
            artifact.add_file(str(cfg_path), name=f"params/{cfg_path.name}")

    wandb_run.log_artifact(artifact, aliases=["latest", "best"])
    wandb_run.summary["best_checkpoint"] = str(best_ckpt)
    wandb_run.summary["latest_checkpoint"] = str(latest_ckpt)
    wandb_run.summary["checkpoint_count"] = len(checkpoint_paths)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    """Train with the skrl runner."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    config_name = _get_experiment_group_name(agent_cfg)
    log_root_path = str((REPO_ROOT / "logs" / "skrl" / config_name).resolve())
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{timestamp}_{algorithm}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        experiment_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"

    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = experiment_name
    log_dir = os.path.join(log_root_path, experiment_name)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    wandb_run = None
    if args_cli.track:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights & Biases entity must be specified for tracking.")
        wandb_run_dir = _prepare_wandb_dirs()
        import wandb

        wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
        wandb_name = experiment_name if args_cli.wandb_name is None else args_cli.wandb_name
        wandb_run = wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=wandb_name,
            dir=str(wandb_run_dir),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb_run.config.update({"env_cfg": env_cfg.to_dict()})
            wandb_run.config.update({"agent_cfg": agent_cfg})

    _maybe_patch_skrl_sac()
    runner = Runner(env, agent_cfg)
    print(env.observation_space)
    print(env.state_space)
    policy = getattr(runner.agent, "policy", None)
    net_container = getattr(policy, "net_container", None)
    first_layer = net_container[0] if net_container is not None and len(net_container) > 0 else None
    print(getattr(first_layer, "in_features", None))
    _patch_skrl_unbounded_action_bounds(runner.agent)

    if resume_path:
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    start_time = time.time()
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    if wandb_run is not None:
        if args_cli.wandb_upload_model:
            _upload_wandb_checkpoints(wandb_run, Path(log_dir), config_name)
        wandb_run.finish()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
