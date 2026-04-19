# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RL-Games checkpoint for the PegInsert task."""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source"


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
        "for example: <isaaclab_root>/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py"
    ) from exc

parser = argparse.ArgumentParser(description="Play a checkpoint with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric interface.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Registry key for the RL agent configuration."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint is provided, use the last saved checkpoint instead of the default best model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument(
    "--print_peg_metrics",
    action="store_true",
    default=False,
    help="Print peg pose diagnostics during play, including grasp_target_pos when the environment exposes it.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=10,
    help="Print peg metrics every N play steps when --print_peg_metrics is enabled.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
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

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.utils import get_checkpoint_path
from isaaclab_factory_tasks.utils.hydra import hydra_task_config


def _print_peg_metrics(prefix: str, env) -> None:
    """Print peg pose diagnostics for tasks that expose peg state."""
    if not hasattr(env, "held_pos") or not hasattr(env, "task_cfg"):
        return

    held_pos = env.held_pos[0].detach().cpu()
    peg_height = held_pos[2].item()
    lift_height = max(peg_height - env.task_cfg.table_height, 0.0)

    prev_held_pos = getattr(_print_peg_metrics, "_prev_held_pos", None)
    if prev_held_pos is None:
        held_delta = torch.zeros_like(held_pos)
    else:
        held_delta = held_pos - prev_held_pos
    _print_peg_metrics._prev_held_pos = held_pos.clone()

    def _format_vec3(vec: torch.Tensor) -> str:
        return f"({vec[0].item():.4f}, {vec[1].item():.4f}, {vec[2].item():.4f})"

    message = (
        f"{prefix} | held_pos={_format_vec3(held_pos)} | held_delta={_format_vec3(held_delta)}"
        f" | peg_height={peg_height:.4f} | lift_height={lift_height:.4f}"
    )

    if hasattr(env, "grasp_target_pos"):
        grasp_target_pos = env.grasp_target_pos[0].detach().cpu()
        prev_grasp_target_pos = getattr(_print_peg_metrics, "_prev_grasp_target_pos", None)
        if prev_grasp_target_pos is None:
            grasp_target_delta = torch.zeros_like(grasp_target_pos)
        else:
            grasp_target_delta = grasp_target_pos - prev_grasp_target_pos
        _print_peg_metrics._prev_grasp_target_pos = grasp_target_pos.clone()

        grasp_offset = grasp_target_pos - held_pos
        message += (
            f" | grasp_target_pos={_format_vec3(grasp_target_pos)}"
            f" | grasp_target_delta={_format_vec3(grasp_target_delta)}"
            f" | grasp_offset={_format_vec3(grasp_offset)}"
        )

    print(message, flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    """Load a trained checkpoint and play it."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = str((REPO_ROOT / "logs" / "rl_games" / agent_cfg["params"]["config"]["name"]).resolve())
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    timestep = 0
    play_step = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            play_step += 1

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        if args_cli.print_peg_metrics and (play_step % max(args_cli.print_interval, 1) == 0):
            _print_peg_metrics(f"play_step={play_step}", base_env)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
