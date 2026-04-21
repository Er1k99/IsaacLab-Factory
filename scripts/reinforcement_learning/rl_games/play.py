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

import yaml

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

parser = argparse.ArgumentParser(description="Play a PPO or SAC checkpoint with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric interface.")
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
    choices=["PPO", "PPO_GRU", "PPO_LSTM", "PPO_MLP", "SAC"],
    help="RL-Games algorithm/network preset to play. 'PPO' is kept as a backward-compatible alias for 'PPO_GRU'.",
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
    help="Print peg height and lift height during play when the environment exposes them.",
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
from isaaclab_factory_tasks.utils.rl_games_sac import (
    FactoryRlGamesVecEnvWrapper,
    register_factory_rl_games_sac,
    upgrade_factory_sac_agent_cfg,
)


PPO_GRU = "PPO_GRU"
PPO_LSTM = "PPO_LSTM"
PPO_MLP = "PPO_MLP"
SAC = "SAC"
PPO_VARIANTS = {PPO_GRU, PPO_LSTM, PPO_MLP}


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
    if normalized == "sac":
        return SAC
    raise ValueError(
        f"Unsupported RL-Games algorithm: '{algorithm_name}'. Expected one of: "
        "PPO_GRU, PPO_LSTM, PPO_MLP, SAC."
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
        SAC: "rl_games_sac_cfg_entry_point",
    }
    return entry_points[algorithm_name]


def _enable_player_vecenv(agent_cfg: dict) -> None:
    player_cfg = agent_cfg.setdefault("params", {}).setdefault("config", {}).setdefault("player", {})
    player_cfg["use_vecenv"] = True


if args_cli.agent is None:
    algorithm = _normalize_rl_games_algorithm(args_cli.algorithm)
    agent_cfg_entry_point = _resolve_default_agent_cfg_entry_point(algorithm)
else:
    agent_cfg_entry_point = args_cli.agent


def _get_agent_cfg_algorithm(agent_cfg: dict) -> str:
    algo_name = agent_cfg.get("params", {}).get("algo", {}).get("name")
    normalized_algo_name = str(algo_name).strip().lower() if algo_name is not None else "a2c_continuous"
    if normalized_algo_name == "sac":
        return SAC
    if normalized_algo_name not in {"ppo", "a2c_continuous", "a2c"}:
        raise ValueError(f"Unsupported RL-Games algo in agent cfg: '{algo_name}'.")

    rnn_cfg = agent_cfg.get("params", {}).get("network", {}).get("rnn")
    if not isinstance(rnn_cfg, dict):
        return PPO_MLP

    recurrent_type = str(rnn_cfg.get("name", "")).strip().lower()
    if recurrent_type == "gru":
        return PPO_GRU
    if recurrent_type == "lstm":
        return PPO_LSTM
    return PPO_MLP


def _resolve_run_dir_pattern(configured_name: str | None) -> str:
    if configured_name is None:
        return ".*"
    configured_name = str(configured_name).strip()
    return configured_name if configured_name else ".*"


def _load_saved_agent_cfg_from_checkpoint(checkpoint_path: str | Path) -> dict | None:
    checkpoint_file = Path(checkpoint_path).resolve()
    params_file = checkpoint_file.parents[1] / "params" / "agent.yaml"
    if not params_file.is_file():
        return None

    with open(params_file, encoding="utf-8") as file_handle:
        return yaml.full_load(file_handle)


def _print_peg_metrics(prefix: str, env) -> None:
    """Print peg z-height diagnostics for tasks that expose peg state."""
    if not hasattr(env, "held_pos") or not hasattr(env, "task_cfg"):
        return

    peg_height = env.held_pos[:, 2].detach().cpu()
    lift_height = torch.clamp(env.held_pos[:, 2] - env.task_cfg.table_height, min=0.0).detach().cpu()
    print(
        f"{prefix} | peg_height={peg_height[0].item():.4f} | lift_height={lift_height[0].item():.4f}",
        flush=True,
    )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    """Load a trained checkpoint and play it."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.checkpoint is None:
        effective_algorithm = _get_agent_cfg_algorithm(agent_cfg)
        log_root_path = str((REPO_ROOT / "logs" / "rl_games" / agent_cfg["params"]["config"]["name"]).resolve())
        print(f"[INFO] Loading RL-Games {effective_algorithm} experiment from directory: {log_root_path}")
        run_dir = _resolve_run_dir_pattern(agent_cfg["params"]["config"].get("full_experiment_name"))
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        saved_agent_cfg = _load_saved_agent_cfg_from_checkpoint(resume_path)
        if saved_agent_cfg is not None:
            agent_cfg = saved_agent_cfg

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    effective_algorithm = _get_agent_cfg_algorithm(agent_cfg)
    effective_algorithm_family = _get_algorithm_family(effective_algorithm)
    if effective_algorithm_family == SAC:
        upgrade_factory_sac_agent_cfg(agent_cfg)

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

    wrapper_cls = FactoryRlGamesVecEnvWrapper if effective_algorithm_family == SAC else RlGamesVecEnvWrapper
    env = wrapper_cls(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading RL-Games {effective_algorithm} checkpoint from: {resume_path}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    _enable_player_vecenv(agent_cfg)
    runner = Runner()
    if effective_algorithm_family == SAC:
        register_factory_rl_games_sac(runner)
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
