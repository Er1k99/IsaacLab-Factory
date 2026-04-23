# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play (rollout) a trained skrl policy for the Factory tasks."""

import argparse
import os
import sys
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
        "Could not import IsaacLab runtime modules. Use <isaaclab_root>/isaaclab.sh -p to run this script."
    ) from exc

parser = argparse.ArgumentParser(description="Play a trained skrl policy.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-Direct-v0", help="Task name.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Registry key for skrl agent config. Defaults to skrl_<algorithm>_cfg_entry_point.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="SAC",
    choices=["SAC", "PPO", "TD3"],
    help="skrl algorithm family to load config for.",
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to play.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt). Required.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use the latest checkpoint from logs dir.")
parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video in steps.")
parser.add_argument(
    "--video_dir", type=str, default=None,
    help="Directory to save videos. Defaults to <checkpoint_dir>/../videos/play.",
)
parser.add_argument(
    "--pause_before_rollout", action="store_true",
    help="Pause after scene init and wait for ENTER before running the policy. Use with GUI to set up Movie Capture.",
)
parser.add_argument(
    "--viewer_eye", type=float, nargs=3, default=[1.2, -0.6, 0.5],
    metavar=("X", "Y", "Z"),
    help="Viewport camera position (in env-local meters). Default tuned for Factory peg tasks.",
)
parser.add_argument(
    "--viewer_lookat", type=float, nargs=3, default=[0.55, 0.0, 0.1],
    metavar=("X", "Y", "Z"),
    help="Viewport camera target (in env-local meters). Default tuned for Factory peg tasks.",
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
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.utils.hydra import hydra_task_config
from isaaclab_factory_tasks.utils.parse_cfg import get_checkpoint_path

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = args_cli.agent or f"skrl_{algorithm}_cfg_entry_point"


def _get_experiment_group_name(agent_cfg: dict) -> str:
    configured_dir = agent_cfg["agent"]["experiment"].get("directory", "")
    return Path(str(configured_dir)).expanduser().name if configured_dir else args_cli.task


def _get_skrl_states(env, obs: torch.Tensor) -> torch.Tensor:
    try:
        return env.state()
    except (AttributeError, NotImplementedError):
        return obs


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


def _set_skrl_agent_eval_mode(agent) -> None:
    set_legacy_training_mode = getattr(agent, "enable_models_training_mode", None)
    if callable(set_legacy_training_mode):
        set_legacy_training_mode(False)
    elif callable(getattr(agent, "set_mode", None)):
        agent.set_mode("eval")
    else:
        for model in getattr(agent, "models", {}).values():
            if model is None:
                continue
            if callable(getattr(model, "set_mode", None)):
                model.set_mode("eval")
            elif callable(getattr(model, "eval", None)):
                model.eval()

    if callable(getattr(agent, "set_running_mode", None)):
        agent.set_running_mode("eval")
    if hasattr(agent, "_exploration_noise"):
        agent._exploration_noise = None


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # Override viewport camera for close-up recording (env-local frame, env 0).
    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
        env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    elif args_cli.use_last_checkpoint:
        config_name = _get_experiment_group_name(agent_cfg)
        log_root = str((REPO_ROOT / "logs" / "skrl" / config_name).resolve())
        resume_path = get_checkpoint_path(
            log_root, run_dir=".*", checkpoint=".*\\.pt", other_dirs=["checkpoints"], sort_alpha=False
        )
    else:
        raise ValueError("Pass --checkpoint PATH or --use_last_checkpoint to play.")

    print(f"[INFO] Loading checkpoint from: {resume_path}")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        if args_cli.video_dir:
            video_folder = os.path.abspath(args_cli.video_dir)
        else:
            video_folder = os.path.join(os.path.dirname(os.path.dirname(resume_path)), "videos", "play")
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 1,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_folder} (length={args_cli.video_length})")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)
    runner.agent.load(resume_path)
    _patch_skrl_unbounded_action_bounds(runner.agent)
    _set_skrl_agent_eval_mode(runner.agent)

    obs, _ = env.reset()
    if args_cli.pause_before_rollout:
        import select
        print("\n" + "=" * 60)
        print("[PAUSE] Scene ready. Adjust the Isaac Sim viewport camera now.")
        print("[PAUSE] Open Window > Movie Capture, configure output, and arm recording.")
        print("[PAUSE] Press ENTER in THIS terminal to start policy rollout.")
        print("=" * 60, flush=True)
        while simulation_app.is_running():
            simulation_app.update()  # keep GUI event loop + rendering alive
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
        print("[RUN] Starting policy rollout now.", flush=True)
    step = 0
    # When recording, stop shortly after the video finishes so the file flushes.
    max_steps = args_cli.video_length + 20 if args_cli.video else None
    eval_timestep = 10**9  # bypass skrl's random_timesteps warmup gate
    while simulation_app.is_running():
        with torch.inference_mode():
            states = _get_skrl_states(env, obs)
            actions = runner.agent.act(obs, states, timestep=eval_timestep, timesteps=eval_timestep)[0]
            actions = torch.clamp(actions, -1.0, 1.0)
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated.any() or truncated.any():
                pass  # vec env auto-resets per-env
        step += 1
        if max_steps is not None and step >= max_steps:
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
