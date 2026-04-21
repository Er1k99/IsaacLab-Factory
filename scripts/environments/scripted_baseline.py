"""Scripted baseline plus RL-Games PPO and SAC evaluation for the PegInsert task."""

import argparse
import os
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        "for example: <isaaclab_root>/isaaclab.sh -p scripts/environments/scripted_baseline.py"
    ) from exc

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument(
    "--eval_mode",
    type=str,
    default="scripted",
    choices=["scripted", "ppo_gru", "ppo_lstm", "sac"],
    help="Evaluation mode: scripted IK baseline, RL-Games PPO checkpoint, or an RL-Games SAC checkpoint.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Factory-PegInsert-Local-Direct-v0",
    help="PegInsert task id to evaluate.",
)
parser.add_argument(
    "--experiment",
    type=str,
    default=None,
    help="Optional experiment directory name under logs/rl_games.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Optional PPO/SAC checkpoint path.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint path is provided, use the latest saved checkpoint instead of the default best model.",
)
parser.add_argument(
    "--stochastic_policy",
    action="store_true",
    help="Sample policy actions instead of using deterministic actions in PPO/SAC evaluation.",
)
parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.direct.peg_insert.env_cfg import PegInsertEnvCfg


def _get_step_dt(env) -> float:
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    return float(env.cfg.episode_length_s / env.max_episode_length)


def _normalize_recurrent_type(rnn_name: str | None) -> str | None:
    if rnn_name is None:
        return None
    normalized = str(rnn_name).strip().lower()
    if normalized not in {"gru", "lstm"}:
        raise ValueError(f"Unsupported recurrent type: '{rnn_name}'. Expected one of: gru, lstm.")
    return normalized


def _get_agent_cfg_recurrent_type(agent_cfg: dict) -> str | None:
    rnn_cfg = agent_cfg.get("params", {}).get("network", {}).get("rnn")
    if not isinstance(rnn_cfg, dict):
        return None
    return _normalize_recurrent_type(rnn_cfg.get("name"))


def _set_agent_cfg_recurrent_type(agent_cfg: dict, recurrent_type: str) -> None:
    recurrent_type = _normalize_recurrent_type(recurrent_type)
    rnn_cfgs = [
        agent_cfg.get("params", {}).get("network", {}).get("rnn"),
        agent_cfg.get("params", {})
        .get("config", {})
        .get("central_value_config", {})
        .get("network", {})
        .get("rnn"),
    ]

    updated = False
    for rnn_cfg in rnn_cfgs:
        if isinstance(rnn_cfg, dict):
            rnn_cfg["name"] = recurrent_type
            updated = True

    if not updated:
        raise ValueError("The loaded RL-Games config does not define an RNN block to override.")


def _strip_recurrent_suffix(config_name: str) -> str:
    lowered = config_name.lower()
    for suffix in ("gru", "lstm"):
        if lowered.endswith(suffix):
            return config_name[: -len(suffix)]
    return config_name


def _resolve_config_name_for_recurrent_type(
    config_name: str,
    current_recurrent_type: str | None,
    requested_recurrent_type: str | None,
) -> str:
    requested_recurrent_type = _normalize_recurrent_type(requested_recurrent_type)
    current_recurrent_type = _normalize_recurrent_type(current_recurrent_type)
    if requested_recurrent_type is None or requested_recurrent_type == current_recurrent_type:
        return config_name
    return f"{_strip_recurrent_suffix(config_name)}{requested_recurrent_type.upper()}"


def _list_experiments(log_group: str) -> list[str]:
    logs_root = REPO_ROOT / "logs" / log_group
    if not logs_root.is_dir():
        return []
    return sorted(path.name for path in logs_root.iterdir() if path.is_dir())


def _resolve_log_root(log_group: str, preferred_name: str) -> Path:
    logs_root = REPO_ROOT / "logs" / log_group
    if not logs_root.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: '{logs_root}'.")

    log_root = logs_root / preferred_name
    if log_root.is_dir():
        return log_root

    available = _list_experiments(log_group)
    available_str = ", ".join(available) if available else "<none>"
    raise FileNotFoundError(
        "Could not find the expected experiment directory.\n"
        f"Expected: '{log_root}'\n"
        f"Available experiments: {available_str}\n"
        "If you want to evaluate an older run, pass either "
        "`--experiment <dir_name>` or `--checkpoint <path/to/model>`."
    )


def _get_run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_file = Path(checkpoint_path).resolve()
    if checkpoint_file.parent.name == "nn":
        return checkpoint_file.parent.parent
    return checkpoint_file.parent


def _load_saved_agent_cfg_from_checkpoint(checkpoint_path: str | Path) -> dict | None:
    params_file = _get_run_dir_from_checkpoint(checkpoint_path) / "params" / "agent.yaml"
    if not params_file.is_file():
        return None

    with open(params_file, encoding="utf-8") as file_handle:
        return yaml.full_load(file_handle)


def _resolve_run_dir_pattern(configured_name: str | None) -> str:
    if configured_name is None:
        return ".*"
    configured_name = str(configured_name).strip()
    return configured_name if configured_name else ".*"


def _enable_player_vecenv(agent_cfg: dict) -> None:
    player_cfg = agent_cfg.setdefault("params", {}).setdefault("config", {}).setdefault("player", {})
    player_cfg["use_vecenv"] = True


def _print_progress(step: int, total_episodes: int, total_successes: int, label: str) -> None:
    success_rate = total_successes / max(total_episodes, 1) * 100.0
    print(f"  {label} step {step} | Episodes: {total_episodes} | Success: {success_rate:.1f}%")


def _print_summary(
    label: str,
    total_episodes: int,
    total_successes: int,
    total_steps_list: list[int],
    total_time_list: list[float],
) -> None:
    success_rate = total_successes / max(total_episodes, 1) * 100.0
    avg_steps = sum(total_steps_list) / len(total_steps_list) if total_steps_list else 0.0
    avg_time = sum(total_time_list) / len(total_time_list) if total_time_list else 0.0

    print("\n" + "=" * 50)
    print(f"[{label} Results]")
    print(f"  Episodes:              {total_episodes}")
    print(f"  Successes:             {total_successes}")
    print(f"  Success rate:          {success_rate:.1f}%")
    print(f"  Avg steps to success:  {avg_steps:.1f}")
    print(f"  Avg time  to success:  {avg_time:.2f} s")
    print("=" * 50)


def run_scripted_baseline() -> None:
    env_cfg = PegInsertEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)

    u = env.unwrapped
    device = u.device
    num_envs = u.num_envs
    step_dt = _get_step_dt(u)

    total_episodes = 0
    total_successes = 0
    total_steps_list: list[int] = []
    total_time_list: list[float] = []

    env.reset()

    phase = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    print(f"[Scripted] Running {args_cli.num_episodes} episodes...")
    print(f"[Scripted] Step dt = {step_dt:.4f}s, max_episode_length = {u.max_episode_length}")

    max_steps = int(u.max_episode_length) * 5

    for step in range(max_steps):
        actions = torch.zeros((num_envs, 6), device=device)
        thresh = u.pos_threshold.clamp(min=1e-6)

        phase0 = (phase == 0).nonzero(as_tuple=False).squeeze(-1)
        if phase0.numel() > 0:
            target = u.fixed_pos_obs_frame[phase0].clone()
            target[:, 2] = u.fingertip_midpoint_pos[phase0, 2]
            delta = target - u.fingertip_midpoint_pos[phase0]
            actions[phase0, 0:3] = (delta / thresh[phase0]).clamp(-1.0, 1.0)
            xy_dist = torch.norm(delta[:, 0:2], dim=-1)
            phase[phase0[xy_dist < 0.003]] = 1

        phase1 = (phase == 1).nonzero(as_tuple=False).squeeze(-1)
        if phase1.numel() > 0:
            target = u.fixed_pos_obs_frame[phase1].clone()
            target[:, 2] -= 0.03
            delta = target - u.fingertip_midpoint_pos[phase1]
            actions[phase1, 0:3] = (delta / thresh[phase1]).clamp(-0.5, 0.5)

        episode_steps += 1
        _, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        curr_successes = u._get_curr_successes(u.task_cfg.success_threshold)
        first_success = curr_successes & ~episode_success
        success_step[first_success] = episode_steps[first_success]
        episode_success |= curr_successes

        done_envs = done.nonzero(as_tuple=False).squeeze(-1)
        if done_envs.numel() > 0:
            for env_id in done_envs.tolist():
                total_episodes += 1
                if episode_success[env_id]:
                    total_successes += 1
                    steps = int(success_step[env_id].item())
                    total_steps_list.append(steps)
                    total_time_list.append(steps * step_dt)

            phase[done_envs] = 0
            episode_steps[done_envs] = 0
            episode_success[done_envs] = False
            success_step[done_envs] = 0

            if total_episodes >= args_cli.num_episodes:
                break

        if step % 100 == 0:
            _print_progress(step, total_episodes, total_successes, "scripted")

    _print_summary("Scripted Baseline", total_episodes, total_successes, total_steps_list, total_time_list)
    env.close()


def run_ppo_eval(requested_recurrent_type: str) -> None:
    try:
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RL-Games is not installed. Run this script inside a working IsaacLab RL environment."
        ) from exc

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    from isaaclab_factory_tasks.utils import get_checkpoint_path, load_cfg_from_registry

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    registry_agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    registry_recurrent_type = _get_agent_cfg_recurrent_type(registry_agent_cfg)
    requested_recurrent_type = _normalize_recurrent_type(requested_recurrent_type)
    if requested_recurrent_type is not None and requested_recurrent_type != registry_recurrent_type:
        _set_agent_cfg_recurrent_type(registry_agent_cfg, requested_recurrent_type)

    env_cfg = PegInsertEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.checkpoint is None:
        default_experiment = _resolve_config_name_for_recurrent_type(
            registry_agent_cfg["params"]["config"]["name"], registry_recurrent_type, requested_recurrent_type
        )
        preferred_experiment = args_cli.experiment or default_experiment
        log_root_path = _resolve_log_root("rl_games", preferred_experiment)
        print(f"[PPO-{requested_recurrent_type.upper()}] Loading experiment from directory: {log_root_path}")

        run_dir = _resolve_run_dir_pattern(registry_agent_cfg["params"]["config"].get("full_experiment_name"))
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{preferred_experiment}.pth"
        resume_path = get_checkpoint_path(str(log_root_path), run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    agent_cfg = _load_saved_agent_cfg_from_checkpoint(resume_path) or registry_agent_cfg
    effective_recurrent_type = _get_agent_cfg_recurrent_type(agent_cfg)
    if effective_recurrent_type is None and requested_recurrent_type is not None:
        _set_agent_cfg_recurrent_type(agent_cfg, requested_recurrent_type)
        effective_recurrent_type = requested_recurrent_type

    rl_device = args_cli.device if args_cli.device is not None else agent_cfg["params"]["config"]["device"]
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    env_cfg.log_dir = str(_get_run_dir_from_checkpoint(resume_path))
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    print(f"[PPO-{effective_recurrent_type.upper()}] Loading checkpoint: {resume_path}")

    _enable_player_vecenv(agent_cfg)
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    step_dt = _get_step_dt(base_env)
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    total_episodes = 0
    total_successes = 0
    total_steps_list: list[int] = []
    total_time_list: list[float] = []

    device = base_env.device
    episode_steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)
    episode_success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)
    success_step = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)

    progress_label = f"ppo_{effective_recurrent_type}"
    summary_label = f"PPO {effective_recurrent_type.upper()} Evaluation"
    print(f"[PPO-{effective_recurrent_type.upper()}] Running {args_cli.num_episodes} episodes...")
    print(f"[PPO-{effective_recurrent_type.upper()}] Step dt = {step_dt:.4f}s, max_episode_length = {base_env.max_episode_length}")

    play_step = 0
    while total_episodes < args_cli.num_episodes and simulation_app.is_running():
        play_step += 1
        episode_steps += 1

        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=not args_cli.stochastic_policy)
            obs, _, dones, _ = env.step(actions)

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        curr_successes = base_env._get_curr_successes(base_env.task_cfg.success_threshold)
        first_success = curr_successes & ~episode_success
        success_step[first_success] = episode_steps[first_success]
        episode_success |= curr_successes

        done_envs = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_envs.numel() > 0:
            for env_id in done_envs.tolist():
                total_episodes += 1
                if episode_success[env_id]:
                    total_successes += 1
                    steps = int(success_step[env_id].item())
                    total_steps_list.append(steps)
                    total_time_list.append(steps * step_dt)
                if total_episodes >= args_cli.num_episodes:
                    break

            episode_steps[done_envs] = 0
            episode_success[done_envs] = False
            success_step[done_envs] = 0

        if play_step % 100 == 0:
            _print_progress(play_step, total_episodes, total_successes, progress_label)

    _print_summary(summary_label, total_episodes, total_successes, total_steps_list, total_time_list)
    env.close()


def run_sac_eval() -> None:
    try:
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RL-Games is not installed. Run this script inside a working IsaacLab RL environment."
        ) from exc

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rl_games import RlGamesGpuEnv

    from isaaclab_factory_tasks.utils import get_checkpoint_path, load_cfg_from_registry
    from isaaclab_factory_tasks.utils.rl_games_sac import (
        FactoryRlGamesVecEnvWrapper,
        register_factory_rl_games_sac,
        upgrade_factory_sac_agent_cfg,
    )

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg = PegInsertEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    registry_agent_cfg = None
    if args_cli.checkpoint is None:
        try:
            registry_agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_sac_cfg_entry_point")
        except ValueError as exc:
            raise ValueError(
                f"{exc}\nFor this repository's RL-Games SAC setup, use the local task id "
                "'Isaac-Factory-PegInsert-Local-Direct-v0' or pass an explicit RL-Games checkpoint."
            ) from exc

        default_experiment = registry_agent_cfg["params"]["config"]["name"]
        preferred_experiment = args_cli.experiment or default_experiment
        log_root_path = _resolve_log_root("rl_games", preferred_experiment)
        print(f"[SAC] Loading experiment from directory: {log_root_path}")

        run_dir = _resolve_run_dir_pattern(registry_agent_cfg["params"]["config"].get("full_experiment_name"))
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{preferred_experiment}.pth"
        resume_path = get_checkpoint_path(str(log_root_path), run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    agent_cfg = _load_saved_agent_cfg_from_checkpoint(resume_path)
    if agent_cfg is None:
        if registry_agent_cfg is None:
            try:
                registry_agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_sac_cfg_entry_point")
            except ValueError as exc:
                raise ValueError(
                    f"{exc}\nCould not recover RL-Games SAC config from checkpoint metadata either. "
                    "Pass a checkpoint produced by this repository or use the local task id."
                ) from exc
        agent_cfg = registry_agent_cfg
    upgrade_factory_sac_agent_cfg(agent_cfg)

    rl_device = args_cli.device if args_cli.device is not None else agent_cfg["params"]["config"]["device"]
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]
    env_cfg.log_dir = str(_get_run_dir_from_checkpoint(resume_path))

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    step_dt = _get_step_dt(base_env)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = FactoryRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    print(f"[SAC] Loading checkpoint: {resume_path}")
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    _enable_player_vecenv(agent_cfg)
    runner = Runner()
    register_factory_rl_games_sac(runner)
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    total_episodes = 0
    total_successes = 0
    total_steps_list: list[int] = []
    total_time_list: list[float] = []

    device = base_env.device
    episode_steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)
    episode_success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)
    success_step = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)

    print(f"[SAC] Running {args_cli.num_episodes} episodes...")
    print(f"[SAC] Step dt = {step_dt:.4f}s, max_episode_length = {base_env.max_episode_length}")

    play_step = 0
    while total_episodes < args_cli.num_episodes and simulation_app.is_running():
        play_step += 1
        episode_steps += 1

        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=not args_cli.stochastic_policy)
            obs, _, dones, _ = env.step(actions)

        curr_successes = base_env._get_curr_successes(base_env.task_cfg.success_threshold)
        first_success = curr_successes & ~episode_success
        success_step[first_success] = episode_steps[first_success]
        episode_success |= curr_successes

        done_envs = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_envs.numel() > 0:
            for env_id in done_envs.tolist():
                total_episodes += 1
                if episode_success[env_id]:
                    total_successes += 1
                    steps = int(success_step[env_id].item())
                    total_steps_list.append(steps)
                    total_time_list.append(steps * step_dt)
                if total_episodes >= args_cli.num_episodes:
                    break

            episode_steps[done_envs] = 0
            episode_success[done_envs] = False
            success_step[done_envs] = 0

        if play_step % 100 == 0:
            _print_progress(play_step, total_episodes, total_successes, "sac")

    _print_summary("SAC Evaluation", total_episodes, total_successes, total_steps_list, total_time_list)
    env.close()


if __name__ == "__main__":
    if args_cli.eval_mode == "scripted":
        run_scripted_baseline()
    elif args_cli.eval_mode == "ppo_lstm":
        run_ppo_eval("lstm")
    elif args_cli.eval_mode == "sac":
        run_sac_eval()
    else:
        run_ppo_eval("gru")
    simulation_app.close()
