# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for resolving environment and agent configs from the Gym registry."""

import collections
import importlib
import inspect
import os
import re

import gymnasium as gym
import yaml

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """Load a task config object or YAML dict from the Gym registry."""
    cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        agents = collections.defaultdict(list)
        for key in gym.spec(task_name.split(":")[-1]).kwargs:
            if key.endswith("_cfg_entry_point") and key != "env_cfg_entry_point":
                spec = (
                    key.replace("_cfg_entry_point", "")
                    .replace("rl_games", "rl-games")
                    .replace("rsl_rl", "rsl-rl")
                    .split("_")
                )
                agent = spec[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)

        msg = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            msg += f"\n  |-- {agent}: {', '.join(algorithms)}"

        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{msg if agents else ''}"
        )

    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            config_file = cfg_entry_point
        else:
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            config_file = os.path.join(mod_path, file_name)

        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as file_handle:
            cfg = yaml.full_load(file_handle)
    else:
        if callable(cfg_entry_point):
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point

        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        cfg = cfg_cls() if callable(cfg_cls) else cfg_cls

    return cfg


def parse_env_cfg(
    task_name: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Load the environment config and optionally override device/fabric/env count."""
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")

    cfg.sim.device = device
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


def get_checkpoint_path(
    log_path: str,
    run_dir: str = ".*",
    checkpoint: str = ".*",
    other_dirs: list[str] | None = None,
    sort_alpha: bool = True,
) -> str:
    """Resolve a checkpoint path from a log directory."""
    try:
        runs = [os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)]
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)

        run_path = os.path.join(runs[-1], *other_dirs) if other_dirs is not None else runs[-1]
    except IndexError as exc:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.") from exc

    model_checkpoints = [file_name for file_name in os.listdir(run_path) if re.match(checkpoint, file_name)]
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")

    model_checkpoints.sort(key=lambda item: f"{item:0>15}")
    return os.path.join(run_path, model_checkpoints[-1])
