# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra utilities aligned with the IsaacLab task package workflow."""

import functools
import sys
from collections.abc import Callable
from pathlib import Path

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.") from exc

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

from .parse_cfg import load_cfg_from_registry

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_OUTPUT_ROOT = REPO_ROOT / "outputs"
HYDRA_RUN_DIR = f"{HYDRA_OUTPUT_ROOT.as_posix()}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"
HYDRA_SWEEP_DIR = f"{(HYDRA_OUTPUT_ROOT / 'multirun').as_posix()}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"


def register_task_to_hydra(
    task_name: str,
    agent_cfg_entry_point: str,
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict | object | None]:
    """Register task configs into Hydra's config store."""
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point) if agent_cfg_entry_point else None

    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    env_cfg_dict = env_cfg.to_dict()
    agent_cfg_dict = agent_cfg if isinstance(agent_cfg, dict) or agent_cfg is None else agent_cfg.to_dict()

    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    cfg_dict = replace_slices_with_strings(cfg_dict)
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def _inject_repo_output_overrides(argv: list[str]) -> list[str]:
    """Ensure Hydra outputs are stored under this repository unless explicitly overridden."""
    overrides = list(argv)
    if not any(arg.startswith("hydra.run.dir=") for arg in overrides):
        overrides.append(f"hydra.run.dir={HYDRA_RUN_DIR}")
    if not any(arg.startswith("hydra.sweep.dir=") for arg in overrides):
        overrides.append(f"hydra.sweep.dir={HYDRA_SWEEP_DIR}")
    return overrides


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator that resolves task config from Gym registry and lets Hydra override it."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)

            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)

                env_cfg.from_dict(hydra_env_cfg["env"])
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)

                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    resolved_agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                    resolved_agent_cfg = agent_cfg

                func(env_cfg, resolved_agent_cfg, *args, **kwargs)

            original_argv = sys.argv[:]
            try:
                sys.argv = _inject_repo_output_overrides(sys.argv)
                hydra_main()
            finally:
                sys.argv = original_argv

        return wrapper

    return decorator
