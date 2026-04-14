# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the refactored Factory PegPick task."""

import gymnasium as gym

from . import agents


def _register_env(task_id: str) -> None:
    if task_id in gym.registry:
        return

    gym.register(
        id=task_id,
        entry_point=f"{__name__}.env:PegPickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:PegPickEnvCfg",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )


_register_env("Isaac-Factory-PegPick-Direct-v0")
_register_env("Isaac-Factory-PegPick-Refactored-Direct-v0")
