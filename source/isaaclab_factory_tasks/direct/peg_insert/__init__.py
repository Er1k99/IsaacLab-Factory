# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the local Factory PegInsert task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Factory-PegInsert-Local-Direct-v0",
    entry_point=f"{__name__}.env:PegInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PegInsertEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_gru_cfg.yaml",
        "rl_games_ppo_gru_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_gru_cfg.yaml",
        "rl_games_ppo_lstm_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "rl_games_ppo_mlp_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_mlp_cfg.yaml",
        "rl_games_ppo_transformer_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_transformer_cfg.yaml",
        "rl_games_ppo_transformer_gru_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_transformer_gru_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)
