# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific configuration for the refactored PegPick environment."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..peg_insert.task_cfg import ASSET_DIR, FixedAssetCfg, HeldAssetCfg, Hole8mm, Peg8mm, RobotCfg


@configclass
class PegPickTaskCfg:
    name: str = "peg_pick"
    duration_s: float = 10.0

    robot_cfg: RobotCfg = RobotCfg()
    fixed_asset_cfg: FixedAssetCfg = Hole8mm()
    held_asset_cfg: HeldAssetCfg = Peg8mm()

    # Match the holder/table working height used by the Factory peg assets so reset placement,
    # lift metrics, and drop checks all use the same z reference.
    table_height: float = 0.0
    fixed_asset_init_pos: list[float] = [0.55, 0.0, 0.05]
    fixed_asset_init_pos_noise: list[float] = [0.08, 0.08, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0
    held_asset_holder_pos_offset: list[float] = [0.0, 0.0, 0.0]

    hand_init_pos: list[float] = [0.0, 0.0, 0.06]
    hand_init_pos_noise: list[float] = [0.015, 0.015, 0.01]
    hand_init_orn: list[float] = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list[float] = [0.0, 0.0, 0.785]

    gripper_open_dof_pos: float = 0.04
    grasp_pos_local: list[float] = [0.0, 0.0, 0.0324]

    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    reach_reward_coef: list[float] = [25, 2]
    reach_reward_scale: float = 1.0
    close_reward_scale: float = 0.5
    grasp_reward_scale: float = 1.0
    lift_reward_scale: float = 2.0
    success_reward_scale: float = 3.0

    close_reward_radius: float = 0.02
    grasp_candidate_radius: float = 0.025
    gripper_close_reward_threshold: float = 0.006
    gripper_closed_threshold: float = 0.004
    grasp_contact_force_threshold: float = 0.5
    grasp_contact_angle_threshold_deg: float = 110.0
    lift_target_height: float = 0.05
    lift_success_height: float = 0.05
    success_reach_threshold: float = 0.03
    success_hold_steps: int = 5
    drop_below_table_margin: float = 0.02

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
