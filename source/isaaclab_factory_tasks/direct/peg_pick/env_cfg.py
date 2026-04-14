# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the refactored PegPick task."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .task_cfg import ASSET_DIR, PegPickTaskCfg

OBS_DIM_CFG = {
    "fingertip_pos_rel_grasp": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "gripper_opening": 1,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_quat": 4,
    "grasp_target_pos": 3,
    "gripper_opening": 1,
}

SCENE_TASK_CFG = PegPickTaskCfg()


@configclass
class CtrlCfg:
    ema_factor: float = 0.2

    pos_action_bounds: list[float] = [0.05, 0.05, 0.08]
    rot_action_bounds: list[float] = [1.0, 1.0, 1.0]

    pos_action_threshold: list[float] = [0.02, 0.02, 0.02]
    rot_action_threshold: list[float] = [0.097, 0.097, 0.097]

    reset_joints: list[float] = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains: list[float] = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale: float = 10.0
    default_task_prop_gains: list[float] = [100, 100, 100, 30, 30, 30]

    default_dof_pos_tensor: list[float] = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246


@configclass
class PegPickSceneCfg(InteractiveSceneCfg):
    """Scene configuration for PegPick with scene-managed contact sensors."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    fixed_asset = SCENE_TASK_CFG.fixed_asset.replace(prim_path="{ENV_REGEX_NS}/FixedAsset")
    held_asset = SCENE_TASK_CFG.held_asset.replace(prim_path="{ENV_REGEX_NS}/HeldAsset")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),
            rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/HeldAsset/.*"],
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/HeldAsset/.*"],
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )


@configclass
class PegPickEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 7
    observation_space = 1
    state_space = 1

    obs_order: list[str] = [
        "fingertip_pos_rel_grasp",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "gripper_opening",
    ]
    state_order: list[str] = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_quat",
        "grasp_target_pos",
        "gripper_opening",
    ]

    task: PegPickTaskCfg = PegPickTaskCfg()
    ctrl: CtrlCfg = CtrlCfg()
    episode_length_s = 8.0

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # Keep PhysX GPU buffers modest so the task can start on workstation GPUs.
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**20,
            gpu_collision_stack_size=2**26,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # ContactSensor initialization counts environments via USD stage queries. Fabric cloning only exposes the
    # source env reliably to those queries, which breaks the sensor's body-count validation on cloned envs.
    scene: PegPickSceneCfg = PegPickSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)
