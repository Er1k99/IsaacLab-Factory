# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Refactored direct RL environment for the Factory PegInsert task."""

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import control as peg_insert_control
from . import utils as peg_insert_utils
from .env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, PegInsertEnvCfg


class PegInsertEnv(DirectRLEnv):
    """Standalone PegInsert version of the original multi-task Factory environment."""

    cfg: PegInsertEnvCfg

    def __init__(self, cfg: PegInsertEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = sum(OBS_DIM_CFG[name] for name in cfg.obs_order) + cfg.action_space
        cfg.state_space = sum(STATE_DIM_CFG[name] for name in cfg.state_order) + cfg.action_space
        self.task_cfg = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        peg_insert_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self.task_prop_gains = self.default_gains.clone()
        self.task_deriv_gains = peg_insert_utils.get_deriv_gains(self.task_prop_gains)

    def _init_tensors(self) -> None:
        """Allocate task tensors once after the scene is created."""
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.dead_zone_thresholds = None

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        self.last_update_timestamp = 0.0
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = peg_insert_utils.identity_quat(self.num_envs, self.device)
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ee_linvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.task_prop_gains = torch.zeros((self.num_envs, 6), device=self.device)
        self.task_deriv_gains = torch.zeros((self.num_envs, 6), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _set_default_dynamics_parameters(self) -> None:
        """Set default gains, bounds, and contact parameters."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device).repeat(
            (self.num_envs, 1)
        )

        peg_insert_utils.set_friction(self._held_asset, self.task_cfg.held_asset_cfg.friction, self.scene.num_envs)
        peg_insert_utils.set_friction(self._fixed_asset, self.task_cfg.fixed_asset_cfg.friction, self.scene.num_envs)
        peg_insert_utils.set_friction(self._robot, self.task_cfg.robot_cfg.friction, self.scene.num_envs)

    def _setup_scene(self) -> None:
        """Build the simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.task_cfg.fixed_asset)
        self._held_asset = Articulation(self.task_cfg.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt: float) -> None:
        """Update tensors derived from simulator state."""
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()
        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = 0.5 * (self.left_finger_jacobian + self.right_finger_jacobian)
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]

        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat,
            torch_utils.quat_conjugate(self.prev_fingertip_quat),
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.joint_vel_fd = joint_diff / dt
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_obs_state_dict(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build observation and critic-state dictionaries."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get actor and critic observations using the configured field order."""
        obs_dict, state_dict = self._get_obs_state_dict()
        obs_tensors = peg_insert_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = peg_insert_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids: torch.Tensor) -> None:
        """Reset episode bookkeeping buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        """Apply EMA smoothing to policy actions."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._reset_buffers(env_ids)

        self.actions = self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (
            1.0 - self.cfg.ctrl.ema_factor
        ) * self.actions

    def _compute_ctrl_target_pose(
        self,
        action_delta: torch.Tensor,
        clip_to_workspace: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert delta actions into fingertip target pose commands."""
        pos_actions = action_delta[:, 0:3] * self.pos_threshold
        ctrl_target_pos = self.fingertip_midpoint_pos + pos_actions

        if clip_to_workspace:
            fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            delta_pos = ctrl_target_pos - fixed_pos_action_frame
            delta_pos = torch.maximum(torch.minimum(delta_pos, self.pos_action_bounds), -self.pos_action_bounds)
            ctrl_target_pos = fixed_pos_action_frame + delta_pos

        rot_actions = action_delta[:, 3:6] * self.rot_threshold
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = torch.zeros_like(rot_actions)
        nonzero_mask = angle > 1.0e-6
        axis[nonzero_mask] = rot_actions[nonzero_mask] / angle[nonzero_mask].unsqueeze(-1)

        rot_actions_quat = peg_insert_utils.identity_quat(self.num_envs, self.device)
        if torch.any(nonzero_mask):
            rot_actions_quat[nonzero_mask] = torch_utils.quat_from_angle_axis(angle[nonzero_mask], axis[nonzero_mask])

        ctrl_target_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_quat), dim=1)
        target_euler_xyz[:, 0] = np.pi
        target_euler_xyz[:, 1] = 0.0
        ctrl_target_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2],
        )
        return ctrl_target_pos, ctrl_target_quat

    def close_gripper_in_place(self) -> None:
        """Hold the fingertip pose while closing the gripper."""
        zero_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        ctrl_target_pos, ctrl_target_quat = self._compute_ctrl_target_pose(zero_actions, clip_to_workspace=False)
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _apply_action(self) -> None:
        """Apply policy actions as delta pose targets."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        ctrl_target_pos, ctrl_target_quat = self._compute_ctrl_target_pose(self.actions, clip_to_workspace=True)
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        ctrl_target_gripper_dof_pos: float,
    ) -> None:
        """Map fingertip-space commands into joint torques and gripper position targets."""
        self.joint_torque, self.applied_wrench = peg_insert_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate only when the synchronized episode horizon is reached."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold: float) -> torch.Tensor:
        """Return the current PegInsert success mask."""
        held_base_pos, _ = peg_insert_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.num_envs,
            self.device,
        )
        target_held_base_pos, _ = peg_insert_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = xy_dist < 0.0025
        height_threshold = self.task_cfg.fixed_asset_cfg.height * success_threshold
        is_close_or_below = z_disp < height_threshold
        return torch.logical_and(is_centered, is_close_or_below)

    def _log_metrics(self, rew_dict: dict[str, torch.Tensor], curr_successes: torch.Tensor) -> None:
        """Record success metrics and reward diagnostics."""
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded.bool()))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if nonzero_success_ids.numel() > 0:
            self.extras["success_times"] = self.ep_success_times[nonzero_success_ids].sum() / nonzero_success_ids.numel()

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward and update episode statistics."""
        curr_successes = self._get_curr_successes(success_threshold=self.task_cfg.success_threshold)
        rew_dict, rew_scales = self._get_reward_terms(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()
        self._log_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_reward_terms(
        self,
        curr_successes: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Compute reward terms for the current simulator state."""
        held_base_pos, held_base_quat = peg_insert_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.num_envs,
            self.device,
        )
        target_held_base_pos, target_held_base_quat = peg_insert_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.num_envs,
            self.device,
        )

        keypoints_held = torch.zeros((self.num_envs, self.task_cfg.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.task_cfg.num_keypoints, 3), device=self.device)
        keypoint_offsets = peg_insert_utils.get_keypoint_offsets(self.task_cfg.num_keypoints, self.device)
        keypoint_offsets *= self.task_cfg.keypoint_scale

        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                peg_insert_utils.identity_quat(self.num_envs, self.device),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                peg_insert_utils.identity_quat(self.num_envs, self.device),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        a0, b0 = self.task_cfg.keypoint_coef_baseline
        a1, b1 = self.task_cfg.keypoint_coef_coarse
        a2, b2 = self.task_cfg.keypoint_coef_fine

        action_penalty_ee = torch.norm(self.actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.task_cfg.engage_threshold)

        rew_dict = {
            "kp_baseline": peg_insert_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": peg_insert_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": peg_insert_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.task_cfg.action_penalty_ee_scale,
            "action_grad_penalty": -self.task_cfg.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset all environments using the PegInsert initialization procedure."""
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids: torch.Tensor) -> None:
        """Restore held and fixed assets to their default root states."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set arm joint positions with damped least-squares IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            pos_error, axis_angle_error = peg_insert_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            delta_dof_pos = peg_insert_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )

            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_vel[env_ids, :])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the default asset pose relative to the fingertip frame."""
        held_asset_relative_pos = torch.zeros((num_envs, 3), device=self.device)
        held_asset_relative_pos[:, 2] = self.task_cfg.held_asset_cfg.height
        held_asset_relative_pos[:, 2] -= self.task_cfg.robot_cfg.franka_fingerpad_length
        held_asset_relative_quat = peg_insert_utils.identity_quat(num_envs, self.device)
        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints: list[float], env_ids: torch.Tensor) -> None:
        """Move Franka back to the reset joint configuration."""
        gripper_width = self.task_cfg.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)

        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self) -> None:
        """Advance the simulator without policy actions during reset logic."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids: torch.Tensor) -> None:
        """Randomize fixed asset pose, robot pose, and held asset grasp pose."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]

        rand_sample = torch.rand((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.task_cfg.fixed_asset_init_pos_noise,
            dtype=torch.float32,
            device=self.device,
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]

        fixed_orn_init_yaw = np.deg2rad(self.task_cfg.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.task_cfg.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0],
            fixed_orn_euler[:, 1],
            fixed_orn_euler[:, 2],
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        fixed_state[:, 7:] = 0.0

        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        fixed_asset_pos_noise = torch.randn((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise

        self.step_sim_no_action()

        fixed_tip_pos_local = torch.zeros((env_ids.numel(), 3), device=self.device)
        fixed_tip_pos_local[:, 2] = self.task_cfg.fixed_asset_cfg.height + self.task_cfg.fixed_asset_cfg.base_height
        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat[env_ids],
            self.fixed_pos[env_ids],
            peg_insert_utils.identity_quat(env_ids.numel(), self.device),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[env_ids] = fixed_tip_pos

        bad_envs = env_ids.clone()
        ik_attempts = 0
        max_ik_attempts = 32

        while bad_envs.numel() > 0:
            n_bad = bad_envs.numel()

            above_fixed_pos = self.fixed_pos_obs_frame[bad_envs].clone()
            above_fixed_pos[:, 2] += self.task_cfg.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.task_cfg.hand_init_pos_noise, device=self.device)
            above_fixed_pos += above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)

            hand_down_euler = torch.tensor(self.task_cfg.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.task_cfg.hand_init_orn_noise, device=self.device)
            hand_down_euler += above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_quat = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0],
                pitch=hand_down_euler[:, 1],
                yaw=hand_down_euler[:, 2],
            )

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )

            pos_error_failed = torch.linalg.norm(pos_error, dim=1) > 1.0e-3
            angle_error_failed = torch.norm(aa_error, dim=1) > 1.0e-3
            bad_envs = bad_envs[torch.logical_or(pos_error_failed, angle_error_failed)]

            if bad_envs.numel() == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
                env_ids=bad_envs,
            )
            ik_attempts += 1
            if ik_attempts >= max_ik_attempts:
                raise RuntimeError("PegInsert reset IK failed to converge within 32 attempts.")

        self.step_sim_no_action()

        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(env_ids.numel(), 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat[env_ids],
            t1=self.fingertip_midpoint_pos[env_ids],
            q2=flip_z_quat,
            t2=torch.zeros((env_ids.numel(), 3), device=self.device),
        )

        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose(env_ids.numel())
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat,
            held_asset_relative_pos,
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat,
            t1=fingertip_flipped_pos,
            q2=asset_in_hand_quat,
            t2=asset_in_hand_pos,
        )

        rand_sample = torch.rand((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)
        held_asset_pos_noise_level = torch.tensor(self.task_cfg.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=peg_insert_utils.identity_quat(env_ids.numel(), self.device),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins[env_ids]
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = peg_insert_utils.get_deriv_gains(
            reset_task_prop_gains,
            self.cfg.ctrl.reset_rot_deriv_scale,
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        self.task_prop_gains = self.default_gains.clone()
        self.task_deriv_gains = peg_insert_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
