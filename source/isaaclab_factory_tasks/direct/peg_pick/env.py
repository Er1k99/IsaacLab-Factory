# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL environment for a standalone Factory PegPick task."""

from __future__ import annotations

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import axis_angle_from_quat

from ..peg_insert import control as peg_pick_control
from ..peg_insert import utils as peg_pick_utils
from .env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, PegPickEnvCfg


class PegPickEnv(DirectRLEnv):
    """Standalone grasping task that learns to pick the peg from the table."""

    cfg: PegPickEnvCfg

    def __init__(self, cfg: PegPickEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = sum(OBS_DIM_CFG[name] for name in cfg.obs_order) + cfg.action_space
        cfg.state_space = sum(STATE_DIM_CFG[name] for name in cfg.state_order) + cfg.action_space
        self.task_cfg = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        peg_pick_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self.task_prop_gains = self.default_gains.clone()
        self.task_deriv_gains = peg_pick_utils.get_deriv_gains(self.task_prop_gains)

    def _init_tensors(self) -> None:
        """Allocate task tensors once after the scene is created."""
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.dead_zone_thresholds = None

        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        self.last_update_timestamp = 0.0
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = peg_pick_utils.identity_quat(self.num_envs, self.device)
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ee_linvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.task_prop_gains = torch.zeros((self.num_envs, 6), device=self.device)
        self.task_deriv_gains = torch.zeros((self.num_envs, 6), device=self.device)

        self.grasp_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.gripper_opening = torch.zeros((self.num_envs, 1), device=self.device)

        self.success_hold_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.grasp_pos_local = torch.tensor(self.task_cfg.grasp_pos_local, dtype=torch.float32, device=self.device)
        self.grasp_pos_local = self.grasp_pos_local.unsqueeze(0).repeat(self.num_envs, 1)
        self.max_gripper_opening = 2.0 * self.task_cfg.gripper_open_dof_pos
        self.grasp_contact_angle_threshold_deg = self.task_cfg.grasp_contact_angle_threshold_deg

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

        peg_pick_utils.set_friction(self._fixed_asset, self.task_cfg.fixed_asset_cfg.friction, self.scene.num_envs)
        peg_pick_utils.set_friction(self._held_asset, self.task_cfg.held_asset_cfg.friction, self.scene.num_envs)
        peg_pick_utils.set_friction(self._robot, self.task_cfg.robot_cfg.friction, self.scene.num_envs)

    def _setup_scene(self) -> None:
        """Bind scene-managed assets and spawn global scene elements."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self._robot = self.scene.articulations["robot"]
        self._fixed_asset = self.scene.articulations["fixed_asset"]
        self._held_asset = self.scene.articulations["held_asset"]
        self._left_finger_contact_sensor = self.scene.sensors["left_finger_contact"]
        self._right_finger_contact_sensor = self.scene.sensors["right_finger_contact"]

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt: float) -> None:
        """Update tensors derived from simulator state."""
        self._left_finger_contact_sensor.update(dt, force_recompute=True)
        self._right_finger_contact_sensor.update(dt, force_recompute=True)

        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w
        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        _, self.grasp_target_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            peg_pick_utils.identity_quat(self.num_envs, self.device),
            self.grasp_pos_local,
        )

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]
        self.left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_body_idx] - self.scene.env_origins
        self.right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_body_idx] - self.scene.env_origins

        jacobians = self._robot.root_physx_view.get_jacobians()
        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = 0.5 * (self.left_finger_jacobian + self.right_finger_jacobian)
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]

        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()
        self.gripper_opening = self.joint_pos[:, 7:9].sum(dim=-1, keepdim=True)

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
        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos_rel_grasp": self.fingertip_midpoint_pos - self.grasp_target_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "gripper_opening": self.gripper_opening,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "grasp_target_pos": self.grasp_target_pos,
            "gripper_opening": self.gripper_opening,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get actor and critic observations using the configured field order."""
        obs_dict, state_dict = self._get_obs_state_dict()
        obs_tensors = peg_pick_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = peg_pick_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids: torch.Tensor) -> None:
        """Reset episode bookkeeping buffers."""
        self.success_hold_buf[env_ids] = 0
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
            delta_pos = ctrl_target_pos - self.grasp_target_pos
            delta_pos = torch.maximum(torch.minimum(delta_pos, self.pos_action_bounds), -self.pos_action_bounds)
            ctrl_target_pos = self.grasp_target_pos + delta_pos

        rot_actions = action_delta[:, 3:6] * self.rot_threshold
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = torch.zeros_like(rot_actions)
        nonzero_mask = angle > 1.0e-6
        axis[nonzero_mask] = rot_actions[nonzero_mask] / angle[nonzero_mask].unsqueeze(-1)

        rot_actions_quat = peg_pick_utils.identity_quat(self.num_envs, self.device)
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

    def _compute_gripper_target(self, action_delta: torch.Tensor) -> torch.Tensor:
        """Map the final action dimension to a gripper joint target."""
        gripper_action = torch.clamp(action_delta[:, 6], min=-1.0, max=1.0)
        gripper_frac = 0.5 * (gripper_action + 1.0)
        return gripper_frac.unsqueeze(-1) * self.task_cfg.gripper_open_dof_pos

    def hold_gripper_in_place(self, gripper_dof_pos: float | torch.Tensor) -> None:
        """Hold the fingertip pose while commanding the gripper."""
        zero_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        ctrl_target_pos, ctrl_target_quat = self._compute_ctrl_target_pose(zero_actions, clip_to_workspace=False)
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_quat,
            ctrl_target_gripper_dof_pos=gripper_dof_pos,
        )

    def _apply_action(self) -> None:
        """Apply policy actions as delta pose targets and gripper targets."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        ctrl_target_pos, ctrl_target_quat = self._compute_ctrl_target_pose(self.actions, clip_to_workspace=True)
        ctrl_target_gripper_dof_pos = self._compute_gripper_target(self.actions)
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_quat,
            ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
        )

    def generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        ctrl_target_gripper_dof_pos: float | torch.Tensor,
    ) -> None:
        """Map fingertip-space commands into joint torques and gripper position targets."""
        self.joint_torque, self.applied_wrench = peg_pick_control.compute_dof_torque(
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

        if isinstance(ctrl_target_gripper_dof_pos, torch.Tensor):
            self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos.expand(-1, 2)
        else:
            self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    @staticmethod
    def _safe_normalize(vectors: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        """Normalize vectors while guarding against division by zero."""
        norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        return vectors / torch.clamp(norms, min=eps)

    def _get_grasp_contact_state(self) -> dict[str, torch.Tensor]:
        """Compute dual-finger effective-contact signals from per-finger contact sensors."""
        left_force_matrix_w = self._left_finger_contact_sensor.data.force_matrix_w
        right_force_matrix_w = self._right_finger_contact_sensor.data.force_matrix_w
        left_net_force_w = self._left_finger_contact_sensor.data.net_forces_w[:, 0, :]
        right_net_force_w = self._right_finger_contact_sensor.data.net_forces_w[:, 0, :]

        if left_force_matrix_w is None or right_force_matrix_w is None:
            left_force_w = torch.zeros((self.num_envs, 3), device=self.device)
            right_force_w = torch.zeros((self.num_envs, 3), device=self.device)
        else:
            left_force_w = left_force_matrix_w[:, 0, :, :].sum(dim=1)
            right_force_w = right_force_matrix_w[:, 0, :, :].sum(dim=1)

        left_force_norm = torch.linalg.vector_norm(left_force_w, dim=-1)
        right_force_norm = torch.linalg.vector_norm(right_force_w, dim=-1)
        left_net_force_norm = torch.linalg.vector_norm(left_net_force_w, dim=-1)
        right_net_force_norm = torch.linalg.vector_norm(right_net_force_w, dim=-1)

        left_open_dir = self._safe_normalize(self.left_finger_pos - self.fingertip_midpoint_pos)
        right_open_dir = self._safe_normalize(self.right_finger_pos - self.fingertip_midpoint_pos)

        left_angle_deg = torch.full_like(left_force_norm, 180.0)
        right_angle_deg = torch.full_like(right_force_norm, 180.0)

        left_force_mask = left_force_norm > 1.0e-8
        right_force_mask = right_force_norm > 1.0e-8

        if torch.any(left_force_mask):
            left_force_dir = self._safe_normalize(left_force_w[left_force_mask])
            left_cos = torch.sum(left_force_dir * left_open_dir[left_force_mask], dim=-1).clamp(-1.0, 1.0)
            left_angle_deg[left_force_mask] = torch.rad2deg(torch.acos(left_cos))

        if torch.any(right_force_mask):
            right_force_dir = self._safe_normalize(right_force_w[right_force_mask])
            right_cos = torch.sum(right_force_dir * right_open_dir[right_force_mask], dim=-1).clamp(-1.0, 1.0)
            right_angle_deg[right_force_mask] = torch.rad2deg(torch.acos(right_cos))

        left_valid_contact = torch.logical_and(
            left_force_norm >= self.task_cfg.grasp_contact_force_threshold,
            left_angle_deg <= self.grasp_contact_angle_threshold_deg,
        )
        right_valid_contact = torch.logical_and(
            right_force_norm >= self.task_cfg.grasp_contact_force_threshold,
            right_angle_deg <= self.grasp_contact_angle_threshold_deg,
        )
        valid_grasp = torch.logical_and(left_valid_contact, right_valid_contact)

        return {
            "left_force_w": left_force_w,
            "right_force_w": right_force_w,
            "left_force_norm": left_force_norm,
            "right_force_norm": right_force_norm,
            "left_net_force_w": left_net_force_w,
            "right_net_force_w": right_net_force_w,
            "left_net_force_norm": left_net_force_norm,
            "right_net_force_norm": right_net_force_norm,
            "left_contact_angle_deg": left_angle_deg,
            "right_contact_angle_deg": right_angle_deg,
            "left_valid_contact": left_valid_contact,
            "right_valid_contact": right_valid_contact,
            "valid_grasp": valid_grasp,
        }

    def _get_curr_successes(self, grasp_contact_state: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Return the current PegPick success mask."""
        if grasp_contact_state is None:
            grasp_contact_state = self._get_grasp_contact_state()

        lift_height = self.held_pos[:, 2] - self.task_cfg.table_height

        is_lifted = lift_height > self.task_cfg.lift_success_height
        return torch.logical_and(is_lifted, grasp_contact_state["valid_grasp"])

    def _log_metrics(
        self,
        rew_dict: dict[str, torch.Tensor],
        curr_successes: torch.Tensor,
        reach_dist: torch.Tensor,
        lift_height: torch.Tensor,
        grasp_contact_state: dict[str, torch.Tensor],
    ) -> None:
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

        self.extras["reach_distance"] = reach_dist.mean()
        self.extras["lift_height"] = lift_height.mean()
        self.extras["left_finger_contact_force"] = grasp_contact_state["left_force_norm"].mean()
        self.extras["right_finger_contact_force"] = grasp_contact_state["right_force_norm"].mean()
        self.extras["left_finger_net_contact_force"] = grasp_contact_state["left_net_force_norm"].mean()
        self.extras["right_finger_net_contact_force"] = grasp_contact_state["right_net_force_norm"].mean()
        self.extras["left_contact_angle_deg"] = grasp_contact_state["left_contact_angle_deg"].mean()
        self.extras["right_contact_angle_deg"] = grasp_contact_state["right_contact_angle_deg"].mean()
        self.extras["left_valid_contact_ratio"] = grasp_contact_state["left_valid_contact"].float().mean()
        self.extras["right_valid_contact_ratio"] = grasp_contact_state["right_valid_contact"].float().mean()
        self.extras["valid_grasp_ratio"] = grasp_contact_state["valid_grasp"].float().mean()

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward and update episode statistics."""
        grasp_contact_state = self._get_grasp_contact_state()
        curr_successes = self._get_curr_successes(grasp_contact_state)
        rew_dict, rew_scales, reach_dist, lift_height = self._get_reward_terms(curr_successes, grasp_contact_state)

        rew_buf = torch.zeros_like(rew_dict["reach"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()
        self._log_metrics(rew_dict, curr_successes, reach_dist, lift_height, grasp_contact_state)
        return rew_buf

    def _get_reward_terms(
        self,
        curr_successes: torch.Tensor,
        grasp_contact_state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor, torch.Tensor]:
        """Compute reward terms for the current simulator state."""
        reach_dist = torch.linalg.vector_norm(self.fingertip_midpoint_pos - self.grasp_target_pos, dim=1)
        lift_height = torch.clamp(self.held_pos[:, 2] - self.task_cfg.table_height, min=0.0)

        a0, b0 = self.task_cfg.reach_reward_coef
        reach_reward = peg_pick_utils.squashing_fn(reach_dist, a0, b0)

        near_grasp = reach_dist < self.task_cfg.close_reward_radius
        gripper_open_frac = torch.clamp(self.gripper_opening.squeeze(-1) / self.max_gripper_opening, min=0.0, max=1.0)
        close_reward = near_grasp.float() * (1.0 - gripper_open_frac)

        grasp_candidate = grasp_contact_state["valid_grasp"]

        lift_progress = torch.clamp(lift_height / self.task_cfg.lift_target_height, min=0.0, max=1.0)

        action_penalty_ee = torch.norm(self.actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)

        rew_dict = {
            "reach": reach_reward,
            "close": close_reward,
            "grasp": grasp_candidate.float(),
            "lift": grasp_candidate.float() * lift_progress,
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "reach": self.task_cfg.reach_reward_scale,
            "close": self.task_cfg.close_reward_scale,
            "grasp": self.task_cfg.grasp_reward_scale,
            "lift": self.task_cfg.lift_reward_scale,
            "action_penalty_ee": -self.task_cfg.action_penalty_ee_scale,
            "action_grad_penalty": -self.task_cfg.action_grad_penalty_scale,
            "curr_success": self.task_cfg.success_reward_scale,
        }
        return rew_dict, rew_scales, reach_dist, lift_height

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on sustained success, drops below the table, or timeout."""
        self._compute_intermediate_values(dt=self.physics_dt)
        grasp_contact_state = self._get_grasp_contact_state()
        curr_successes = self._get_curr_successes(grasp_contact_state)
        self.success_hold_buf = torch.where(
            curr_successes,
            self.success_hold_buf + 1,
            torch.zeros_like(self.success_hold_buf),
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.held_pos[:, 2] < (self.task_cfg.table_height - self.task_cfg.drop_below_table_margin)
        success_done = self.success_hold_buf >= self.task_cfg.success_hold_steps
        terminated = torch.logical_or(dropped, success_done)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset all environments using the PegPick initialization procedure."""
        super()._reset_idx(env_ids)
        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids: torch.Tensor) -> None:
        """Restore the holder and peg to their default root states."""
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

    def set_pos_inverse_kinematics(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set arm joint positions with damped least-squares IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            pos_error, axis_angle_error = peg_pick_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            delta_dof_pos = peg_pick_control.get_delta_dof_pos(
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

    def _set_franka_to_default_pose(self, joints: list[float], env_ids: torch.Tensor) -> None:
        """Move Franka back to the reset joint configuration."""
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = self.task_cfg.gripper_open_dof_pos
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
        """Randomize the holder pose, insert the peg into it, and place the hand above it."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] = torch.tensor(self.task_cfg.fixed_asset_init_pos, dtype=torch.float32, device=self.device)
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]

        rand_sample = torch.rand((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.task_cfg.fixed_asset_init_pos_noise,
            dtype=torch.float32,
            device=self.device,
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand

        fixed_state[:, 2] = self.task_cfg.table_height + self.scene.env_origins[env_ids, 2]

        fixed_orn_init_yaw = np.deg2rad(self.task_cfg.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.task_cfg.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0
        fixed_state[:, 3:7] = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0],
            fixed_orn_euler[:, 1],
            fixed_orn_euler[:, 2],
        )
        fixed_state[:, 7:] = 0.0

        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        holder_offset = torch.tensor(self.task_cfg.held_asset_holder_pos_offset, dtype=torch.float32, device=self.device)
        holder_offset = holder_offset.unsqueeze(0).repeat(env_ids.numel(), 1)
        held_quat, held_pos = torch_utils.tf_combine(
            fixed_state[:, 3:7],
            fixed_state[:, 0:3] - self.scene.env_origins[env_ids],
            peg_pick_utils.identity_quat(env_ids.numel(), self.device),
            holder_offset,
        )

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] = held_pos + self.scene.env_origins[env_ids]
        held_state[:, 3:7] = held_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        self.step_sim_no_action()

        bad_envs = env_ids.clone()
        ik_attempts = 0
        max_ik_attempts = 32

        while bad_envs.numel() > 0:
            n_bad = bad_envs.numel()

            above_peg_pos = self.grasp_target_pos[bad_envs].clone()
            above_peg_pos[:, 2] += self.task_cfg.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_peg_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.task_cfg.hand_init_pos_noise, device=self.device)
            above_peg_pos += above_peg_pos_rand @ torch.diag(hand_init_pos_rand)

            hand_down_euler = torch.tensor(self.task_cfg.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_peg_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.task_cfg.hand_init_orn_noise, device=self.device)
            hand_down_euler += above_peg_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_quat = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0],
                pitch=hand_down_euler[:, 1],
                yaw=hand_down_euler[:, 2],
            )

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_peg_pos,
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
                raise RuntimeError("PegPick reset IK failed to converge within 32 attempts.")

        self.step_sim_no_action()

        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = peg_pick_utils.get_deriv_gains(
            reset_task_prop_gains,
            self.cfg.ctrl.reset_rot_deriv_scale,
        )

        open_time = 0.0
        while open_time < 0.1:
            self.hold_gripper_in_place(self.task_cfg.gripper_open_dof_pos)
            self.step_sim_no_action()
            open_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        self.task_prop_gains = self.default_gains.clone()
        self.task_deriv_gains = peg_pick_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
