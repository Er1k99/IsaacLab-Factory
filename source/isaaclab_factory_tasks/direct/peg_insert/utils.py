# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the refactored PegInsert environment."""

import torch

import isaacsim.core.utils.torch as torch_utils


def identity_quat(num_envs: int, device: str) -> torch.Tensor:
    """Return batched identity quaternions."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)


def get_keypoint_offsets(num_keypoints: int, device: str) -> torch.Tensor:
    """Get uniformly spaced keypoints along a line of unit length, centered at zero."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_deriv_gains(prop_gains: torch.Tensor, rot_deriv_scale: float = 1.0) -> torch.Tensor:
    """Set critically damped task-space derivative gains."""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains


def set_friction(asset, value: float, num_envs: int) -> None:
    """Update material properties for a given asset."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value
    materials[..., 1] = value
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)


def set_body_inertias(robot, num_envs: int) -> None:
    """Offset inertias to match the armature assumptions used by the original task."""
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    robot.root_physx_view.set_inertias(inertias + offset, torch.arange(num_envs))


def get_held_base_pose(
    held_pos: torch.Tensor,
    held_quat: torch.Tensor,
    num_envs: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get current held-base pose used by success checks and reward computation."""
    held_base_pos_local = torch.zeros((num_envs, 3), device=device)
    held_base_quat_local = identity_quat(num_envs, device)
    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat,
        held_pos,
        held_base_quat_local,
        held_base_pos_local,
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(
    fixed_pos: torch.Tensor,
    fixed_quat: torch.Tensor,
    num_envs: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get target held-base pose for the PegInsert success condition."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    fixed_success_quat_local = identity_quat(num_envs, device)
    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat,
        fixed_pos,
        fixed_success_quat_local,
        fixed_success_pos_local,
    )
    return target_held_base_pos, target_held_base_quat


def squashing_fn(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Compute the bounded multi-scale reward shaping function."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def collapse_obs_dict(obs_dict: dict[str, torch.Tensor], obs_order: list[str]) -> torch.Tensor:
    """Stack observation tensors in the requested order."""
    return torch.cat([obs_dict[obs_name] for obs_name in obs_order], dim=-1)
