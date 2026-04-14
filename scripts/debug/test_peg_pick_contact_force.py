# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug script to verify PegPick finger contact forces with ContactSensor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
        "for example: <isaaclab_root>/isaaclab.sh -p scripts/debug/test_peg_pick_contact_force.py"
    ) from exc

parser = argparse.ArgumentParser(description="Verify PegPick finger contact forces with a closed gripper grasp.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Factory-PegPick-Direct-v0",
    help="Name of the PegPick task to load.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and fabric cloning to simplify debugging.",
)
parser.add_argument(
    "--move_offset_z",
    type=float,
    default=0.0,
    help="Vertical offset applied to the grasp target before closing the gripper.",
)
parser.add_argument(
    "--open_steps",
    type=int,
    default=10,
    help="Number of simulation steps to hold the hand pose with the gripper open.",
)
parser.add_argument(
    "--close_steps",
    type=int,
    default=3000,
    help="Number of simulation steps to hold the hand pose with the gripper closed.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=100,
    help="Print contact forces every N close steps.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.utils import parse_env_cfg


def _print_force_summary(prefix: str, env) -> None:
    contact_state = env._get_grasp_contact_state()
    left_force = contact_state["left_force_norm"].detach().cpu()
    right_force = contact_state["right_force_norm"].detach().cpu()
    left_net_force = contact_state["left_net_force_norm"].detach().cpu()
    right_net_force = contact_state["right_net_force_norm"].detach().cpu()
    valid_grasp = contact_state["valid_grasp"].detach().cpu()
    left_angle = contact_state["left_contact_angle_deg"].detach().cpu()
    right_angle = contact_state["right_contact_angle_deg"].detach().cpu()
    gripper_opening = env.gripper_opening.squeeze(-1).detach().cpu()
    reach_distance = torch.linalg.vector_norm(
        env.fingertip_midpoint_pos - env.grasp_target_pos,
        dim=-1,
    ).detach().cpu()

    if env.num_envs <= 4:
        print(
            f"{prefix} | left_force_N={left_force.tolist()} | right_force_N={right_force.tolist()} "
            f"| left_net_force_N={left_net_force.tolist()} | right_net_force_N={right_net_force.tolist()} "
            f"| gripper_opening={gripper_opening.tolist()} | reach_distance={reach_distance.tolist()} "
            f"| left_angle_deg={left_angle.tolist()} | right_angle_deg={right_angle.tolist()} "
            f"| valid_grasp={valid_grasp.tolist()}"
        )
    else:
        print(
            f"{prefix} | left_force_mean={left_force.mean().item():.4f} N | "
            f"right_force_mean={right_force.mean().item():.4f} N | "
            f"left_net_force_mean={left_net_force.mean().item():.4f} N | "
            f"right_net_force_mean={right_net_force.mean().item():.4f} N | "
            f"gripper_opening_mean={gripper_opening.mean().item():.4f} | "
            f"reach_distance_mean={reach_distance.mean().item():.4f} | "
            f"valid_grasp_ratio={valid_grasp.float().mean().item():.4f}"
        )


def _patch_visual_debug_step(env_base, render_enabled: bool) -> None:
    """Force step_sim_no_action to refresh the GUI when visualization is enabled."""

    def _step_sim_no_action_with_render() -> None:
        env_base.scene.write_data_to_sim()
        env_base.sim.step(render=render_enabled)
        env_base.scene.update(dt=env_base.physics_dt)
        env_base._compute_intermediate_values(dt=env_base.physics_dt)

    env_base.step_sim_no_action = _step_sim_no_action_with_render


def main() -> None:
    use_fabric = False if args_cli.disable_fabric else None
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=use_fabric,
    )
    if args_cli.disable_fabric:
        env_cfg.scene.clone_in_fabric = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        _obs, _info = reset_output

    env_base = env.unwrapped
    _patch_visual_debug_step(env_base, render_enabled=not getattr(args_cli, "headless", False))
    env_ids = torch.arange(env_base.num_envs, device=env_base.device, dtype=torch.long)

    env_base._compute_intermediate_values(dt=env_base.physics_dt)

    target_pos = env_base.grasp_target_pos.clone()
    target_pos[:, 2] += args_cli.move_offset_z
    target_quat = env_base.fingertip_midpoint_quat.clone()

    pos_error, aa_error = env_base.set_pos_inverse_kinematics(
        ctrl_target_fingertip_midpoint_pos=target_pos,
        ctrl_target_fingertip_midpoint_quat=target_quat,
        env_ids=env_ids,
    )
    env_base.step_sim_no_action()

    print(
        "IK target reached:"
        f" pos_error_max={torch.linalg.vector_norm(pos_error, dim=-1).max().item():.6f},"
        f" rot_error_max={torch.linalg.vector_norm(aa_error, dim=-1).max().item():.6f}"
    )
    _print_force_summary("after_ik", env_base)

    for _ in range(args_cli.open_steps):
        env_base.hold_gripper_in_place(env_base.task_cfg.gripper_open_dof_pos)
        env_base.step_sim_no_action()
    _print_force_summary("after_open_hold", env_base)

    for step_idx in range(args_cli.close_steps):
        env_base.hold_gripper_in_place(0.0)
        env_base.step_sim_no_action()
        should_print = ((step_idx + 1) % args_cli.print_interval == 0) or (step_idx == args_cli.close_steps - 1)
        if should_print:
            _print_force_summary(f"close_step={step_idx + 1}", env_base)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
