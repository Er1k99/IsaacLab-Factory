"""
RRT Baseline with collision avoidance for PegInsert task.
Avoids the hole box when planning the approach path.
"""

import argparse
import sys
from pathlib import Path

import torch

for parent in Path(__file__).resolve().parents:
    source_root = parent / "source"
    if (source_root / "isaaclab_factory_tasks").is_dir():
        source_root_str = str(source_root)
        if source_root_str not in sys.path:
            sys.path.insert(0, source_root_str)
        break

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_factory_tasks  # noqa: F401
from isaaclab_factory_tasks.direct.peg_insert.env_cfg import PegInsertEnvCfg


class CartesianRRT:
    """
    RRT planner in 3D Cartesian space with simple box collision avoidance.
    Avoids the hole box (fixed asset) during path planning.
    """
    def __init__(self, device, max_nodes=300, step_size=0.02, goal_bias=0.3):
        self.device = device
        self.max_nodes = max_nodes
        self.step_size = step_size
        self.goal_bias = goal_bias

        # Hole box dimensions (from task_cfg)
        # base is ~5cm x 5cm x 2.5cm (approximate from USD)
        self.box_half_size = torch.tensor([0.04, 0.04, 0.025], device=device)

    def _check_collision(self, pos, box_center):
        """
        Check if a point collides with the hole box.
        Box is approximated as an axis-aligned bounding box (AABB).
        Add safety margin of 3cm.
        """
        margin = 0.03
        half = self.box_half_size + margin
        diff = (pos - box_center).abs()
        # Only check collision if below box top + margin
        if pos[2] > box_center[2] + self.box_half_size[2] + margin:
            return False  # above box, no collision
        return bool((diff[0] < half[0]) and (diff[1] < half[1]))

    def _in_workspace(self, pos):
        x_ok = 0.2 < pos[0].item() < 0.9
        y_ok = -0.5 < pos[1].item() < 0.5
        z_ok = 0.0 < pos[2].item() < 0.6
        return x_ok and y_ok and z_ok

    def _plan_single(self, start, goal, box_center):
        nodes = [start.clone()]
        parent = [-1]

        for _ in range(self.max_nodes):
            # Goal bias sampling
            if torch.rand(1).item() < self.goal_bias:
                q_rand = goal.clone()
            else:
                q_rand = start + torch.FloatTensor(3).uniform_(-0.3, 0.3).to(self.device)

            # Find nearest node
            dists = torch.stack([torch.norm(q_rand - n) for n in nodes])
            nearest_idx = torch.argmin(dists).item()
            q_near = nodes[nearest_idx]

            # Step toward q_rand
            direction = q_rand - q_near
            dist = torch.norm(direction)
            if dist < 1e-6:
                continue
            q_new = q_near + (direction / dist) * min(self.step_size, dist.item())

            # Collision check with hole box
            if self._check_collision(q_new, box_center):
                continue

            # Workspace check
            if not self._in_workspace(q_new):
                continue

            nodes.append(q_new.clone())
            parent.append(nearest_idx)

            # Check if reached goal
            if torch.norm(q_new - goal) < self.step_size:
                nodes.append(goal.clone())
                parent.append(len(nodes) - 2)
                break

        # Backtrack to get path
        path_indices = []
        idx = len(nodes) - 1
        while idx != -1:
            path_indices.append(idx)
            idx = parent[idx]
        path_indices.reverse()

        return torch.stack([nodes[i] for i in path_indices])

    def plan(self, starts, goals, box_centers):
        paths = []
        for i in range(starts.shape[0]):
            path = self._plan_single(starts[i], goals[i], box_centers[i])
            paths.append(path)
        return paths


def run_rrt_baseline():
    env_cfg = PegInsertEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-Factory-PegInsert-Local-Direct-v0", cfg=env_cfg)

    u = env.unwrapped
    device = u.device
    num_envs = u.num_envs
    step_dt = u.cfg.episode_length_s / u.max_episode_length

    planner = CartesianRRT(device=device)

    total_episodes = 0
    total_successes = 0
    total_steps_list = []
    total_time_list = []

    obs, _ = env.reset()

    phase = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    waypoints = [None] * num_envs
    waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    def replan(env_ids):
        starts = u.fingertip_midpoint_pos[env_ids].clone()
        goals = u.fixed_pos_obs_frame[env_ids].clone()
        goals[:, 2] += 0.06
        # Box center = fixed_pos + half box height
        box_centers = u.fixed_pos[env_ids].clone()
        box_centers[:, 2] += u.task_cfg.fixed_asset_cfg.height / 2

        paths = planner.plan(starts, goals, box_centers)
        for local_i, global_i in enumerate(env_ids.tolist()):
            waypoints[global_i] = paths[local_i]
            waypoint_idx[global_i] = 0

    all_ids = torch.arange(num_envs, device=device)
    replan(all_ids)

    print(f"[RRT Baseline] Running {args_cli.num_episodes} episodes...")

    max_steps = int(u.max_episode_length) * 5

    for step in range(max_steps):
        actions = torch.zeros((num_envs, 6), device=device)
        thresh = u.pos_threshold.clamp(min=1e-6)

        # Phase 0: follow RRT waypoints
        phase0 = (phase == 0).nonzero(as_tuple=False).squeeze(-1)
        if phase0.numel() > 0:
            for env_id in phase0.tolist():
                wp = waypoints[env_id]
                if wp is None:
                    continue
                w_idx = waypoint_idx[env_id].item()
                if w_idx >= len(wp):
                    phase[env_id] = 1
                    continue
                current_wp = wp[w_idx]
                delta = current_wp - u.fingertip_midpoint_pos[env_id]
                if torch.norm(delta).item() < 0.008:
                    waypoint_idx[env_id] += 1
                else:
                    actions[env_id, 0:3] = (delta / thresh[env_id]).clamp(-1.0, 1.0)

        # Phase 1: descend into hole
        phase1 = (phase == 1).nonzero(as_tuple=False).squeeze(-1)
        if phase1.numel() > 0:
            target = u.fixed_pos_obs_frame[phase1].clone()
            target[:, 2] -= 0.03
            delta = target - u.fingertip_midpoint_pos[phase1]
            actions[phase1, 0:3] = (delta / thresh[phase1]).clamp(-0.5, 0.5)

        episode_steps += 1
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated | truncated

        curr_successes = u._get_curr_successes(u.task_cfg.success_threshold)
        first_success = curr_successes & ~episode_success
        success_step[first_success] = episode_steps[first_success]
        episode_success |= curr_successes

        done_envs = done.nonzero(as_tuple=False).squeeze(-1)
        if done_envs.numel() > 0:
            for env_id in done_envs:
                total_episodes += 1
                if episode_success[env_id]:
                    total_successes += 1
                    steps = success_step[env_id].item()
                    total_steps_list.append(steps)
                    total_time_list.append(steps * step_dt)

            phase[done_envs] = 0
            episode_steps[done_envs] = 0
            episode_success[done_envs] = False
            success_step[done_envs] = 0
            replan(done_envs)

            if total_episodes >= args_cli.num_episodes:
                break

        if step % 100 == 0:
            sr = total_successes / max(total_episodes, 1) * 100
            print(f"  Step {step} | Episodes: {total_episodes} | Success: {sr:.1f}%")

    sr = total_successes / max(total_episodes, 1) * 100
    avg_steps = sum(total_steps_list) / len(total_steps_list) if total_steps_list else 0
    avg_time  = sum(total_time_list)  / len(total_time_list)  if total_time_list  else 0

    print("\n" + "="*50)
    print(f"[RRT Baseline Results]")
    print(f"  Episodes:              {total_episodes}")
    print(f"  Successes:             {total_successes}")
    print(f"  Success rate:          {sr:.1f}%")
    print(f"  Avg steps to success:  {avg_steps:.1f}")
    print(f"  Avg time  to success:  {avg_time:.2f} s")
    print("="*50)

    env.close()


if __name__ == "__main__":
    run_rrt_baseline()
    simulation_app.close()
