"""Scripted IK Baseline for PegInsert task."""

import argparse
import torch
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

def run_baseline():
    env_cfg = PegInsertEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-Factory-PegInsert-Refactored-Direct-v0", cfg=env_cfg)

    u = env.unwrapped
    device = u.device
    num_envs = u.num_envs

    # 每步对应的物理时间
    step_dt = u.cfg.episode_length_s / u.max_episode_length

    total_episodes = 0
    total_successes = 0
    total_steps_list = []
    total_time_list = []   # 秒

    obs, _ = env.reset()

    phase = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    print(f"[Baseline] Running {args_cli.num_episodes} episodes...")
    print(f"[Baseline] Step dt = {step_dt:.4f}s, max_episode_length = {u.max_episode_length}")

    max_steps = int(u.max_episode_length) * 5

    for step in range(max_steps):
        actions = torch.zeros((num_envs, 6), device=device)
        thresh = u.pos_threshold.clamp(min=1e-6)

        # Phase 0: XY 对齐
        phase0 = (phase == 0).nonzero(as_tuple=False).squeeze(-1)
        if phase0.numel() > 0:
            target = u.fixed_pos_obs_frame[phase0].clone()
            target[:, 2] = u.fingertip_midpoint_pos[phase0, 2]
            delta = target - u.fingertip_midpoint_pos[phase0]
            actions[phase0, 0:3] = (delta / thresh[phase0]).clamp(-1.0, 1.0)
            xy_dist = torch.norm(delta[:, 0:2], dim=-1)
            phase[phase0[xy_dist < 0.003]] = 1

        # Phase 1: Z 下降
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

        # 记录第一次成功的步数
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
                    time_s = steps * step_dt
                    total_steps_list.append(steps)
                    total_time_list.append(time_s)

            phase[done_envs] = 0
            episode_steps[done_envs] = 0
            episode_success[done_envs] = False
            success_step[done_envs] = 0

            if total_episodes >= args_cli.num_episodes:
                break

        if step % 100 == 0:
            sr = total_successes / max(total_episodes, 1) * 100
            print(f"  Step {step} | Episodes: {total_episodes} | Success: {sr:.1f}%")

    sr = total_successes / max(total_episodes, 1) * 100
    avg_steps = sum(total_steps_list) / len(total_steps_list) if total_steps_list else 0
    avg_time  = sum(total_time_list)  / len(total_time_list)  if total_time_list  else 0

    print("\n" + "="*50)
    print(f"[Scripted Baseline Results]")
    print(f"  Episodes:              {total_episodes}")
    print(f"  Successes:             {total_successes}")
    print(f"  Success rate:          {sr:.1f}%")
    print(f"  Avg steps to success:  {avg_steps:.1f}")
    print(f"  Avg time  to success:  {avg_time:.2f} s")
    print("="*50)

    env.close()

if __name__ == "__main__":
    run_baseline()
    simulation_app.close()
