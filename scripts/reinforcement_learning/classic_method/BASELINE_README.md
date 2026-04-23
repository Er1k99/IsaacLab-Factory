# Conventional Baselines for PegInsert

This document describes the three conventional (non-learning) baselines implemented for the `Isaac-Factory-PegInsert-Local-Direct-v0` task, used to compare against the trained RL policies.

All baseline scripts are located in `scripts/reinforcement_learning/`.

---

## Overview

Three scripted controllers were implemented, each using a two-phase strategy:

| Script | Method | Collision Avoidance |
|--------|--------|---------------------|
| `scripted_baseline.py` | Direct IK delta-action control | None |
| `rrt_baseline.py` | Cartesian RRT path planning | None |
| `rrt2_baseline.py` | Cartesian RRT path planning | Box AABB avoidance |

All baselines share the same two-phase execution logic:

- **Phase 0**: Move the end-effector to a position directly above the hole (XY alignment)
- **Phase 1**: Descend vertically into the hole to complete the insertion

The key difference between methods lies in how Phase 0 is planned.

---

## Method Details

### 1. Scripted IK Baseline (`scripted_baseline.py`)

A purely rule-based controller with no path planning. The end-effector is commanded to move directly toward the target in a straight line using delta-action IK control.

**Phase 0 — XY Alignment:**
Compute the delta between the current fingertip position and the hole position (keeping Z fixed), then send it as a normalized action:

```
delta = fixed_pos_obs_frame[:, XY] - fingertip_pos[:, XY]
action = clamp(delta / threshold, -1, 1)
```

Transition to Phase 1 when XY error < 3 mm.

**Phase 1 — Descent:**
Move the fingertip straight down to 3 cm below the hole top:

```
target_z = fixed_pos_obs_frame[:, Z] - 0.03
action = clamp(delta / threshold, -0.5, 0.5)
```

No obstacle avoidance is performed. The path is always a straight line.

---

### 2. RRT Baseline (`rrt_baseline.py`)

A Cartesian-space RRT planner that generates a sequence of waypoints from the current end-effector position to a point 6 cm above the hole. The robot then tracks these waypoints one by one using the same delta-action IK control.

**RRT Algorithm (per environment):**

```
nodes = [start]
for _ in range(max_nodes):
    if rand() < goal_bias:
        q_rand = goal
    else:
        q_rand = start + uniform(-0.3, 0.3)

    q_near = nearest node in tree
    q_new  = q_near + step_size * direction(q_near → q_rand)

    if q_new is within workspace bounds:
        add q_new to tree

    if dist(q_new, goal) < step_size:
        add goal to tree; break

path = backtrack from goal to start
```

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `max_nodes` | 300 |
| `step_size` | 0.02 m |
| `goal_bias` | 0.3 |
| Workspace X | [0.2, 0.9] m |
| Workspace Y | [-0.5, 0.5] m |
| Workspace Z | [0.0, 0.6] m |

No collision avoidance against scene objects is performed.

---

### 3. RRT with Collision Avoidance (`rrt2_baseline.py`)

Same as `rrt_baseline.py`, but with an additional axis-aligned bounding box (AABB) collision check against the hole box (fixed asset) during tree expansion.

**Collision Check:**

The hole box is approximated as a rectangular prism with a 3 cm safety margin:

```python
half_size = [0.04, 0.04, 0.025]  # metres, approximate box half-extents
margin    = 0.03                  # safety margin

# A waypoint is rejected if ALL of the following hold:
#   |pos_x - box_x| < half_size_x + margin
#   |pos_y - box_y| < half_size_y + margin
#   pos_z < box_center_z + half_size_z + margin
```

Any sampled node that violates this check is discarded, encouraging the planner to route around the box rather than through it.

---

## Control Interface

All baselines interact with the environment through the same 6-DOF delta-action interface used by the RL policy:

```
action[0:3] = Δposition (normalised by pos_threshold)
action[3:6] = Δorientation (not used; set to zero)
```

The environment's internal operational-space controller converts these delta actions into joint torques via the Jacobian-based impedance controller defined in `control.py`.

---

## Evaluation Metrics

Each baseline is evaluated over 1024 episodes using the same success criterion as the RL policy:

**Success condition** (from `_get_curr_successes`):
- XY distance between peg base and hole centre < **2.5 mm**
- Peg insertion depth ≥ `fixed_asset_height × success_threshold` = 0.025 × 0.04 = **1 mm**

Both conditions must be satisfied simultaneously within one episode.

**Reported metrics:**
- `Success rate (%)` — fraction of episodes where the success condition is met
- `Avg steps to success` — mean number of environment steps until first success, over successful episodes only
- `Avg time to success (s)` — mean wall-clock-equivalent time (`steps × step_dt`) to first success

---

## Running the Baselines

Activate the environment first:

```bash
source ~/uv_isaaclab/bin/activate
export LD_LIBRARY_PATH=/home/cyh/uv_isaaclab/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
cd ~/IsaacLab-Factory
```

**Scripted IK:**
```bash
python scripts/reinforcement_learning/scripted_baseline.py \
  --num_envs 32 --num_episodes 1000
```

**RRT (no collision avoidance):**
```bash
python scripts/reinforcement_learning/rrt_baseline.py \
  --num_envs 32 --num_episodes 1000
```

**RRT with collision avoidance:**
```bash
python scripts/reinforcement_learning/rrt2_baseline.py \
  --num_envs 32 --num_episodes 1000
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_envs` | 32 | Number of parallel simulation environments |
| `--num_episodes` | 1000 | Total episodes to evaluate across all envs |

---

## Visualisation

To watch the robot execute a baseline policy in the Isaac Sim GUI, set `--num_envs 1` and remove the headless flag. Each script contains the line `args_cli.headless = True` — change it to `False` before running, or use the one-liner below:

```bash
# Example: visualise Scripted IK for 5 episodes
sed -i 's/args_cli.headless = True/args_cli.headless = False/' \
  scripts/reinforcement_learning/scripted_baseline.py

python scripts/reinforcement_learning/scripted_baseline.py \
  --num_envs 1 --num_episodes 5
```

Isaac Sim will open automatically. The robot will attempt peg insertion in each episode. To record the screen on Ubuntu, install and launch Kazam:

```bash
sudo apt install kazam -y
kazam &
```

Start recording in Kazam before launching the script, then stop once the episodes finish. Remember to revert `headless` back to `True` after recording if you want to run batch evaluations again:

```bash
sed -i 's/args_cli.headless = False/args_cli.headless = True/' \
  scripts/reinforcement_learning/scripted_baseline.py
```

The same steps apply for `rrt_baseline.py` and `rrt2_baseline.py`.

---

## Results (1024 episodes each)

| Method | Episodes | Success Rate | Avg Steps | Avg Time |
|--------|----------|-------------|-----------|----------|
| Scripted IK | 1024 | **72.8%** | 62.4 | 4.16 s |
| RRT (no collision avoidance) | 1024 | 24.0% | 64.8 | 4.32 s |
| RRT (with collision avoidance) | 1024 | 25.5% | 65.8 | 4.39 s |

### Key Observations

- **Scripted IK achieves the highest success rate** (72.8%) because it takes the shortest, most direct path to the target, minimising tracking error accumulation.
- **Both RRT variants perform significantly worse** (~24-25%) than Scripted IK. The random tree structure introduces unnecessary path detours, causing larger positional errors at the hole entry point.
- **RRT with collision avoidance marginally outperforms plain RRT** (25.5% vs 24.0%) by avoiding waypoints that pass through the hole box region.
- The results suggest that in obstacle-free environments, simpler direct controllers outperform sampling-based planners due to lower path tracking error.

---

## Environment Notes

- The hole position (`fixed_pos_obs_frame`) and robot initial pose are randomised at each episode reset, matching the conditions used during RL training. This ensures a fair comparison.
- The peg is already held in the gripper at episode start; baselines do not perform a separate grasping phase.
- The robot used is **Franka Panda** (not Realman 7-DOF as in the original proposal), due to simulator asset availability in Isaac Sim 4.5.
- Episode length is 10 seconds (150 steps at `step_dt ≈ 0.0667 s`).
- All BASELINE evaluations were run on an NVIDIA GeForce RTX 4080 Laptop GPU (12 GB VRAM) with 32 parallel environments.

---

## File Structure

```
scripts/reinforcement_learning/
├── scripted_baseline.py     # Phase 0: direct IK; Phase 1: descent
├── rrt_baseline.py          # Phase 0: Cartesian RRT; Phase 1: descent
├── rrt2_baseline.py         # Phase 0: Cartesian RRT + AABB check; Phase 1: descent
```
