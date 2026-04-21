"""Local RL-Games SAC extensions for IsaacLab Factory tasks.

This module patches two SAC limitations in upstream ``rl_games`` for the current
Factory tasks:

1. Store the true terminal observation in replay instead of the post-reset observation.
2. Use asymmetric actor/critic inputs so the critic consumes privileged state.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from rl_games.algos_torch import model_builder, models, network_builder, players, sac_agent, torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper


FACTORY_SAC_COMPONENT_NAME = "factory_soft_actor_critic"


class FactoryVectorizedReplayBuffer:
    """Replay buffer that stores both actor observations and critic states."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        capacity: int,
        device: str,
    ):
        self.device = device

        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.states = torch.empty((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_states = torch.empty((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0:overflow] = obs[-overflow:]
            self.states[0:overflow] = state[-overflow:]
            self.actions[0:overflow] = action[-overflow:]
            self.rewards[0:overflow] = reward[-overflow:]
            self.next_obses[0:overflow] = next_obs[-overflow:]
            self.next_states[0:overflow] = next_state[-overflow:]
            self.dones[0:overflow] = done[-overflow:]
            self.full = True

        self.obses[self.idx : self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.states[self.idx : self.idx + remaining_capacity] = state[:remaining_capacity]
        self.actions[self.idx : self.idx + remaining_capacity] = action[:remaining_capacity]
        self.rewards[self.idx : self.idx + remaining_capacity] = reward[:remaining_capacity]
        self.next_obses[self.idx : self.idx + remaining_capacity] = next_obs[:remaining_capacity]
        self.next_states[self.idx : self.idx + remaining_capacity] = next_state[:remaining_capacity]
        self.dones[self.idx : self.idx + remaining_capacity] = done[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        idxs = torch.randint(
            0,
            self.capacity if self.full else self.idx,
            (batch_size,),
            device=self.device,
        )
        return (
            self.obses[idxs],
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obses[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )


class FactorySACBuilder(network_builder.NetworkBuilder):
    """SAC network builder with asymmetric actor/critic inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return FactorySACBuilder.Network(self.params, **kwargs)

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            kwargs.pop("actions_num")
            kwargs.pop("input_shape")
            obs_dim = kwargs.pop("obs_dim")
            action_dim = kwargs.pop("action_dim")
            state_dim = kwargs.pop("state_dim", obs_dim)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            super().__init__()
            self.load(params)

            actor_mlp_args = {
                "input_size": obs_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            critic_mlp_args = {
                "input_size": state_dim + action_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }

            self.actor = network_builder.DiagGaussianActor(2 * action_dim, self.log_std_bounds, **actor_mlp_args)

            if self.separate:
                self.critic = network_builder.DoubleQCritic(1, **critic_mlp_args)
                self.critic_target = network_builder.DoubleQCritic(1, **critic_mlp_args)
                self.critic_target.load_state_dict(self.critic.state_dict())

            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if getattr(module, "bias", None) is not None:
                        torch.nn.init.zeros_(module.bias)

        def forward(self, obs_dict):
            return self.actor(obs_dict["obs"])

        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get("separate", True)
            self.units = params["mlp"]["units"]
            self.activation = params["mlp"]["activation"]
            self.initializer = params["mlp"]["initializer"]
            self.is_d2rl = params["mlp"].get("d2rl", False)
            self.norm_only_first_layer = params["mlp"].get("norm_only_first_layer", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_space = "space" in params
            self.value_shape = params.get("value_shape", 1)
            self.central_value = params.get("central_value", False)
            self.joint_obs_actions_config = params.get("joint_obs_actions", None)
            self.log_std_bounds = params.get("log_std_bounds", None)

            if self.has_space:
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
            else:
                self.is_discrete = False
                self.is_continuous = False


class FactoryModelSACContinuous(models.BaseModel):
    """SAC model with separate running statistics for actor observations and critic states."""

    def __init__(self, network):
        super().__init__(FACTORY_SAC_COMPONENT_NAME)
        self.network_builder = network

    def build(self, config):
        obs_shape = config["input_shape"]
        state_shape = config.get("state_shape")
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        value_size = config.get("value_size", 1)
        return self.Network(
            self.network_builder.build(self.model_class, **config),
            obs_shape=obs_shape,
            state_shape=state_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
        )

    class Network(models.BaseModelNetwork):
        def __init__(self, sac_network, state_shape=None, **kwargs):
            super().__init__(**kwargs)
            self.sac_network = sac_network
            self.state_shape = tuple(state_shape) if state_shape is not None else None
            if self.normalize_input and self.state_shape is not None and tuple(self.obs_shape) != self.state_shape:
                self.state_running_mean_std = RunningMeanStd(self.state_shape)

        def get_aux_loss(self):
            return self.sac_network.get_aux_loss()

        def critic(self, state, action):
            return self.sac_network.critic(state, action)

        def critic_target(self, state, action):
            return self.sac_network.critic_target(state, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)

        def is_rnn(self):
            return False

        def norm_critic_state(self, state):
            with torch.no_grad():
                if hasattr(self, "state_running_mean_std"):
                    return self.state_running_mean_std(state)
                if self.normalize_input and self.state_shape is not None and tuple(self.obs_shape) == self.state_shape:
                    return self.running_mean_std(state)
                return state

        def forward(self, input_dict):
            return self.actor(self.norm_obs(input_dict["obs"]))


class FactorySACAgent(sac_agent.SACAgent):
    """RL-Games SAC agent with terminal-observation replay and asymmetric critic input."""

    def __init__(self, base_name, params):
        self.config = config = params["config"]
        print(config)

        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input = config.get("normalize_input", False)

        self.max_env_steps = config.get("max_env_steps", 1000)
        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True

        action_space = self.env_info["action_space"]
        state_space = self.env_info.get("state_space")
        if state_space is None or state_space.shape[0] == 0:
            raise ValueError(
                "Factory SAC requires a non-empty critic state space. "
                "Set `env.obs_groups.states: [critic]` in the RL-Games SAC config."
            )

        self.actions_num = action_space.shape[0]
        self.action_range = [
            float(action_space.low.min()),
            float(action_space.high.max()),
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        state_shape = state_space.shape
        net_config = {
            "obs_dim": self.env_info["observation_space"].shape[0],
            "state_dim": state_shape[0],
            "action_dim": action_space.shape[0],
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "state_shape": state_shape,
            "normalize_input": self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        self.actor_optimizer = torch.optim.Adam(
            self.model.sac_network.actor.parameters(),
            lr=float(self.config["actor_lr"]),
            betas=self.config.get("actor_betas", [0.9, 0.999]),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.sac_network.critic.parameters(),
            lr=float(self.config["critic_lr"]),
            betas=self.config.get("critic_betas", [0.9, 0.999]),
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(self.config["alpha_lr"]),
            betas=self.config.get("alphas_betas", [0.9, 0.999]),
        )

        self.replay_buffer = FactoryVectorizedReplayBuffer(
            self.env_info["observation_space"].shape,
            state_shape,
            action_space.shape,
            self.replay_buffer_size,
            self._device,
        )
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -action_space.shape[0]
        self.algo_observer = config["features"]["observer"]

    def get_weights(self):
        state = {
            "actor": self.model.sac_network.actor.state_dict(),
            "critic": self.model.sac_network.critic.state_dict(),
            "critic_target": self.model.sac_network.critic_target.state_dict(),
        }
        if self.normalize_input and hasattr(self.model, "running_mean_std"):
            state["running_mean_std"] = self.model.running_mean_std.state_dict()
        if hasattr(self.model, "state_running_mean_std"):
            state["state_running_mean_std"] = self.model.state_running_mean_std.state_dict()
        return state

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights["actor"])
        self.model.sac_network.critic.load_state_dict(weights["critic"])
        self.model.sac_network.critic_target.load_state_dict(weights["critic_target"])

        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        if hasattr(self.model, "state_running_mean_std") and "state_running_mean_std" in weights:
            self.model.state_running_mean_std.load_state_dict(weights["state_running_mean_std"])

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs["obs"]
        return self.model.norm_obs(obs)

    def preproc_state(self, state):
        return self.model.norm_critic_state(state)

    @staticmethod
    def _ensure_tensor_obs(obs, key_name: str):
        if isinstance(obs, dict):
            raise TypeError(
                f"Factory SAC expects concatenated observation groups, but received a Dict for '{key_name}'. "
                "Set `concate_obs_groups: True` in the RL-Games SAC config."
            )
        return obs

    def _split_obs(self, obs):
        if isinstance(obs, dict):
            actor_obs = self._ensure_tensor_obs(obs["obs"], "obs")
            critic_state = self._ensure_tensor_obs(obs.get("states", actor_obs), "states")
            return actor_obs, critic_state
        return obs, obs

    @staticmethod
    def _select_done_values(done_mask: torch.Tensor, terminal_values: torch.Tensor, rollout_values: torch.Tensor):
        if terminal_values.shape != rollout_values.shape:
            raise ValueError("Terminal observations must match rollout observation shapes.")
        broadcast_shape = (done_mask.shape[0],) + (1,) * (rollout_values.ndim - 1)
        return torch.where(done_mask.view(broadcast_shape), terminal_values, rollout_values)

    def _resolve_next_replay_tensors(self, next_obs, infos, dones):
        next_actor_obs, next_states = self._split_obs(next_obs)
        terminal_observation = infos.get("terminal_observation")
        if terminal_observation is None:
            return next_actor_obs, next_states

        terminal_actor_obs, terminal_states = self._split_obs(terminal_observation)
        done_mask = dones.to(dtype=torch.bool)
        if not torch.any(done_mask):
            return next_actor_obs, next_states

        return (
            self._select_done_values(done_mask, terminal_actor_obs, next_actor_obs),
            self._select_done_values(done_mask, terminal_states, next_states),
        )

    def update_critic(self, obs, state, action, reward, next_obs, next_state, not_done, step):
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_q1, target_q2 = self.model.critic_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * log_prob

            target_q = reward + (not_done * self.gamma * target_v)
            target_q = target_q.detach()

        current_q1, current_q2 = self.model.critic(state, action)

        critic1_loss = self.c_loss(current_q1, target_q)
        critic2_loss = self.c_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def update_actor_and_alpha(self, obs, state, step):
        for param in self.model.sac_network.critic.parameters():
            param.requires_grad = False

        dist = self.model.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = -log_prob.mean()
        actor_q1, actor_q2 = self.model.critic(state, action)
        actor_q = torch.min(actor_q1, actor_q2)

        actor_loss = (torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.model.sac_network.critic.parameters():
            param.requires_grad = True

        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        return actor_loss.detach(), entropy.detach(), self.alpha.detach(), alpha_loss

    def update(self, step):
        obs, state, action, reward, next_obs, next_state, done = self.replay_buffer.sample(self.batch_size)
        not_done = ~done

        obs = self.preproc_obs(obs)
        state = self.preproc_state(state)
        next_obs = self.preproc_obs(next_obs)
        next_state = self.preproc_state(next_state)

        critic_loss, critic1_loss, critic2_loss = self.update_critic(
            obs, state, action, reward, next_obs, next_state, not_done, step
        )
        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, state, step)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss
        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target, self.critic_tau)
        return actor_loss_info, critic1_loss, critic2_loss

    def play_steps(self, random_exploration=False):
        total_time_start = time.perf_counter()
        total_update_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []

        self.obs = self.obs if self.obs is not None else self.env_reset()
        current_obs = self.obs
        actor_obs, critic_state = self._split_obs(current_obs)

        for _ in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = torch.rand(
                    (self.num_actors, *self.env_info["action_space"].shape),
                    device=self._device,
                ) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(actor_obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.perf_counter()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.perf_counter()

            self.current_rewards += rewards
            self.current_lengths += 1
            step_time += step_end - step_start

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()
            self.algo_observer.process_infos(infos, done_indices)

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            next_actor_obs, next_state = self._resolve_next_replay_tensors(next_obs, infos, dones)

            rewards = self.rewards_shaper(rewards)
            self.replay_buffer.add(
                actor_obs,
                critic_state,
                action,
                torch.unsqueeze(rewards, 1),
                next_actor_obs,
                next_state,
                torch.unsqueeze(dones, 1),
            )

            self.obs = next_obs
            actor_obs, critic_state = self._split_obs(self.obs)

            if not random_exploration:
                self.set_train()
                update_time_start = time.perf_counter()
                actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time_end = time.perf_counter()

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                total_update_time += update_time_end - update_time_start

        total_time_end = time.perf_counter()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return (
            step_time,
            play_time,
            total_update_time,
            total_time,
            actor_losses,
            entropies,
            alphas,
            alpha_losses,
            critic1_losses,
            critic2_losses,
        )


class FactorySACPlayer(BasePlayer):
    """SAC player that restores asymmetric-critic checkpoints."""

    def __init__(self, params):
        super().__init__(params)
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.action_range = [
            float(self.env_info["action_space"].low.min()),
            float(self.env_info["action_space"].high.max()),
        ]

        self.normalize_input = self.config.get("normalize_input", False)
        state_space = self.env_info.get("state_space")
        state_shape = state_space.shape if state_space is not None else self.obs_shape
        config = {
            "obs_dim": self.env_info["observation_space"].shape[0],
            "state_dim": state_shape[0],
            "action_dim": self.env_info["action_space"].shape[0],
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "state_shape": state_shape,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": False,
            "normalize_input": self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint["actor"])
        self.model.sac_network.critic.load_state_dict(checkpoint["critic"])
        self.model.sac_network.critic_target.load_state_dict(checkpoint["critic_target"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if hasattr(self.model, "state_running_mean_std") and "state_running_mean_std" in checkpoint:
            self.model.state_running_mean_std.load_state_dict(checkpoint["state_running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension is False:
            obs = unsqueeze_obs(obs)

        obs = self.model.norm_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if not is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension is False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        return None


class FactoryRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """RL-Games wrapper that forwards processed terminal observations in infos."""

    def step(self, actions):  # noqa: D102
        actions = actions.detach().clone().to(device=self._sim_device)
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        extras = dict(extras)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)

        terminal_observation = extras.get("terminal_observation")
        if isinstance(terminal_observation, dict):
            extras["terminal_observation"] = self._process_obs(terminal_observation)

        obs_and_states = self._process_obs(obs_dict)
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            key: value.to(device=self._rl_device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs_and_states, rew, dones, extras


def upgrade_factory_sac_agent_cfg(agent_cfg: dict) -> None:
    """Upgrade SAC configs to the local asymmetric-critic implementation."""

    algo_name = agent_cfg.get("params", {}).get("algo", {}).get("name", "")
    if str(algo_name).strip().lower() != "sac":
        return

    params_cfg = agent_cfg.setdefault("params", {})
    env_cfg = params_cfg.setdefault("env", {})
    env_cfg["obs_groups"] = {"obs": ["policy"], "states": ["critic"]}
    env_cfg["concate_obs_groups"] = True

    params_cfg.setdefault("model", {})["name"] = FACTORY_SAC_COMPONENT_NAME
    params_cfg.setdefault("network", {})["name"] = FACTORY_SAC_COMPONENT_NAME


def register_factory_rl_games_sac(runner) -> None:
    """Register the local SAC builder/model/agent/player with an RL-Games runner."""

    model_builder.register_network(FACTORY_SAC_COMPONENT_NAME, FactorySACBuilder)
    model_builder.register_model(FACTORY_SAC_COMPONENT_NAME, FactoryModelSACContinuous)
    runner.algo_factory.register_builder("sac", lambda **kwargs: FactorySACAgent(**kwargs))
    runner.player_factory.register_builder("sac", lambda **kwargs: FactorySACPlayer(**kwargs))
