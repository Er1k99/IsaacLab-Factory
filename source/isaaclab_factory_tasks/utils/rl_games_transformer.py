"""Custom RL-Games temporal transformer support for PegInsert PPO."""

from __future__ import annotations

import gym.spaces
import torch
import torch.nn as nn
from rl_games.algos_torch import model_builder, network_builder

try:
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper as _BaseRlGamesVecEnvWrapper
    _RL_GAMES_WRAPPER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _BaseRlGamesVecEnvWrapper = object
    _RL_GAMES_WRAPPER_IMPORT_ERROR = exc

FACTORY_TRANSFORMER_COMPONENT_NAME = "factory_transformer_actor_critic"


def is_factory_transformer_network_cfg(network_cfg: dict | None) -> bool:
    """Return True when the RL-Games network config uses the local transformer builder."""
    if not isinstance(network_cfg, dict):
        return False
    return str(network_cfg.get("name", "")).strip().lower() == FACTORY_TRANSFORMER_COMPONENT_NAME


def is_factory_transformer_agent_cfg(agent_cfg: dict | None) -> bool:
    """Return True when the agent config uses the local transformer PPO network."""
    if not isinstance(agent_cfg, dict):
        return False
    return is_factory_transformer_network_cfg(agent_cfg.get("params", {}).get("network"))


def _get_network_recurrent_type(network_cfg: dict | None) -> str | None:
    """Return the recurrent cell type configured for the local transformer network."""
    if not is_factory_transformer_network_cfg(network_cfg):
        return None

    rnn_cfg = network_cfg.get("rnn")
    if not isinstance(rnn_cfg, dict):
        return None

    recurrent_type = str(rnn_cfg.get("name", "")).strip().lower()
    if recurrent_type == "":
        return None
    return recurrent_type


def get_factory_transformer_recurrent_type(agent_cfg: dict | None) -> str | None:
    """Return the recurrent cell type configured for the local transformer PPO network."""
    if not isinstance(agent_cfg, dict):
        return None
    return _get_network_recurrent_type(agent_cfg.get("params", {}).get("network"))


def is_factory_transformer_gru_agent_cfg(agent_cfg: dict | None) -> bool:
    """Return True when the local transformer PPO network adds a GRU memory block."""
    return get_factory_transformer_recurrent_type(agent_cfg) == "gru"


def get_factory_transformer_history_length(agent_cfg: dict | None) -> int:
    """Return the configured temporal history length for the transformer PPO branch."""
    if not is_factory_transformer_agent_cfg(agent_cfg):
        raise ValueError("History length is only defined for the local transformer PPO config.")

    network_cfg = agent_cfg["params"]["network"]
    transformer_cfg = network_cfg.get("transformer")
    if not isinstance(transformer_cfg, dict):
        raise ValueError("Factory transformer PPO requires a `network.transformer` config block.")

    history_length = int(transformer_cfg.get("history_length", 0))
    if history_length <= 0:
        raise ValueError("Factory transformer PPO requires `network.transformer.history_length > 0`.")

    central_network_cfg = (
        agent_cfg.get("params", {}).get("config", {}).get("central_value_config", {}).get("network", None)
    )
    if isinstance(central_network_cfg, dict) and is_factory_transformer_network_cfg(central_network_cfg):
        central_transformer_cfg = central_network_cfg.get("transformer")
        if not isinstance(central_transformer_cfg, dict):
            raise ValueError("Central value transformer network requires its own `transformer` config block.")
        critic_history_length = int(central_transformer_cfg.get("history_length", history_length))
        if critic_history_length != history_length:
            raise ValueError(
                "Actor and central-value transformer history lengths must match. "
                f"Received actor={history_length}, critic={critic_history_length}."
            )

    return history_length


class FactoryTemporalRlGamesVecEnvWrapper(_BaseRlGamesVecEnvWrapper):
    """RL-Games wrapper that stacks the last K observations along a temporal axis."""

    def __init__(
        self,
        env,
        rl_device: str,
        clip_obs: float,
        clip_actions: float,
        obs_groups: dict[str, list[str]] | None = None,
        concate_obs_group: bool = True,
        history_length: int = 1,
    ):
        if _RL_GAMES_WRAPPER_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "Factory temporal transformer wrapper requires a working IsaacLab RL runtime."
            ) from _RL_GAMES_WRAPPER_IMPORT_ERROR
        self._history_length = int(history_length)
        if self._history_length <= 0:
            raise ValueError("Temporal history length must be positive.")

        self._obs_history: torch.Tensor | None = None
        self._states_history: torch.Tensor | None = None

        super().__init__(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_group)

    @property
    def observation_space(self) -> gym.spaces.Box:
        base_space = super().observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("Factory temporal transformer wrapper requires concatenated Box observations.")
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, (self._history_length, base_space.shape[0]))

    @property
    def state_space(self) -> gym.spaces.Box | None:
        base_space = super().state_space
        if base_space is None:
            return None
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("Factory temporal transformer wrapper requires concatenated Box critic states.")
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, (self._history_length, base_space.shape[0]))

    def reset(self):  # noqa: D102
        obs_and_states = super().reset()
        self._obs_history = self._repeat_as_history(obs_and_states["obs"])

        if "states" in obs_and_states:
            self._states_history = self._repeat_as_history(obs_and_states["states"])
        else:
            self._states_history = None

        return self._history_view()

    def step(self, actions):  # noqa: D102
        actions = actions.detach().clone().to(device=self._sim_device)
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        extras = dict(extras)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)

        terminal_history = None
        terminal_observation = extras.get("terminal_observation")
        if isinstance(terminal_observation, dict):
            terminal_processed = self._process_obs(terminal_observation)
            terminal_history = self._build_history_snapshot(terminal_processed)

        obs_and_states = self._process_obs(obs_dict)
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)

        self._obs_history = self._append_to_history(self._obs_history, obs_and_states["obs"])
        self._obs_history = self._reset_done_histories(self._obs_history, obs_and_states["obs"], dones)

        if "states" in obs_and_states:
            self._states_history = self._append_to_history(self._states_history, obs_and_states["states"])
            self._states_history = self._reset_done_histories(self._states_history, obs_and_states["states"], dones)
        else:
            self._states_history = None

        if terminal_history is not None:
            extras["terminal_observation"] = terminal_history

        extras = {
            key: value.to(device=self._rl_device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return self._history_view(), rew, dones, extras

    def _repeat_as_history(self, current: torch.Tensor) -> torch.Tensor:
        repeat_dims = [1, self._history_length] + [1] * (current.dim() - 1)
        return current.unsqueeze(1).repeat(*repeat_dims)

    def _append_to_history(self, history: torch.Tensor | None, current: torch.Tensor) -> torch.Tensor:
        if history is None:
            return self._repeat_as_history(current)
        return torch.cat([history[:, 1:], current.unsqueeze(1)], dim=1)

    def _reset_done_histories(self, history: torch.Tensor, current: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        done_mask = dones.bool()
        if not torch.any(done_mask):
            return history
        history = history.clone()
        history[done_mask] = self._repeat_as_history(current[done_mask])
        return history

    def _build_history_snapshot(self, obs_and_states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_history = self._append_to_history(self._obs_history, obs_and_states["obs"])
        history_view = {"obs": obs_history}
        if "states" in obs_and_states:
            history_view["states"] = self._append_to_history(self._states_history, obs_and_states["states"])
        return history_view

    def _history_view(self) -> dict[str, torch.Tensor]:
        history_view = {"obs": self._obs_history}
        if self._states_history is not None:
            history_view["states"] = self._states_history
        return history_view


class FactoryTransformerA2CBuilder(network_builder.NetworkBuilder):
    """Temporal transformer actor-critic builder compatible with RL-Games PPO."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        """Temporal transformer torso with optional GRU memory and PPO policy/value heads."""

        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            super().__init__()
            self.load(params)

            if self.separate:
                raise NotImplementedError("Factory transformer PPO does not support separate actor/critic towers.")
            if self.has_cnn:
                raise NotImplementedError("Factory transformer PPO expects vector observations stacked over time.")
            if self.has_rnn and self.rnn_name != "gru":
                raise NotImplementedError("Factory transformer PPO currently supports only `rnn.name: gru`.")
            if self.has_rnn and not self.is_rnn_before_mlp:
                raise ValueError("Factory transformer PPO requires `rnn.before_mlp: True` for transformer + GRU.")
            if self.has_rnn and (self.rnn_concat_input or self.rnn_concat_output):
                raise ValueError(
                    "Factory transformer PPO does not support `rnn.concat_input` or `rnn.concat_output`."
                )

            sequence_length, feature_dim = self._resolve_temporal_shape(input_shape)
            if sequence_length != self.history_length:
                raise ValueError(
                    "Temporal transformer input sequence length does not match the configured history length. "
                    f"Expected {self.history_length}, received {sequence_length}."
                )
            if feature_dim <= 0:
                raise ValueError("Temporal transformer feature dimension must be positive.")

            self.input_projection = nn.Linear(feature_dim, self.hidden_size, bias=self.use_bias)
            self.token_norm = nn.LayerNorm(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout)

            encoder_sequence_length = self.history_length + (1 if self.pooling == "cls" else 0)
            self.positional_embedding = nn.Parameter(torch.zeros(1, encoder_sequence_length, self.hidden_size))
            if self.pooling == "cls":
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            else:
                self.register_parameter("cls_token", None)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.feedforward_size,
                dropout=self.dropout,
                activation=self.transformer_activation,
                batch_first=True,
                norm_first=self.norm_first,
                bias=self.use_bias,
            )
            encoder_norm = nn.LayerNorm(self.hidden_size) if self.final_norm else None
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, norm=encoder_norm)

            head_input_size = self.hidden_size
            if self.has_rnn:
                self.rnn = self._build_rnn(self.rnn_name, self.hidden_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.layer_norm = nn.LayerNorm(self.rnn_units)
                head_input_size = self.rnn_units

            out_size = head_input_size if len(self.units) == 0 else self.units[-1]
            mlp_args = {
                "input_size": head_input_size,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.head_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
                mu_init = self.init_factory.create(**self.space_config["mu_init"])
                self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True
                    )
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if getattr(module, "bias", None) is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)

            torch.nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
            if self.cls_token is not None:
                torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if getattr(self.mu, "bias", None) is not None:
                    torch.nn.init.zeros_(self.mu.bias)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)
                    if getattr(self.sigma, "bias", None) is not None:
                        torch.nn.init.zeros_(self.sigma.bias)

        def _resolve_temporal_shape(self, input_shape) -> tuple[int, int]:
            if isinstance(input_shape, dict):
                if "observation" in input_shape:
                    input_shape = input_shape["observation"]
                else:
                    input_shape = next(iter(input_shape.values()))
            if isinstance(input_shape, int):
                raise ValueError(
                    "Factory temporal transformer PPO expects observations shaped as (history_length, feature_dim)."
                )
            if len(input_shape) != 2:
                raise ValueError(
                    "Factory temporal transformer PPO expects 2D per-sample observations shaped as "
                    f"(history_length, feature_dim), received: {input_shape}."
                )
            return int(input_shape[0]), int(input_shape[1])

        def _encode_time_sequence(self, obs: torch.Tensor) -> torch.Tensor:
            if obs.dim() == 2:
                obs = obs.view(obs.size(0), self.history_length, -1)
            elif obs.dim() != 3:
                raise ValueError(
                    "Factory temporal transformer PPO expects batched observations shaped as (B, T, F). "
                    f"Received tensor with shape: {tuple(obs.shape)}."
                )

            x = self.input_projection(obs)
            if self.cls_token is not None:
                cls_token = self.cls_token.expand(obs.size(0), -1, -1)
                x = torch.cat([cls_token, x], dim=1)

            x = x + self.positional_embedding[:, : x.size(1)]
            x = self.token_norm(x)
            x = self.input_dropout(x)
            x = self.encoder(x)

            if self.pooling == "cls":
                return x[:, 0]
            return x.mean(dim=1)

        def _build_zero_rnn_state(self, num_seqs: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
            state_shape = (self.rnn_layers, num_seqs, self.rnn_units)
            return (torch.zeros(state_shape, device=device, dtype=dtype),)

        def _apply_gru_memory(self, encoded: torch.Tensor, obs_dict) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            seq_length = int(obs_dict.get("seq_length", 1))
            if seq_length <= 0:
                raise ValueError("Factory transformer PPO requires `seq_length > 0` when using a GRU block.")

            batch_size = encoded.size(0)
            if batch_size % seq_length != 0:
                raise ValueError(
                    "Factory transformer PPO GRU received a batch size that is not divisible by `seq_length`. "
                    f"Received batch_size={batch_size}, seq_length={seq_length}."
                )

            num_seqs = batch_size // seq_length
            encoded = encoded.reshape(num_seqs, seq_length, -1).transpose(0, 1)

            dones = obs_dict.get("dones")
            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

            states = obs_dict.get("rnn_states")
            if states is None:
                states = self._build_zero_rnn_state(num_seqs, encoded.device, encoded.dtype)
            if len(states) != 1:
                raise ValueError(
                    "Factory transformer PPO GRU expects exactly one GRU hidden-state tensor. "
                    f"Received {len(states)} tensors."
                )

            state = states[0]
            bptt_len = obs_dict.get("bptt_len", 0)
            encoded, state = self.rnn(encoded, state, dones, bptt_len)
            encoded = encoded.transpose(0, 1).contiguous().reshape(batch_size, -1)

            if self.rnn_ln:
                encoded = self.layer_norm(encoded)

            return encoded, (state,)

        def forward(self, obs_dict):
            out = self._encode_time_sequence(obs_dict["obs"])
            states = None
            if self.has_rnn:
                out, states = self._apply_gru_memory(out, obs_dict)
            out = self.head_mlp(out)

            value = self.value_act(self.value(out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, sigma, value, states

            raise RuntimeError("Factory transformer PPO network could not determine the action space type.")

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            return (torch.zeros((self.rnn_layers, self.num_seqs, self.rnn_units)),)

        def get_value_layer(self):
            return self.value

        def get_aux_loss(self):
            return None

        def load(self, params):
            self.separate = params.get("separate", False)
            self.units = params["mlp"]["units"]
            self.activation = params["mlp"]["activation"]
            self.initializer = params["mlp"]["initializer"]
            self.is_d2rl = params["mlp"].get("d2rl", False)
            self.norm_only_first_layer = params["mlp"].get("norm_only_first_layer", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_rnn = "rnn" in params
            self.has_cnn = "cnn" in params
            self.has_space = "space" in params
            self.central_value = params.get("central_value", False)

            if self.has_space:
                self.is_multi_discrete = "multi_discrete" in params["space"]
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                    self.fixed_sigma = self.space_config["fixed_sigma"]
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
                elif self.is_multi_discrete:
                    self.space_config = params["space"]["multi_discrete"]
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_rnn:
                self.rnn_units = int(params["rnn"]["units"])
                self.rnn_layers = int(params["rnn"]["layers"])
                self.rnn_name = str(params["rnn"]["name"]).strip().lower()
                self.rnn_ln = bool(params["rnn"].get("layer_norm", False))
                self.is_rnn_before_mlp = bool(params["rnn"].get("before_mlp", True))
                self.rnn_concat_input = bool(params["rnn"].get("concat_input", False))
                self.rnn_concat_output = bool(params["rnn"].get("concat_output", False))
            else:
                self.rnn_units = 0
                self.rnn_layers = 0
                self.rnn_name = ""
                self.rnn_ln = False
                self.is_rnn_before_mlp = True
                self.rnn_concat_input = False
                self.rnn_concat_output = False

            transformer_cfg = params.get("transformer")
            if not isinstance(transformer_cfg, dict):
                raise ValueError("Factory transformer PPO requires a `network.transformer` config block.")

            self.history_length = int(transformer_cfg.get("history_length", 0))
            self.hidden_size = int(transformer_cfg.get("hidden_size", 128))
            self.num_heads = int(transformer_cfg.get("num_heads", 4))
            self.num_layers = int(transformer_cfg.get("num_layers", 4))
            self.feedforward_size = int(transformer_cfg.get("feedforward_size", self.hidden_size * 4))
            self.dropout = float(transformer_cfg.get("dropout", 0.0))
            self.pooling = str(transformer_cfg.get("pooling", "cls")).strip().lower()
            self.transformer_activation = str(transformer_cfg.get("activation", "gelu")).strip().lower()
            self.norm_first = bool(transformer_cfg.get("norm_first", True))
            self.final_norm = bool(transformer_cfg.get("final_norm", True))
            self.use_bias = bool(transformer_cfg.get("use_bias", True))

            if self.history_length <= 0:
                raise ValueError("Factory transformer PPO requires `history_length > 0`.")
            if self.hidden_size <= 0:
                raise ValueError("Factory transformer PPO requires `hidden_size > 0`.")
            if self.num_heads <= 0 or self.hidden_size % self.num_heads != 0:
                raise ValueError("Factory transformer PPO requires `hidden_size` to be divisible by `num_heads`.")
            if self.num_layers <= 0:
                raise ValueError("Factory transformer PPO requires `num_layers > 0`.")
            if self.feedforward_size < self.hidden_size:
                raise ValueError("Factory transformer PPO requires `feedforward_size >= hidden_size`.")
            if self.pooling not in {"cls", "mean"}:
                raise ValueError("Factory transformer PPO supports only `pooling: cls` or `pooling: mean`.")
            if self.transformer_activation not in {"relu", "gelu"}:
                raise ValueError("Factory transformer PPO supports only `activation: relu` or `activation: gelu`.")
            if self.has_rnn and self.rnn_units <= 0:
                raise ValueError("Factory transformer PPO requires `rnn.units > 0` when using a GRU block.")
            if self.has_rnn and self.rnn_layers <= 0:
                raise ValueError("Factory transformer PPO requires `rnn.layers > 0` when using a GRU block.")

    def build(self, name, **kwargs):
        return FactoryTransformerA2CBuilder.Network(self.params, **kwargs)


def register_factory_rl_games_transformer() -> None:
    """Register the local temporal transformer PPO network with RL-Games."""
    model_builder.register_network(FACTORY_TRANSFORMER_COMPONENT_NAME, FactoryTransformerA2CBuilder)
