import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import PPO, SAC

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, **kwargs):
        super(CustomFeaturesExtractor, self).__init__(observation_space, kwargs['features_dim'])
        self.include_demand_pred = kwargs['include_demand_pred']
        self.include_soc_tp_buffer = kwargs['include_soc_tp_buffer']
        # Add MLP to process the observations
        layers = []
        # Add the first layer
        layers.append(nn.Linear(observation_space['obs_array'].shape[0], kwargs['hidden_dim']))
        layers.append(nn.ReLU())
        # Add the intermediate layers
        for i in range(0, kwargs['num_mlp_layers']-2):
            layers.append(nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim']))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(kwargs['hidden_dim'], kwargs['features_dim']))
        self.mlp = nn.Sequential(*layers)

        # Add LSTM to process the demand and wind power predictions
        if self.include_demand_pred:
            self.demand_lstm = nn.LSTM(input_size=observation_space['demand_pred_array'].shape[0], hidden_size=kwargs['features_dim'], batch_first=True)
            self.wind_power_lstm = nn.LSTM(input_size=observation_space['available_wind_power_pred_array'].shape[0], hidden_size=kwargs['features_dim'], batch_first=True)

            self.head = nn.Linear(3*kwargs['features_dim'], kwargs['features_dim'])
            if self.include_soc_tp_buffer == 'all_soc_tp_buffer' or self.include_soc_tp_buffer == 'only_last_soc_tp':
                self.soc_tp_lstm = nn.LSTM(input_size=1, hidden_size=self.features_dim, batch_first=True)
                self.head = nn.Linear(4*kwargs['features_dim'], kwargs['features_dim'])

    def forward(self, observations):
        
        out = self.mlp(observations['obs_array'])
        
        if self.include_demand_pred:
            
            pred_array, _ = self.demand_lstm(observations['demand_pred_array'].flip([2]).transpose(1, 2)) # flip and transpose to match the shape of the input and give the closest prediction more weight
            wind_power_array, _ = self.wind_power_lstm(observations['available_wind_power_pred_array'].flip([2]).transpose(1, 2))
            concatenated_array = torch.cat((out, pred_array[:, -1, :], wind_power_array[: , -1, :]), dim=-1)
            
            if self.include_soc_tp_buffer == 'no_soc_tp_buffer':
                pass
            elif self.include_soc_tp_buffer == 'only_last_soc_tp' or self.include_soc_tp_buffer == 'all_soc_tp_buffer':
                if self.include_soc_tp_buffer == 'only_last_soc_tp':
                    soc_buffer_tensor = observations['soc_tp_buffer'].transpose(1, 2)[:, -1, :].unsqueeze(1)
                elif self.include_soc_tp_buffer == 'all_soc_tp_buffer':
                    soc_buffer_tensor = observations['soc_tp_buffer'].transpose(1, 2)
                soc_tp_buffer, _ = self.soc_tp_lstm(soc_buffer_tensor)
                concatenated_array = torch.cat((concatenated_array, soc_tp_buffer[:, -1, :]), dim=-1)
            else:
                raise ValueError("Invalid value for include_soc_tp_buffer. Must be either 'no_soc_tp_buffer', 'only_last_soc_tp' or 'all_soc_tp_buffer'.")
            
            out = self.head(concatenated_array)
        
        return out

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(*args, features_extractor_class=CustomFeaturesExtractor, features_extractor_kwargs=kwargs['features_extractor_kwargs'])


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, features_extractor_class=CustomFeaturesExtractor, features_extractor_kwargs=kwargs['features_extractor_kwargs'])


class LagrangianRewardWrapper(gym.Wrapper):
    """
    Reward-penalty constrained RL wrapper:
      r_t^lagrangian = r_t - lambda * c_t
    where c_t is computed from safety-related signals in `info`.
    """
    def __init__(self, env, lambda_init: float = 0.0, lambda_init_by_component: dict | None = None):
        super().__init__(env)
        self.constraint_components = [
            "neg_balance",
            "pos_balance",
            "battery_shield",
            "genset_shield",
            "genset_overload",
        ]
        self.lagrange_multipliers = {
            key: float(max(0.0, lambda_init)) for key in self.constraint_components
        }
        if lambda_init_by_component is not None:
            for key, value in lambda_init_by_component.items():
                if key in self.lagrange_multipliers:
                    self.lagrange_multipliers[key] = float(max(0.0, value))

        base_env = env.unwrapped
        reward_cfg = base_env.env_params["reward"]
        base_balance = getattr(base_env, "max_genset_group_active_power", 0.0)
        self.p_nom = float(
            base_env.env_params["microgrid"]["device"]["init_params"]["battery"]["device"]["const_params"]["P_nom"]["value"]
        )
        self.balance_norm = max(1e-8, float(base_balance) + self.p_nom)
        # Reuse environment reward coefficients as LSAC constraint weights.
        self.constraint_weights = {
            "neg_balance": abs(float(reward_cfg.get("neg_balance_coeff", 1.0))),
            "pos_balance": abs(float(reward_cfg.get("pos_balance_coeff", 1.0))),
            "battery_shield": abs(float(reward_cfg.get("battery_shield_coeff", 1.0))),
            "genset_shield": abs(float(reward_cfg.get("genset_shield_coeff", 1.0))),
            "genset_overload": abs(float(reward_cfg.get("genset_overload_coeff", 1.0))),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_components, weighted_components = self.compute_constraint_cost_components(info)
        lagrangian_penalty = sum(
            self.lagrange_multipliers[key] * weighted_components[key]
            for key in self.constraint_components
        )
        lagrangian_reward = float(reward) - lagrangian_penalty

        # Expose useful diagnostics for callbacks/logging.
        info["constraint_components_raw"] = raw_components
        info["constraint_components"] = weighted_components
        info["constraint_cost"] = sum(weighted_components.values())
        info["lagrangian_reward"] = lagrangian_reward
        info["lagrangian_penalty"] = lagrangian_penalty
        info["lagrange_multipliers"] = dict(self.lagrange_multipliers)
        return obs, lagrangian_reward, terminated, truncated, info

    def compute_constraint_cost_components(self, info: dict) -> tuple[dict, dict]:
        # Positive when demand is unmet (balance < 0).
        balance = float(info.get("balance", 0.0))
        neg_balance = max(0.0, -balance / self.balance_norm)
        pos_balance = max(0.0, balance / self.balance_norm)

        action_command = info.get("action_command", {})
        action_implemented = info.get("action_implemented", {})

        cmd_battery = action_command.get("battery", {}).get("p_grid", 0.0)
        imp_battery = action_implemented.get("battery", {}).get("p_grid", 0.0)
        battery_shield = abs(float(imp_battery) - float(cmd_battery)) / max(1e-8, 2.0 * self.p_nom)

        cmd_status = action_command.get("genset_group", {}).get("status_change")
        imp_status = action_implemented.get("genset_group", {}).get("status_change")
        genset_shield = 1.0 if cmd_status != imp_status else 0.0

        gensets = info.get("genset_group", {}).get("device_observations", {}).get("gensets", [])
        if isinstance(gensets, dict):
            genset_iter = gensets.values()
        else:
            genset_iter = gensets
        genset_overload = 1.0 if any(bool(g["device_observations"]["overload"]) for g in genset_iter) else 0.0

        raw = {
            "neg_balance": neg_balance,
            "pos_balance": pos_balance,
            "battery_shield": battery_shield,
            "genset_shield": genset_shield,
            "genset_overload": genset_overload,
        }
        weighted = {
            key: self.constraint_weights[key] * raw[key]
            for key in self.constraint_components
        }
        return raw, weighted


class LagrangianUpdateCallback(BaseCallback):
    """Dual-ascent update of lambda based on average episode cost."""
    def __init__(
        self,
        lambda_lr: float = 1e-3,
        cost_limit: float = 0.05,
        lambda_lrs: dict | None = None,
        cost_limits: dict | None = None,
    ):
        super().__init__()
        self.default_lambda_lr = float(lambda_lr)
        self.default_cost_limit = float(cost_limit)
        self.lambda_lrs = lambda_lrs or {}
        self.cost_limits = cost_limits or {}
        self.ep_cost = None
        self.ep_len = None
        self.wrappers = []
        self.constraint_components = []

    def _find_lagrangian_wrapper(self, env):
        cur = env
        while cur is not None:
            if isinstance(cur, LagrangianRewardWrapper):
                return cur
            cur = getattr(cur, "env", None)
        return None

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self.constraint_components = list(self.wrappers[0].constraint_components) if self.wrappers else []
        self.ep_cost = {
            key: np.zeros(n_envs, dtype=np.float64) for key in self.constraint_components
        }
        self.ep_len = np.zeros(n_envs, dtype=np.int64)
        vec_envs = getattr(self.training_env, "envs", [])
        self.wrappers = [self._find_lagrangian_wrapper(e) for e in vec_envs]
        if any(w is None for w in self.wrappers):
            raise RuntimeError("LagrangianUpdateCallback could not find LagrangianRewardWrapper in training env stack.")
        self.constraint_components = list(self.wrappers[0].constraint_components)
        self.ep_cost = {
            key: np.zeros(n_envs, dtype=np.float64) for key in self.constraint_components
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            components = info.get("constraint_components", {})
            for key in self.constraint_components:
                self.ep_cost[key][idx] += float(components.get(key, 0.0))
            self.ep_len[idx] += 1

            if bool(dones[idx]):
                wrapper = self.wrappers[idx]
                total_avg_cost = 0.0
                for key in self.constraint_components:
                    avg_cost = self.ep_cost[key][idx] / max(1, self.ep_len[idx])
                    total_avg_cost += avg_cost
                    lr = float(self.lambda_lrs.get(key, self.default_lambda_lr))
                    limit = float(self.cost_limits.get(key, self.default_cost_limit))
                    new_lambda = wrapper.lagrange_multipliers[key] + lr * (avg_cost - limit)
                    wrapper.lagrange_multipliers[key] = float(max(0.0, new_lambda))

                    self.logger.record(f"train/constraint_{key}_ep_mean", float(avg_cost))
                    self.logger.record(f"train/lambda_{key}", float(wrapper.lagrange_multipliers[key]))
                    self.ep_cost[key][idx] = 0.0

                self.logger.record("train/constraint_cost_ep_mean", float(total_avg_cost))
                self.ep_len[idx] = 0

        return True


class RLAgent:

    def __new__(self, agent_type, env, tensorboard_log, verbose, device, **kwargs):
        policy_kwargs = dict(kwargs.get("policy_kwargs", {}))
        kwargs["policy_kwargs"] = policy_kwargs
        lagrangian_cfg = {}
        if isinstance(policy_kwargs, dict):
            lagrangian_cfg = policy_kwargs.pop("lagrangian", {})

        if agent_type == "SAC":
            return SAC(CustomSACPolicy, env, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **kwargs)
        elif agent_type == "LSAC":
            lambda_init_by_component = lagrangian_cfg.get("lambda_init_by_component", None)
            lagrangian_env = LagrangianRewardWrapper(
                env,
                lambda_init=lagrangian_cfg.get("lambda_init", 0.0),
                lambda_init_by_component=lambda_init_by_component,
            )
            model = SAC(CustomSACPolicy, lagrangian_env, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **kwargs)
            model.is_lagrangian = True
            return model
        elif agent_type == "PPO":
            return PPO(CustomPPOPolicy, env, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **kwargs)
        else:
            raise ValueError("Invalid agent_type. Must be either 'SAC', 'LSAC' or 'PPO'.")