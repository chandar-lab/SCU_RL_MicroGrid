import torch
import torch.nn as nn
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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

class RLAgent:

    def __new__(self, agent_type, env, tensorboard_log, verbose, device, **kwargs):
        if agent_type == "SAC":
            return SAC(CustomSACPolicy, env, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **kwargs)
        elif agent_type == "PPO":
            return PPO(CustomPPOPolicy, env, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **kwargs)
        else:
            raise ValueError("Invalid agent_type. Must be either 'SAC' or 'PPO'.")