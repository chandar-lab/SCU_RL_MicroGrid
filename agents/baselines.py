import numpy as np
import copy

# Random agent
class RandomAgent:
    def __init__(self, env):
        self.battery_P_nom = env.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value']
 
    def predict(self, observation, state=None, episode_start=1, deterministic=True):
        action = {
            'genset_group': {
                'status_change': np.random.choice(['stop_last', 'start_next', 'none']),
            },
            'battery': {
                'p_grid': (np.random.random() * 2 - 1)*self.battery_P_nom
            }
        }
        return action, None


## Constant agents
class ConstantAgent:
    def __init__(self, env, action_list):
        self.action = {
            'genset_group': {
                'status_change': action_list[0],
            },
            'battery': {
                'p_grid': action_list[1]
            }
        }

    def predict(self, observation, state=None, episode_start=1, deterministic=True):
        return copy.deepcopy(self.action), None


class TurnoffDischargeAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['stop_last', 999999])

class TurnoffIdleAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['stop_last', 0])
    
class TurnoffChargeAgent(ConstantAgent):                    
    def __init__(self, env):
        super().__init__(env, ['stop_last', -999999])

        
class NeutralDischargeAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['none', 999999])

class NeutralIdleAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['none', 0])

class NeutralChargeAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['none', -999999])

    
class TurnonDischargeAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['start_next', 999999])

class TurnonIdleAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['start_next', 0])

class TurnonChargeAgent(ConstantAgent):
    def __init__(self, env):
        super().__init__(env, ['start_next', -999999])



class GreedyAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, observation, state=None, episode_start=1, deterministic=True):
        if observation['wind_turbine']['available_wind_power'] >= observation['demand']['demand']:        # If we have too much energy with the wind turbine only
            action =  {                                         # Turn off and charge "maximally" the battery
                'genset_group': {'status_change': 'stop_last'},
                'battery': {'p_grid': -999999}}                    
            
        if observation['wind_turbine']['available_wind_power'] < observation['demand']['demand']:         # If we don't have enough energy with the wind turbine only, we complete with the battery
            action =  {                                         # Try to turn off and use the battery to take the most from the wind and use the battery afterwards
                'genset_group': {'status_change': 'stop_last'},
                'battery': {'p_grid': -999999}}      

        return action, None


class RealisticAgent():
    def __init__(self, env):
        self.action_space = env.action_space
        self.mode = 'battery_charge' # Initial mode
        self.max_high_demand_steps = 5  # minutes
        self.max_low_demand_steps = 5   # minutes
        self.high_demand_steps = 0
        self.low_demand_steps = 0
        self.n_gensets = env.env_params['microgrid']['device']['init_params']['genset_group']['device']['const_params']['n_gensets']['value']
        self.minimum_power_gensets = [env.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['controller']['const_params']['minimum_load']['value'] for idx in range(self.n_gensets)]

    def predict(self, observation, state=None, episode_start=1, deterministic=True):            
        genset_status_change = self.get_genset_group_change(observation)

        battery_charge = self.get_battery_p_grid(observation)



        action =  {                                         # Turn off and discharge the battery
            'genset_group': {'status_change': genset_status_change},
            'battery': {'p_grid': battery_charge}} 
        
        return action, None


    def get_genset_group_change(self, observation):
        
        # Genset group:
        
        # By default, don't plan to change genset configuration
        genset_status_change = 'none'

        genset_in_warmup = False
        genset_in_cooldown = False
        
        
        # Check if need to turn one genset ON

        for idx in range(self.n_gensets):
            # If one genset is over 90% of available power, add a count
            print("Observation genset ", idx, ": Running: ", observation['genset_group']['gensets'][idx]['running'],", Active power: ", observation['genset_group']['gensets'][idx]['active_power'],", Available power: ", observation['genset_group']['gensets'][idx]['available_power'])
            if observation['genset_group']['gensets'][idx]['running'] and observation['genset_group']['gensets'][idx]['active_power'] > 0.9 * observation['genset_group']['gensets'][idx]['available_power']:
                self.high_demand_steps += 1
            elif observation['genset_group']['gensets'][idx]['running']:   # Otherwise, back to 0 as it has to be consecutive
                self.high_demand_steps = 0

              
            # If one genset is over 100% of available power, max the count one shot
            if observation['genset_group']['gensets'][idx]['running'] and observation['genset_group']['gensets'][idx]['active_power'] >= observation['genset_group']['gensets'][idx]['available_power']:
                self.high_demand_steps = self.max_high_demand_steps

            if observation['genset_group']['gensets'][idx]['warmup']:
                genset_in_warmup = True
            if observation['genset_group']['gensets'][idx]['cooldown']:
                genset_in_cooldown = True


        # Check if need to turn one genset OFF
        #   turn a genset off if total demand could be managed by 70% of available power of other groups for 5 minutes in a row. Not counted if at least one genset is in warmup or cooldown.
                
        if not genset_in_warmup and not genset_in_cooldown:
            available_power = []
            for idx in range(self.n_gensets):
                # If one genset is over 90% of available power, add a count
                if observation['genset_group']['gensets'][idx]['running'] or observation['genset_group']['gensets'][idx]['warmup']:
                    available_power.append(observation['genset_group']['gensets'][idx]['available_power'])
               
            if len(available_power) > 1:            # Do never turn off the last genset
                if 0.7 * sum(available_power[:-1]) > observation['genset_group']['genset_group_active_power']:
                    self.low_demand_steps += 1
                else:
                    self.low_demand_steps = 0

            else:  
                self.low_demand_steps = 0

        else:
            self.low_demand_steps = 0        # Resetting self.low_demand_steps as gensets are in warmup or cooldown

        # Set high demand steps to 0 if there is a genset in warmup - that means that it was succesfully turned ON recently
        if genset_in_warmup:
            self.high_demand_steps = 0

        # Set low demand steps to 0 if there is a genset in cooldown - that means that it was succesfully turned OFF recently
        if genset_in_cooldown:
            self.low_demand_steps = 0

        # Switch
        if self.high_demand_steps >= self.max_high_demand_steps:
            genset_status_change = 'start_next'
            # Don't reset high demand steps because the status change may be refused. It is reset at the next time step if the status change is accepted, i.e. if a group is in warmup.
        if self.low_demand_steps >= self.max_low_demand_steps:
            genset_status_change = 'stop_last'
            # Don't reset low demand steps because the status change may be refused. It is reset at the next time step if the status change is accepted, i.e. if a group is in cooldown.
        

        return genset_status_change
    
    def get_battery_p_grid(self, observation):

        if self.mode == 'battery_charge':
            # Check if need to change mode
            if observation['battery']['soc'] >= 0.9:
                self.mode = 'battery_discharge'
                # Relaunch
                battery_p_grid = self.get_battery_p_grid(observation)

            else: # Charge the battery if we have enough power with the wind + minimal power of currently running gensets
                # Compute minimal genset power
                min_genset_power = 0
                for idx in range(self.n_gensets):
                    if observation['genset_group']['gensets'][idx]['running']:
                        min_genset_power += self.minimum_power_gensets[idx]
                    elif observation['genset_group']['gensets'][idx]['warmup']:
                        min_genset_power += observation['genset_group']['gensets'][idx]['active_power']
                # Check if we have enough power
                if observation['wind_turbine']['available_wind_power'] + min_genset_power >= observation['demand']['demand']:
                    # Charge the battery
                    battery_p_grid = observation['demand']['demand'] - (observation['wind_turbine']['available_wind_power'] + min_genset_power)
                else:
                    # Don't charge the battery (the shield takes care of the rest if needed)
                    battery_p_grid = 0
            
        elif self.mode == 'battery_discharge':
            # Check if need to change mode
            if observation['battery']['soc'] <= 0.1:
                self.mode = 'battery_charge'
                # Relaunch
                battery_p_grid = self.get_battery_p_grid(observation)

            else:
            # Discharge battery to the maximum (the shield handles the rest)
                battery_p_grid = 999999



        return battery_p_grid

