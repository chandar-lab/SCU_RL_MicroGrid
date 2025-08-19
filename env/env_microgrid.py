import gymnasium as gym

try:
    from env.microgrid import MicroGrid
    from env.microgrid_controller import MicroGridController, MicrogridInitializationError
    from env.utils import generate_init_parameters
except:
    from microgrid import MicroGrid
    from microgrid_controller import MicroGridController, MicrogridInitializationError
    from utils import generate_init_parameters
import numpy as np
import wandb
import random
import pprint
import copy

class MicroGridEnv(gym.Env):
    def __init__(self, env_params):
        super(MicroGridEnv, self).__init__()
        self.env_params = env_params
        
        wind_turbine_steps = self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nb_pred_time_steps']['value'] if 'wind_turbine' in self.env_params['microgrid']['device']['init_params'].keys() else 10
        if not self.env_params['return_dict']:
            self.action_space = gym.spaces.Box(low=np.array([-1, -1]),
                                    high=np.array([1, 1]), shape=(2,), dtype=float)
            
            self.observation_space = gym.spaces.Dict({
                'obs_array': gym.spaces.Box(low=-1, high=1, shape=(15+11*self.env_params['microgrid']['device']['init_params']['genset_group']['device']['const_params']['n_gensets']['value'],), dtype=float),
                'demand_pred_array': gym.spaces.Box(low=-1, high=1, shape=(1, self.env_params['microgrid']['device']['init_params']['demand']['device']['const_params']['nb_pred_time_steps']['value']), dtype=float),
                'available_wind_power_pred_array': gym.spaces.Box(low=-1, high=1, shape=(1, wind_turbine_steps), dtype=float),
                'soc_tp_buffer': gym.spaces.Box(low=0, high=1, shape=(1, self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['buffer_size']['value']), dtype=float), 
                })
        else:
            self.action_space = gym.spaces.Dict({
                'genset_group': gym.spaces.Dict({
                    'status_change': gym.spaces.Text(max_length = 10)
                }),
                'battery': gym.spaces.Dict({
                    'p_grid': gym.spaces.Box(low=-1, high=1, shape=(), dtype=float)
                })
            })

            gensets_space_dict = {}
            for idx in range(self.env_params['microgrid']['device']['init_params']['genset_group']['device']['const_params']['n_gensets']['value']):
                gensets_space_dict[idx] = gym.spaces.Dict({
                    'running': gym.spaces.Discrete(2),
                    'warmup': gym.spaces.Discrete(2),
                    'cooldown': gym.spaces.Discrete(2),
                    'overload': gym.spaces.Discrete(2),
                    'time_since_warmup': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'time_since_cooldown': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'time_since_start': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'active_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'available_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'average_active_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'fuel_consumption': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float)
                })

            self.observation_space = gym.spaces.Dict({
                 'genset_group': gym.spaces.Dict({
                    'genset_group_active_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'genset_group_fuel_consumption': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'genset_group_available_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'gensets': gym.spaces.Dict(gensets_space_dict)
                }),
                'battery': gym.spaces.Dict({
                    'soc': gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
                    'p_grid': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'degradation_cost': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'soc_tp_buffer': gym.spaces.Box(low=0, high=1, shape=(1,self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['buffer_size']['value']), dtype=float), 

                }),
                'demand': gym.spaces.Dict({
                    'demand': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float),
                    'demand_next': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float),
                    'demand_pred': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_params['microgrid']['device']['init_params']['demand']['device']['const_params']['nb_pred_time_steps']['value'],), dtype=float),
                }),
                'microgrid': gym.spaces.Dict({ 
                    'action_command': gym.spaces.Dict({
                        'genset_group': gym.spaces.Dict({
                            'status_change': gym.spaces.Text(max_length = 10),
                        }),
                        'battery': gym.spaces.Dict({
                            'p_grid': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float)
                        }),
                    }),
                    'action_implemented': gym.spaces.Dict({
                        'genset_group': gym.spaces.Dict({
                            'status_change': gym.spaces.Text(max_length = 10),
                            'power_setpoint': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                        }),
                        'battery': gym.spaces.Dict({
                            'p_grid': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float),
                        }),
                        'wind_turbine': gym.spaces.Dict({
                            'turbine_setpoint': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                        }),
                    }),
                    'balance': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float)
                }),
                'wind_turbine': gym.spaces.Dict({
                    'available_wind_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'wind_power': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=float),
                    'available_wind_power_pred': gym.spaces.Box(low=0, high=np.inf, shape=(wind_turbine_steps,), dtype=float)
                }),
            })

        # Initialize the microgrid and controller. Correct start will be tested in the reset function

        ## Sampling the random parameters
        microgrid_params = copy.deepcopy(self.env_params['microgrid'])
        microgrid_params = generate_init_parameters(microgrid_params)
        ## Instanciate the microgrid and controller
        self.microgrid = MicroGrid(microgrid_params['device'], time_step = self.env_params['time_step'], real=True)
        self.microgrid_controller = MicroGridController(microgrid_params, time_step = self.env_params['time_step'], verbose = False)
        ## Initialize the controller with the current state of the microgrid 
        self.microgrid_controller.update_controller_state({'device_observations': self.microgrid.gather_observations(), 'controller_state': {}})  # Initialize the controller with the current state of the microgrid  


        # Initialize the reward parameters
        self.reward_params = self.env_params['reward']
        # Compute max power and max fuel consumption for further normalization
        self.max_genset_group_active_power = 0                     # kW
        self.max_genset_group_fuel_consumption = 0          # l/h
        self.max_genset_group_fixed_consumption = 0         # l/h
        for idx in range(self.env_params['microgrid']['device']['init_params']['genset_group']['device']['const_params']['n_gensets']['value']):
            prime_power_rating = self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['device']['const_params']['prime_power_rating']['value']
            overload = self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['device']['const_params']['temp_overload_factor']['value']
            alpha_g = self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['device']['const_params']['alpha_g']['value']
            beta_g = self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['device']['const_params']['beta_g']['value']

            self.max_genset_group_active_power += prime_power_rating * overload
            self.max_genset_group_fuel_consumption += alpha_g * prime_power_rating * overload + beta_g
            self.max_genset_group_fixed_consumption += beta_g

        self.return_dict = self.env_params['return_dict']
        
        if self.env_params['train']:
            self.env_total_steps = 0

        self.episode_steps = 0
        self.reset()


    def reset(self, options=None, seed=None):

        ## Resampling the random parameters
        microgrid_params = copy.deepcopy(self.env_params['microgrid'])
        microgrid_params = generate_init_parameters(microgrid_params)

        try:
            self.microgrid.reset(microgrid_params['device']['init_params'], real = True)
            self.microgrid_controller.reset(microgrid_params['controller']['init_params'], microgrid_params['device']['init_params'])
            self.microgrid_controller.update_controller_state({'device_observations': self.microgrid.gather_observations(), 'controller_state': {}})  # Initialize the controller with the current state of the microgrid (especially important for the windturbine and demand dummies)
            verified = self.microgrid_controller.verify_initialization_validity()
            if not verified:
                raise MicrogridInitializationError("The microgrid is not initialized correctly. Check the initial parameters to ensure it is possible to recover from the initial state.")
        except MicrogridInitializationError as e:
            verified = False

        # If need to check initialization, do it and reset the microgrid and controller until it is verified
        i = 1
        while self.env_params['microgrid']['controller']['const_params']['check_initialization']['value'] and not verified and i < self.env_params['microgrid']['controller']['const_params']['init_check_loops']['value']:
            microgrid_params = copy.deepcopy(self.env_params['microgrid'])
            microgrid_params = generate_init_parameters(microgrid_params)
            try:
                self.microgrid.reset(microgrid_params['device']['init_params'], real = True)
                self.microgrid_controller.reset(microgrid_params['controller']['init_params'], microgrid_params['device']['init_params'])
                self.microgrid_controller.update_controller_state({'device_observations': self.microgrid.gather_observations(), 'controller_state': {}})  # Initialize the controller with the current state of the microgrid (especially important for the windturbine and demand dummies)
                verified = self.microgrid_controller.verify_initialization_validity()
                if not verified:
                    raise MicrogridInitializationError("The microgrid is not initialized correctly. Check the initial parameters to ensure it is possible to recover from the initial state.")
            except MicrogridInitializationError as e:
                verified = False
                i += 1        
                print("Microgrid initialization failed, re-sampling microgrid parameters, loop {}...".format(i))

        if not verified:        # After the maximum number of loops, if the microgrid is still not correctly initialized, raise an error
            raise RuntimeError("The microgrid is not correctly initialized after {} loops.".format(self.env_params['microgrid']['controller']['init_check_loops']['value']))
        
        # Run one neutral action to ensure all shields are up at first step
        base_action = {
            'genset_group': {'status_change': 'none'},
            'battery': {'p_grid': 0},
            # 'wind_turbine': {'turbine_setpoint': 0}
        }
        
        safe_action = self.microgrid_controller.generate_safe_action(base_action)
        self.microgrid.step(safe_action)
        self.microgrid_controller.update_controller_state({'device_observations': self.microgrid.gather_observations(), 'controller_state': {}}) 
        observations = self.microgrid_controller.gather_observations()          # The agent has access to the observations coming from the microgrid controller, not from the microgrid itself.
        
        observations['device_observations']['action_command'] = base_action
        observations['device_observations']['action_implemented'] = safe_action

        self.episode_steps = 0
        if not self.return_dict:
            obs = self.obs_dict_to_arrays(observations)
        else:
            obs = self.prune_obs_dict(observations)

        info = {}
        
        return obs, info

    def step(self, action, verbose = False):
        """
        Implements one step of the microgrid environment
        Inputs:
            - action: 
                if not env.params['format_dict'], numpy array with two elements: 
                    - action[0]: genset group action. If not env.params['setpoint_action'] 0 -> do nothing; 1 -> start next genset in the priority list; -1 -> stop last genset in the priority list; else: normalized setpoint number of gensets running (from -1 to 1)
                    - action[1]: normalized battery action: power to charge/discharge the battery on the grid side. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery. Normalized by P_nom
                else:
                    dictionary with the following keys:
                        - genset_group: dictionary with the following keys
                            - status_change: 'start_next', 'none', 'stop_last'
                        - battery: dictionary with the following keys
                            - p_grid: power charging/discharging the battery on the grid side: float, normalized from -1 to 1
        """
        prev_state_dict = self.microgrid.gather_observations()  # We need the microgrid state to compute the reward (including the previous one in case things change)

        if not self.return_dict:
            action_dict = self.action_norm_array_to_dict(action)
        else:
            action_dict = action

        valid_action_dict = self.microgrid_controller.generate_safe_action(action_dict)
        self.microgrid.step(valid_action_dict, verbose = verbose) # We need the microgrid state to compute the reward
        new_state_dict = self.microgrid.gather_observations()    
        self.microgrid_controller.update_controller_state({'device_observations': new_state_dict, 'controller_state': {}})  # Update the controller with the new state of the microgrid
        new_obs_dict = self.microgrid_controller.gather_observations()          # The agent has access to the observations coming from the microgrid controller, not from the microgrid itself.

        new_obs_dict['device_observations']['action_command'] = action_dict
        new_state_dict['action_command'] = action_dict
        new_obs_dict['device_observations']['action_implemented'] = valid_action_dict
        new_state_dict['action_implemented'] = valid_action_dict


        if verbose:
            print("Action: {}".format(action_dict))
            print("Previous state: {}".format(prev_state_dict))
            print("New state: {}".format(new_state_dict))
            print("New observations: {}".format(new_obs_dict))

        reward, reward_components = self.reward_function(prev_state_dict, new_state_dict, verbose = verbose)        # The state use for computing the reward comes from the microgrid, not the microgrid controller
       
       
        # The agent receives the observations from the microgrid controller, not the microgrid itself
        if self.return_dict:
            new_obs = self.prune_obs_dict(new_obs_dict)

        else:
            new_obs = self.obs_dict_to_arrays(new_obs_dict)

        terminated = False
        truncated = False
        if self.env_params['train']:
            if int(self.env_total_steps / self.env_params['max_episode_steps']) % 5 == 0: # logs to wandb every 5 episodes
                wandb.log({'reward_components/': reward_components, 
                        'previous_state/': prev_state_dict,
                        'new_observations/': new_obs_dict,
                        'actions/': {'battery_p_grid': action['battery']['p_grid'] if self.return_dict else action[1],},
                            'env_steps': self.env_total_steps})
            self.env_total_steps += 1
       
        self.episode_steps += 1
        if self.episode_steps == self.env_params['max_episode_steps']:
            truncated = True

        info = new_state_dict   # We return the actual microgrid state (and not the controller's observation) as info

        return new_obs, reward, terminated, truncated, info
                
    

    def action_norm_array_to_dict(self, action_norm_array):

        status_change_dict = {-1: 'stop_last', 0: 'none', 1: 'start_next'}
        if self.env_params['setpoint_action']:
            status_change_int = self.convert_setpoint_act_to_status_change(action_norm_array[0])
        else:
            action_space_range = self.action_space.high[0] - self.action_space.low[0]
            status_change_int = int((action_norm_array[0] + 1) / (action_space_range/3)) - 1                    # -1, 0, 1

        if status_change_int not in status_change_dict.keys():
            raise ValueError('Invalid genset group status change action_array: {}'.format(status_change_int))
        status_change = status_change_dict.get(status_change_int)

        if len(action_norm_array) < 2:          # If the agent is not designed to control the turbine
            action_norm_array[2] = 1            # No power limit is applied on the wind turbine (if there is no turbine, it does not matter; if there is a turbine, it will never be curtailed)

        action_dict = {
            'genset_group': {
                'status_change': status_change
            },
            'battery': {
                'p_grid': action_norm_array[1] * self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'] # -1 to 1 --> -P_nom to P_nom
            }
        }
        return action_dict

    def convert_setpoint_act_to_status_change(self, action):
        """
        Converts a normalized setpoint action into a status change for the genset group in the microgrid.
        The function determines whether to start, stop, or maintain the current number of running gensets
        based on the desired setpoint and the current state of the gensets. It also logs relevant data
        during training for monitoring purposes.

        Args:
            action (float): Normalized action value indicating the desired setpoint for the number of running gensets.

        Returns:
            int: Status change value (1 to start a genset, -1 to stop a genset, 0 to maintain the current state).
        """

        prev_state_dict = self.microgrid.gather_observations()
        num_gensets = len(prev_state_dict['genset_group']['device_observations']['gensets'])
        action_range = (self.action_space.high[0] - self.action_space.low[0]) / (num_gensets+1)
        setpoint_action = int((action + 1) / action_range)

        running_gensets_count = 0
        for _, genset in prev_state_dict['genset_group']['device_observations']['gensets'].items():
            if genset['controller_state']['status'] in ['warmup', 'running']:
                running_gensets_count += 1
        
        if running_gensets_count < setpoint_action:
            status_change = 1
        elif running_gensets_count > setpoint_action:
            status_change = -1
        else:
            status_change = 0

        if self.env_params['train']:
            if int(self.env_total_steps / self.env_params['max_episode_steps']) % 5 == 0: # logs to wandb every 5 episodes
                wandb.log({
                        'actions/': {'genset_group_status_change': status_change,
                                        'setpoint_action': setpoint_action},
                            'env_steps': self.env_total_steps})
                for idx, genset in prev_state_dict['genset_group']['device_observations']['gensets'].items():
                    wandb.log({
                        'observations/': {f'.genset_group.gensets.{idx}.running': int(genset['controller_state']['status'] == "running"),
                                            f'.genset_group.gensets.{idx}.warmup': int(genset['controller_state']['status']== "warmup"),
                                            f'.genset_group.gensets.{idx}.cooldown': int(genset['controller_state']['status']== "cooldown")},
                            'env_steps': self.env_total_steps})
        
        return status_change

    def reward_function(self, prev_state_dict, new_state_dict, verbose = False):
        """
        Defines the reward at a given step. The reward is computed as the sum of the performance, maintenance, and constraints rewards. All rewards are negative, objective is to be as close as possible to 0.
        """
        reward_components = {}
        # Performance 

        ## Fuel consumption: between 0 and 1
        fuel_cons = new_state_dict['genset_group']['device_observations']['genset_group_fuel_consumption']/self.max_genset_group_fuel_consumption     

        ## Controllable fuel consumption:
        controllable_fuel_cons = 0
        controllable_fuel_cons = fuel_cons - (new_state_dict['demand']['device_observations']['demand'] - new_state_dict['wind_turbine']['device_observations']['available_wind_power']) * self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][0]['device']['const_params']['alpha_g']['value']
        controllable_fuel_cons = controllable_fuel_cons /  self.max_genset_group_fuel_consumption

        ## Configuration of the genset group as the reward
        configuration = 0
        for idx in range(len(new_state_dict['genset_group']['device_observations']['gensets'])):
            if new_state_dict['genset_group']['device_observations']['gensets'][idx]['device_observations']['running']:
                configuration += 1

        ## Putting performance rewards together (this is designed so that one or the other reward is used, so one coefficient should be 0. Does not HAVE to be the case though.)
        rew_fuel = self.reward_params['fuel_consumption_coeff']*fuel_cons + self.reward_params['controllable_fuel_cons_coeff']*controllable_fuel_cons + self.reward_params['configuration_coeff']*configuration
        reward_components.update({'rew_fuel': rew_fuel})

        if verbose:
            print("Fuel reward elements: fuel: {}, controllable fuel: {}".format(fuel_cons, controllable_fuel_cons))

        # Maintenance:
        
        ## Battery degradation cost
        battery_degradation = new_state_dict['battery']['device_observations']['degradation_cost']

        ## Starting to charge the battery gets a penalty
        if prev_state_dict['battery']['device_observations']['p_grid'] >= 0 and new_state_dict['battery']['device_observations']['p_grid'] < 0:       # Starting to charge the battery
            battery_startcharge = 1
        else:
            battery_startcharge = 0

        ## Varying the battery charge/discharge power gets a penalty, between 0 and 1
        battery_variation = np.abs(prev_state_dict['battery']['device_observations']['p_grid'] - new_state_dict['battery']['device_observations']['p_grid'])/(2*self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])

        ## Starting a genset (changing configuration) gets a penalty
        if self.check_starting_genset(prev_state_dict['genset_group'], new_state_dict['genset_group']):
            genset_start = 1
        else:
            genset_start = 0

        ## Running any genset in overload gets a penalty, boolean of 0 or 1
        genset_overload = 0
        for idx in range(len(new_state_dict['genset_group']['device_observations']['gensets'])):
            if new_state_dict['genset_group']['device_observations']['gensets'][idx]['device_observations']['overload'] == 1:
                genset_overload = 1
        

        ## Putting maintenance rewards together:
        rew_maintenance = self.reward_params['battery_degradation_coeff']*battery_degradation + self.reward_params['battery_startcharge_coeff']*battery_startcharge + self.reward_params['battery_variation_coeff']*battery_variation + self.reward_params['genset_start_coeff']*genset_start + self.reward_params['genset_overload_coeff']*genset_overload
        
        reward_components.update({'rew_maintenance': rew_maintenance, 'battery_degradation': battery_degradation,'battery_startcharge': battery_startcharge, 'battery_variation': battery_variation, 'genset_start': genset_start, 'genset_overload': genset_overload})

        if verbose:
            print("Maintenance reward elements: battery start charge: {}, battery variation: {}, genset start: {}, genset overload: {}".format(battery_startcharge, battery_variation, genset_start, genset_overload))

        # Constraints:

        ## Penalty for the demand being higher than the produced power (balance < 0), between 0 and 1 (1 being when the balance is the max power of the gensets + batteries)
        if new_state_dict['balance'] < -0.00000001:
            neg_balance = 0.1 - new_state_dict['balance']/(self.max_genset_group_active_power + self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])
        else:
            neg_balance = 0

        ## Penalty for the demand being lower than the produced power (balance > 0), between 0 and 1 (1 being when the extra power is the max power of the gensets + batteries)
        if new_state_dict['balance'] > 0.00000001:
            pos_balance = 0.1 + new_state_dict['balance']/(self.max_genset_group_active_power + self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])
        else:
            pos_balance = 0

        ## Penalty for using the shield on the battery, between 0 and 1 (1 being when the shield changes the action from one extreme to the other of maximal battery power)
        if new_state_dict['action_command'] and new_state_dict['action_implemented']:
            battery_shield = np.abs(new_state_dict['action_implemented']['battery']['p_grid'] - new_state_dict['action_command']['battery']['p_grid'])/(2*self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])
        else:
            battery_shield = 0

        ## Penalty for using the shield on the gensets and forcing a status change
        if new_state_dict['action_command'] and new_state_dict['action_implemented'] and new_state_dict['action_command']['genset_group']['status_change'] != new_state_dict['action_implemented']['genset_group']['status_change']:
            genset_shield = 1 
        else:
            genset_shield = 0

        ## Putting constraint rewards together
        rew_shield =  self.reward_params['battery_shield_coeff']*battery_shield + self.reward_params['genset_shield_coeff']*genset_shield
        rew_balance = self.reward_params['pos_balance_coeff']*pos_balance + self.reward_params['neg_balance_coeff']*neg_balance
        reward_components.update({'rew_shield': rew_shield, 'rew_balance': rew_balance, 'neg_balance': neg_balance, 'pos_balance': pos_balance, 'battery_shield': battery_shield, 'genset_shield': genset_shield})

        
        # Total computation
        reward = - self.reward_params['include_fuel_reward']*rew_fuel - self.reward_params['include_maintenance_reward']*rew_maintenance - self.reward_params['include_shield_reward']*rew_shield  - self.reward_params['include_balance_reward']*rew_balance

        reward_components.update({'reward': reward})
        if verbose:
            print("Constraint reward elements: negative balance: {}, positive balance: {}, battery shield: {}, genset shield: {}".format(neg_balance, pos_balance, battery_shield, genset_shield))

        return reward, reward_components



    def obs_dict_to_arrays(self, obs_dict):
        """
        Converts the observation dictionary to two normalized array for the agent: structured data and time series
        Input:
        - norm_obs_dict: dictionary with the following keys
            - genset_group: genset group observations:
                - genset_group_active_power: float, normalized from -1 to 1
                - genset_group_fuel_consumption: float, normalized from -1 to 1
                - gensets: dictionary of genset with ids (int) as keys 
                    - ids : dictionary with the following keys for each genset:
                        - running: -1 or 1
                        - warmup: -1 or 1
                        - cooldown: -1 or 1
                        - time_since_warmup: float, normalized from -1 to 1
                        - time_since_cooldown: float, normalized from -1 to 1
                        - time_since_start: float, normalized from -1 to 1
                        - active_power: float, normalized from -1 to 1
                        - available_power: float, normalized from -1 to 1
                        - average_active_power: float, normalized from -1 to 1
                        - fuel_consumption: float, normalized from -1 to 1
            - battery: dictionary with the following keys:
                - soc: state of charge: float, normalized from -1 to 1
                - p_grid: power charging/discharging the battery on the grid side: float, normalized from -1 to 1
                - degradation_cost: float, normalized from 0 to 1
                - soc_tp_buffer: state of charge turning points: list, normalized from 0 to 1
            - wind_turbine: wind turbine observations:
                - available_wind_power: float, in kW
                - wind_power: float, in kW
                - available_wind_power_pred: list of floats, in kW
            - demand: dictionary with the following keys:
                - demand: float, normalized from -1 to 1
                - demand_pred: predicted demand, list of floats, normalized from -1 to 1
            - microgrid: dictionary with the following keys:
                - balance: float, normalized from -1 to 1
                - action_command: dictionary with the following keys
                    - genset_group: dictionary with the following keys
                        - status_change: -1, 0, 1
                    - battery: dictionary with the following keys
                        - p_grid: float, normalized from -1 to 1
                - action_implemented: dictionary with the following keys
                    - genset_group: dictionary with the following keys
                        - status_change: -1, 0, 1
                        - power_setpoint: normalized, from -1 to 1
                    - battery: dictionary with the following keys
                        - p_grid: float, normalized from -1 to 1   
                    - wind_turbine: dictionary with the following keys
                        - turbine_setpoint: float, normalized from -1 to 1 
        Outputs:
        - obs_norm_array_dict: dictionary with the following structure:
            - obs_array: numpy array with the following structure:
                - genset_group_active_power: float, normalized from -1 to 1
                - genset_group_fuel_consumption: float, normalized from -1 to 1
                - gensets: for each ID, the following 
                    - running: -1 or 1
                    - warmup: -1 or 1
                    - cooldown: -1 or 1
                    - overload: -1 or 1
                    - time_since_warmup: float, normalized from -1 to 1
                    - time_since_cooldown: float, normalized from -1 to 1
                    - time_since_start: float, normalized from -1 to 1
                    - active_power: float, normalized from -1 to 1
                    - available_power: float, normalized from -1 to 1
                    - average_active_power: float, normalized from -1 to 1
                    - fuel_consumption: float, normalized from -1 to 1
                - battery/soc: state of charge: float, normalized from -1 to 1
                - battery/p_grid: power charging/discharging the battery on the grid side: float, normalized from -1 to 1
                - wind_turbine/available_wind_power: float, normalized from -1 to 1
                - wind_turbine/wind_power: float, normalized from -1 to 1
                - demand: float, normalized from -1 to 1
                - balance: float, normalized from -1 to 1
                - action_command_genset_group_status_change: -1, 0, 1
                - action_command_battery_p_grid: float, normalized from -1 to 1
                - action_implemented_genset_group_status_change: -1, 0, 1
                - action_implemented_genset_group_power_setpoint: normalized, from -1 to 1
                - action_implemented_battery_p_grid: float, normalized from -1 to 1
                - action_implemented_wind_turbine_turbine_setpoint: float, normalized from -1 to 1
            - pred_array: numpy array with the predicted demand, float, normalized from -1 to 1
        
        """

        norm_obs_dict = self.normalize_obs_dict(obs_dict)
        obs_array = np.array([])

    
        # Genset group observations
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['genset_group_active_power'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['genset_group_fuel_consumption'])
        for idx in range(len(norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'])):
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['running'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['warmup'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['cooldown'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['device_observations']['overload'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['time_since_warmup'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['time_since_cooldown'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['time_since_start'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['device_observations']['active_power'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['available_power'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['average_active_power'])
            obs_array = np.append(obs_array, norm_obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['device_observations']['fuel_consumption'])
        
        # Battery observations
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['battery']['device_observations']['soc'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['battery']['device_observations']['p_grid'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['battery']['device_observations']['degradation_cost'])

        # Wind turbine observations
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['wind_turbine']['device_observations']['available_wind_power'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['wind_turbine']['device_observations']['wind_power'])
        
        # Demand observations
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['demand']['device_observations']['demand'])

        # Microgrid observations
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['balance'])

        # Action observations
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['action_command']['genset_group']['status_change'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['action_command']['battery']['p_grid'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['action_implemented']['genset_group']['status_change'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['action_implemented']['genset_group']['power_setpoint'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['action_implemented']['battery']['p_grid'])
        obs_array = np.append(obs_array, norm_obs_dict['device_observations']['action_implemented']['wind_turbine']['turbine_setpoint'])

        demand_pred_array = norm_obs_dict['device_observations']['demand']['device_observations']['demand_pred']
        available_wind_power_pred_array = norm_obs_dict['device_observations']['wind_turbine']['device_observations']['available_wind_power_pred']
        soc_tp_buffer = norm_obs_dict['device_observations']['battery']['device_observations']['soc_tp_buffer']

        obs_norm_array_dict = {
            'obs_array': obs_array,
            'demand_pred_array': demand_pred_array,
            'available_wind_power_pred_array': available_wind_power_pred_array,
            'soc_tp_buffer': soc_tp_buffer
        }

        return obs_norm_array_dict

    def normalize_obs_dict(self, obs_dict):
        """
        Normalizes the observation dictionary
        Inputs:
        - obs_dict: dictionary with the following keys
            - genset_group: genset group observations:
                - genset_group_active_power: in kW
                - genset_group_fuel_consumption: in l/h
                - gensets: dictionary of genset with ids (int) as keys 
                    - ids : dictionary with the following keys for each genset:
                        - running: Boolean
                        - warmup: Boolean
                        - cooldown: Boolean
                        - overload: Boolean
                        - time_since_warmup: in minutes
                        - time_since_cooldown: in minutes
                        - time_since_start: in minutes
                        - active_power: in kW
                        - available_power: in kW
                        - average_active_power: in kW
                        - fuel_consumption: in l/h
            - battery: battery observations:
                - soc: state of charge: float, in ratio from 0 to 1)
                - p_grid: power charging/discharging the battery on the grid side: float, in kW (positive for discharging, negative for charging)
                - degradation_cost: float, in ratio from 0 to 1
                - soc_tp_buffer: state of charge turning points: list of floats, in ratio from 0 to 1
            - demand: demand observations:
                - demand: float, in kW
                - demand_pred: predicted demand, list of floats, in kW
            - wind_turbine: wind turbine observations:
                - available_wind_power: float, in kW
                - wind_power: float, in kW
                - available_wind_power_pred: list of floats, in kW
            - microgrid: dictionary with the following keys:
                - balance: float, normalized from -1 to 1 
                - action_command: dictionary with the following keys
                    - genset_group: dictionary with the following keys
                        - status_change: string: 'stop_last', 'none', 'start_next'
                    - battery: dictionary with the following keys
                        - p_grid: float, in kW
                - action_implemented: dictionary with the following keys
                    - genset_group: dictionary with the following keys
                        - status_change: string: 'stop_last', 'none', 'start_next'
                        - power_setpoint: in kW
                    - battery: dictionary with the following keys
                        - p_grid: float, in kW       
                    - wind_turbine: dictionary with the following keys
                        - turbine_setpoint: in kW  
        Outputs:
        - norm_obs_dict: dictionary with the following keys
            - genset_group: genset group observations:
                - genset_group_active_power: float, normalized from -1 to 1
                - genset_group_fuel_consumption: float, normalized from -1 to 1
                - gensets: dictionary of genset with ids (int) as keys 
                    - ids : dictionary with the following keys for each genset:
                        - running: -1 or 1
                        - warmup: -1 or 1
                        - cooldown: -1 or 1
                        - time_since_warmup: float, normalized from -1 to 1
                        - time_since_cooldown: float, normalized from -1 to 1
                        - time_since_start: float, normalized from -1 to 1
                        - active_power: float, normalized from -1 to 1
                        - available_power: float, normalized from -1 to 1
                        - average_active_power: float, normalized from -1 to 1
                        - fuel_consumption: float, normalized from -1 to 1
            - battery: dictionary with the following keys:
                - soc: state of charge: float, normalized from -1 to 1
                - p_grid: power charging/discharging the battery on the grid side: float, normalized from -1 to 1
            - wind_turbine: wind turbine observations:
                - available_wind_power: float, normalized from -1 to 1
                - wind_power: float, normalized from -1 to 1
            - demand: dictionary with the following keys:
                - demand: float, normalized from -1 to 1
                - demand_pred: predicted demand, list of floats, normalized from -1 to 1
            - microgrid: dictionary with the following keys:
                - balance: float, normalized from -1 to 1
                - action_command: dictionary with the following keys
                    - genset_group: dictionary with the following keys
                        - status_change: -1, 0, 1
                    - battery: dictionary with the following keys
                        - p_grid: float, normalized from -1 to 1
                - action_implemented: dictionary with the following keys
                    - genset_group: dictionary with the following keys
                        - status_change: -1, 0, 1
                        - power_setpoint: normalized, from -1 to 1
                    - battery: dictionary with the following keys
                        - p_grid: float, normalized from -1 to 1 
                    - wind_turbine: dictionary with the following keys
                        - turbine_setpoint: float, normalized from -1 to 1      
        """

        nb_gensets = len(obs_dict['device_observations']['genset_group']['device_observations']['gensets'])

        # Genset group observations
        genset_group_obs = obs_dict['device_observations']['genset_group']
        norm_genset_group_obs = {'device_observations': {}, 'controller_state': {}}

        ## Group observations
        norm_genset_group_obs['device_observations']['genset_group_active_power'] = 2*genset_group_obs['device_observations']['genset_group_active_power']/self.max_genset_group_active_power - 1                 # Normalized between -1 and 1
        norm_genset_group_obs['device_observations']['genset_group_fuel_consumption'] = 2*genset_group_obs['device_observations']['genset_group_fuel_consumption']/self.max_genset_group_fuel_consumption - 1     # Normalized between -1 and 1
        norm_genset_group_obs['device_observations']['gensets'] = {}

        ## Genset observations
        norm_genset_group_obs['device_observations']['gensets'] = [None]*nb_gensets
        for idx in range(nb_gensets):
            genset_obs = genset_group_obs['device_observations']['gensets'][idx]
            const_params = self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['device']['const_params']
            const_control_params = self.env_params['microgrid']['device']['init_params']['genset_group']['device']['init_params']['gensets'][idx]['controller']['const_params']
            max_power = const_params['prime_power_rating']['value']*const_params['temp_overload_factor']['value']
            max_fuel_consumption = const_params['alpha_g']['value']*const_params['prime_power_rating']['value']*const_params['temp_overload_factor']['value'] + const_params['beta_g']['value']
            norm_genset_obs = {'device_observations': {}, 'controller_state': {}}
            norm_genset_obs['controller_state']['running'] = 1 if genset_obs['controller_state']['status'] in ['warmup', 'cooldown', 'running'] else -1
            norm_genset_obs['controller_state']['warmup'] = 1 if genset_obs['controller_state']['status'] == 'warmup' else -1
            norm_genset_obs['controller_state']['cooldown'] = 1 if genset_obs['controller_state']['status'] == 'cooldown' else -1
            norm_genset_obs['device_observations']['overload'] = 1 if genset_obs['device_observations']['overload'] else -1
            norm_genset_obs['controller_state']['time_since_warmup'] = 2*float(genset_obs['controller_state']['time_since_warmup'])/float(const_control_params['warmup_time']['value']) - 1
            norm_genset_obs['controller_state']['time_since_cooldown'] = 2*float(genset_obs['controller_state']['time_since_cooldown'])/float(const_control_params['cooldown_time']['value']) - 1
            norm_genset_obs['controller_state']['time_since_start'] = 2*float(genset_obs['controller_state']['time_since_start'])/float(const_control_params['minimum_run_time']['value']) - 1
            norm_genset_obs['device_observations']['active_power'] = 2*float(genset_obs['device_observations']['active_power'])/max_power - 1
            norm_genset_obs['controller_state']['available_power'] = 2*float(genset_obs['controller_state']['available_power'])/max_power - 1
            norm_genset_obs['controller_state']['average_active_power'] = 2*float(genset_obs['controller_state']['average_active_power'])/max_power - 1
            norm_genset_obs['device_observations']['fuel_consumption'] = 2*genset_obs['device_observations']['fuel_consumption']/max_fuel_consumption - 1
            norm_genset_group_obs['device_observations']['gensets'][idx] = norm_genset_obs

        # Battery observations
        battery_obs = obs_dict['device_observations']['battery']
        norm_battery_obs = {'device_observations': {}, 'controller_state': {}}
        norm_battery_obs['device_observations']['soc'] = 2*battery_obs['device_observations']['soc'] - 1          # Normalized between -1 and 1
        norm_battery_obs['device_observations']['p_grid'] = battery_obs['device_observations']['p_grid']/(self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])    # Normalized between -1 and 1
        norm_battery_obs['device_observations']['degradation_cost'] = battery_obs['device_observations']['degradation_cost']
        norm_battery_obs['device_observations']['soc_tp_buffer'] = battery_obs['device_observations']['soc_tp_buffer']

        # Wind turbine observations
        wind_turbine_obs = obs_dict['device_observations']['wind_turbine']
        norm_wind_turbine_obs = {'device_observations': {}, 'controller_state': {}}
        if 'wind_turbine' not in self.env_params['microgrid']['device']['init_params']:
            norm_wind_turbine_obs['device_observations']['available_wind_power'] = 0
            norm_wind_turbine_obs['device_observations']['wind_power'] = 0
            norm_wind_turbine_obs['device_observations']['available_wind_power_pred'] = np.zeros((1,10))
        elif self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['mode']['value'] == 'perlin':
            norm_wind_turbine_obs['device_observations']['available_wind_power'] = 2 * wind_turbine_obs['device_observations']['available_wind_power']/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1  # Normalized between -1 and 1
            norm_wind_turbine_obs['device_observations']['wind_power'] = 2 * wind_turbine_obs['device_observations']['wind_power']/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1  # Normalized between -1 and 1
            norm_wind_turbine_obs['device_observations']['available_wind_power_pred'] = 2*np.array([wind_turbine_obs['device_observations']['available_wind_power_pred']])/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1  # Normalized between -1 and 1
        elif self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['mode']['value'] == 'data':
            norm_wind_turbine_obs['device_observations']['available_wind_power'] = 2 * wind_turbine_obs['device_observations']['available_wind_power']/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1  # Normalized between -1 and 1
            norm_wind_turbine_obs['device_observations']['wind_power'] = 2 * wind_turbine_obs['device_observations']['wind_power']/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1  # Normalized between -1 and 1
            norm_wind_turbine_obs['device_observations']['available_wind_power_pred'] = 2*np.array([wind_turbine_obs['device_observations']['available_wind_power_pred']])/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1  # Normalized between -1 and 1
        else:
            raise NotImplementedError('Wind turbine mode {} not implemented. '.format(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['mode']['value']) + 'Available modes: perlin, data')


        # Demand observations
        demand_obs = obs_dict['device_observations']['demand']
        norm_demand_obs = {'device_observations': {}, 'controller_state': {}}
        norm_demand_obs['device_observations']['demand'] = 2*demand_obs['device_observations']['demand']/(self.max_genset_group_active_power + self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value']) - 1           # Based on the hypothesis that the demand could not be higher than the max power of the gensets + batteries
        norm_demand_obs['device_observations']['demand_pred'] = 2*np.array([demand_obs['device_observations']['demand_pred']])/(self.max_genset_group_active_power + self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value']) - 1

        # Microgrid
        norm_balance = obs_dict['device_observations']['balance']/(self.max_genset_group_active_power + self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])

        ## Actions
        action_translation = {'stop_last': -1, 'none': 0, 'start_next': 1}

        action_command = obs_dict['device_observations']['action_command']
        norm_action_command = {}
        norm_action_command['genset_group'] = {}
        norm_action_command['genset_group']['status_change'] = action_translation[action_command['genset_group']['status_change']]                  # -1, 0, 1
        norm_action_command['battery'] = {}
        norm_action_command['battery']['p_grid'] = action_command['battery']['p_grid']/(self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])       # Normalized between -1 and 1

        action_implemented = obs_dict['device_observations']['action_implemented']
        norm_action_implemented = {}
        norm_action_implemented['genset_group'] = {}
        norm_action_implemented['genset_group']['status_change'] = action_translation[action_command['genset_group']['status_change']]                  # -1, 0, 1
        norm_action_implemented['genset_group']['power_setpoint'] = 2*action_implemented['genset_group']['power_setpoint']/(self.max_genset_group_active_power) - 1      # Normalized between -1 and 1
        norm_action_implemented['battery'] = {}
        norm_action_implemented['battery']['p_grid'] = action_implemented['battery']['p_grid']/(self.env_params['microgrid']['device']['init_params']['battery']['device']['const_params']['P_nom']['value'])       # Normalized between -1 and 1
        norm_action_implemented['wind_turbine'] = {}
        if 'wind_turbine' not in self.env_params['microgrid']['device']['init_params']:
            norm_action_implemented['wind_turbine']['turbine_setpoint'] = 0
        else:
            norm_action_implemented['wind_turbine']['turbine_setpoint'] = 2*action_implemented['wind_turbine']['turbine_setpoint']/(self.env_params['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['nominal_power']['value']) - 1      # Normalized between -1 and 1



        # Final dictionary
        norm_microgrid_obs = {
            'device_observations': {
                'genset_group': norm_genset_group_obs,
                'battery': norm_battery_obs,
                'demand': norm_demand_obs,
                'wind_turbine': norm_wind_turbine_obs,
                'balance' : norm_balance,
                'action_command': norm_action_command,
                'action_implemented': norm_action_implemented,
            },
            'constroller_state': {},
        }
        return norm_microgrid_obs
    
    def prune_obs_dict(self, obs_dict):
        """
        Only select the observable elements of the observation dictionary
        """

        gensets_obs = []
        for idx in range(len(obs_dict['device_observations']['genset_group']['device_observations']['gensets'])):
            gensets_obs.append({
                'overload': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['device_observations']['overload'],
                'active_power': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['device_observations']['active_power'],
                'fuel_consumption': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['device_observations']['fuel_consumption'],
                'running': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['status'] in ['warmup', 'cooldown', 'running'],
                'warmup': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['status'] == 'warmup',
                'cooldown': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['status'] == 'cooldown',
                'time_since_warmup': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['time_since_warmup'],
                'time_since_cooldown': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['time_since_cooldown'],
                'time_since_start': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['time_since_start'],
                'average_active_power': obs_dict['device_observations']['genset_group']['device_observations']['gensets'][idx]['controller_state']['average_active_power'],
            })


        new_obs_dict = {
            'genset_group': {
                'genset_group_active_power': obs_dict['device_observations']['genset_group']['device_observations']['genset_group_active_power'],
                'genset_group_fuel_consumption': obs_dict['device_observations']['genset_group']['device_observations']['genset_group_fuel_consumption'],
                'genset_group_available_power': obs_dict['device_observations']['genset_group']['device_observations']['genset_group_available_power'],
                'gensets': {
                    idx: self.decompose_genset_obs_dict(genset_dict)
                    for idx, genset_dict in obs_dict['device_observations']['genset_group']['device_observations']['gensets'].items()
                }            
            },
            'battery': {
                'soc': obs_dict['device_observations']['battery']['device_observations']['soc'],
                'p_grid': obs_dict['device_observations']['battery']['device_observations']['p_grid'],
                'degradation_cost': obs_dict['device_observations']['battery']['device_observations']['degradation_cost'],
                'soc_tp_buffer': obs_dict['device_observations']['battery']['device_observations']['soc_tp_buffer']
            },
            'demand': {
                'demand': obs_dict['device_observations']['demand']['device_observations']['demand'],
                'demand_next': obs_dict['device_observations']['demand']['device_observations']['demand_next'],
                'demand_pred': np.array(obs_dict['device_observations']['demand']['device_observations']['demand_pred']),
            },
            'microgrid': {
                'action_command': 
                 {
                    'genset_group': obs_dict['device_observations']['action_command']['genset_group'],
                    'battery': obs_dict['device_observations']['action_command']['battery']
                 },
                'action_implemented': 
                {
                    'genset_group': obs_dict['device_observations']['action_implemented']['genset_group'],
                    'battery': obs_dict['device_observations']['action_implemented']['battery'],
                    'wind_turbine': {
                        'turbine_setpoint': obs_dict['device_observations']['action_implemented']['wind_turbine']['turbine_setpoint']
                        }
                 },
                'balance': obs_dict['device_observations']['balance']
            },
            'wind_turbine': {
                'available_wind_power': obs_dict['device_observations']['wind_turbine']['device_observations']['available_wind_power'],
                'wind_power': obs_dict['device_observations']['wind_turbine']['device_observations']['wind_power'],
                'available_wind_power_pred': np.array(obs_dict['device_observations']['wind_turbine']['device_observations']['available_wind_power_pred'], dtype=float)
            },
        }

        return new_obs_dict
    
    def decompose_genset_obs_dict(self, genset_obs_dict):

        device_obs = genset_obs_dict['device_observations']
        controller_state = genset_obs_dict['controller_state']

        obs = {
            'overload': device_obs['overload'],
            'active_power': device_obs['active_power'],
            'available_power': controller_state['available_power'],
            'fuel_consumption': device_obs['fuel_consumption'],
            'running': controller_state['status'] in ['warmup', 'cooldown', 'running'],
            'warmup': controller_state['status'] == 'warmup',
            'cooldown': controller_state['status'] == 'cooldown',
            'time_since_warmup': controller_state['time_since_warmup'],
            'time_since_cooldown': controller_state['time_since_cooldown'],
            'time_since_start': controller_state['time_since_start'],
            'average_active_power': controller_state['average_active_power'],
        }

        return obs
        
    def check_starting_genset(self, old_group_state, new_group_state):
        """
        Checks if the genset configuration was ordered (by action or shield) and applied
        Inputs:

        """
        if set(old_group_state['device_observations']['config_lists']['running_gensets_ids'] + old_group_state['device_observations']['config_lists']['warmup_gensets_ids']) < set(new_group_state['device_observations']['config_lists']['running_gensets_ids'] + new_group_state['device_observations']['config_lists']['warmup_gensets_ids']):        # If the gensets running or warming up are bigger now than before, then a genset was turned on or off
            return True
        
        else:
            return False

