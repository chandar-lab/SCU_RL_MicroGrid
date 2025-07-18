import numpy as np
import pprint
import copy
try:
    from .utils import add_params_noise
    from env.abstract_classes import System

except:
    from utils import add_params_noise
    from abstract_classes import System


class Genset(System):
    """
    Simulate a fuel generator
    """
    def __init__(self, params, time_step, real):
        """
        Inputs:
            - params: dictionary with the following keys:
            - const_params: dictionary with the following keys:
                - alpha_g: fuel consumption rate in l/h.kW
                - beta_g: fuel consumption rate at no load in l/h
                - prime_power_rating: in kW
                - temp_overload_factor: ratio (1.1 for 10% overload)
                - noise_level: string, level of noise to add ('None', 'low', 'medium', 'high')
                - noise_range_percentage: percentage range for noise
            - init_params: dictionary with the following keys:
                - running: Boolean
                - init_power_setpoint: in kW
        """
        const_params = params['const_params']
        init_params = params['init_params']

        self.time_step = time_step

        # Randomize parameters for the device
        self.real = real
        self.isDevice = True

        if self.real and const_params['noise_level']['value'] not in ['None', 'none']:
            noise_level = const_params['noise_level']['value']
            print(f'Adding {noise_level} noise to Genset device')
            for param in ["prime_power_rating"]:
                print(f'Original {param}: {const_params[param]["value"]}')
                const_params[param]['value'] = add_params_noise(const_params[param]['value'], const_params['noise_level']['value'], const_params['noise_range_percentage']['value'])
                print(f'Noisy {param}: {const_params[param]["value"]}')


        # Fixed parameters
        self.alpha_g = const_params['alpha_g']['value']                             # in l/h.kW   
        self.beta_g = const_params['beta_g']['value']                               # in l/h
        self.prime_power_rating = const_params['prime_power_rating']['value']       # in kW
        self.temp_overload_factor = const_params['temp_overload_factor']['value']   # ratio (1.1 for 10% overload)
        self.available_power = np.round(self.prime_power_rating * self.temp_overload_factor, 2) # in kW

        # Initial params
        self.reset(init_params, real)


    def step(self, action):
        """
        Inputs:
            - action: dictionary with the following keys:
                - status_change: string, 'start', 'none', or 'stop'
                - power_setpoint: float, in kW   
        """
        status_change = action['status_change']
        power_setpoint = action['power_setpoint']
        self.update_state_machine(status_change)
        self.update_active_power(power_setpoint)
        self.update_fuel_consumption()

        #obs = self.gather_observations()

        #return obs

    def update_fuel_consumption(self):
        # Updates the fuel consumption, in l/h
        if self.running:
            self.fuel_consumption = self.alpha_g * self.active_power + self.beta_g
        else:
            self.fuel_consumption = 0
       
    def reset(self, init_params, real):
        """
        Resets the genset to the initial state
        Inputs:
            - init_params: dictionary with the following keys:
                - running: Boolean
                - warmup: Boolean
                - cooldown: Boolean
                - time_since_warmup: in minutes
                - time_since_cooldown: in minutes
                - time_since_start: in minutes
                - active_power: in kW
                - available_power: in kW
                - average_active_power: in kW
        """
        self.real = real
        self.running = init_params['running']['value']                                                  # Boolean
        init_power_setpoint = init_params['init_power_setpoint']['value']  

        self.available_power = np.round(self.prime_power_rating * self.temp_overload_factor, 2)
        self.update_active_power(init_power_setpoint)
        self.update_fuel_consumption()

        obs = self.gather_observations()

        return obs
      
    def update_active_power(self, power_setpoint):
        """
        Updates the active power based on the state and the power setpoint
        """
        ## Update active power
        if self.running:
            self.active_power = np.clip(power_setpoint, 0, self.available_power)
        else:
            self.active_power = 0

        if self.active_power > self.prime_power_rating:
            self.overload = True
        else:
            self.overload = False


    def update_state_machine(self, status_change):
        # Updates the state machine based on the action
        if status_change == 'start':
            self.running = True 
        elif status_change == 'stop':
            self.running = False
        elif status_change == 'none':
            pass
        else:
            raise ValueError(f'{status_change} is not defined. status change should be among start, stop, and none')

    
    def gather_observations(self):
        # Returns the state of the genset

        state = {
            'running': self.running,
            'overload': self.overload,
            'active_power': self.active_power,
            'fuel_consumption': self.fuel_consumption,
            'obs_type': 'state'
        }

        return state

    def simulator_sensor_update(self, device_observations):
        # Only called if this is a simulator. Updates the state of the simulator based on real device observations.

        if self.real:
            raise ValueError("This function should only be called if the genset is a simulator")

        if device_observations['obs_type'] == 'state':
            self.running = device_observations['running']
            self.overload = device_observations['overload']
            self.active_power = device_observations['active_power']
            self.fuel_consumption = device_observations['fuel_consumption']
        elif device_observations['obs_type'] == 'params':
            self.running = device_observations['running']['value']
            self.overload = device_observations['overload']['value']
            self.active_power = device_observations['active_power']['value']
            self.fuel_consumption = device_observations['fuel_consumption']['value']
        else:
            raise ValueError("The observation type should be either state or params")


    def print_state(self):
        print(
            " Running: {}\n Active power: {}\n Available active power: {}\n Fuel consumption: {}".format(
                self.running*1, self.active_power, self.available_power,  self.fuel_consumption
            )
        )

