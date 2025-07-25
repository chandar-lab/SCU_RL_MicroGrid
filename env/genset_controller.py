import numpy as np
import copy

try:

    from env.abstract_classes import Controller
    from env.genset import Genset
    from .utils import set_init_param
except:
    from abstract_classes import Controller
    from genset import Genset
    from utils import set_init_param


class GensetController(Controller):
    def __init__(self, genset_params, time_step):
        """
        Inputs:
            genset_params: dictionary with the following keys:
                - device: dictionary with the following keys
                    - const_params: dictionary with the following keys
                        - alpha_g: fuel consumption rate in l/h.kW
                        - beta_g: fuel consumption rate at no load in l/h
                        - prime_power_rating: in kW
                        - temp_overload_factor: ratio (1.1 for 10% overload)
                        - noise_level: string, level of noise to add ('None', 'low', 'medium', 'high')
                        - noise_range_percentage: percentage range for noise
                    - init_params: dictionary with the following keys (some of them can also be "random")
                        - device_running: Boolean
                        - init_power_setpoint: in kW
                - controller: dictionary with the following keys
                    - const_params: dictionary with the following keys:
                        - average_active_power_constraint: Boolean, True if the average active power constraint is active, False otherwise
                        - average_active_power_window: in minutes, the window over which the average active power is calculated
                        - average_active_power_limit: ratio of the prime power rating for the maximum average active power
                        - minimum_load: in kW, the minimum load of the genset
                        - warmup_power: in kW, the power during warmup
                        - cooldown_power: in kW, the power during cooldown
                        - minimum_run_time: in minutes, the minimum run time of the genset
                        - warmup_time: in minutes, the warmup time of the genset
                        - cooldown_time: in minutes, the cooldown time of the genset
                    - init_params: dictionary with the following keys (some of them can also be "random")
                        - status: string, 'running', 'warmup', 'cooldown', or 'off'
                        - time_since_warmup: in minutes, the time since the genset started warming up
                        - time_since_cooldown: in minutes, the time since the genset started cooling down
                        - time_since_start: in minutes, the time since the genset started running
                        - average_active_power: in kW, the average active power of the genset over the last 24 h of running
            device_running: Boolean, read from the true device. True if the genset is running, False otherwise
        """

        genset_controller_const_params = genset_params['controller']['const_params']
        self.time_step = time_step

        # Initializing the simulator
        self.genset_sim = Genset(genset_params['device'], self.time_step, real = False)

        self.average_active_power_constraint = genset_controller_const_params['average_active_power_constraint']['value']  # Boolean
        self.average_active_power_window = genset_controller_const_params['average_active_power_window']['value']          # in minutes
        self.average_active_power_limit = genset_controller_const_params['average_active_power_limit']['value']       # ratio of the prime power rating
        self.minimum_load = genset_controller_const_params['minimum_load']['value']                   # in kW
        self.warmup_power = genset_controller_const_params['warmup_power']['value']                   # in kW
        self.cooldown_power = genset_controller_const_params['cooldown_power']['value']               # in kW
        self.minimum_run_time = genset_controller_const_params['minimum_run_time']['value']           # in minutes
        self.warmup_time = genset_controller_const_params['warmup_time']['value']                     # in minutes
        self.cooldown_time = genset_controller_const_params['cooldown_time']['value']                 # in minutes

        self.reset(genset_params['controller']['init_params'], genset_params['device']['init_params'])  # Resetting the controller and the simulator

    def reset(self, genset_controller_init_params, genset_device_init_params):
        """
        Resets the controller and its simulator based on the initial parameters.
        """

        self.status = genset_controller_init_params['status']['value'] 
        self.time_since_warmup = genset_controller_init_params['time_since_warmup']['value']                          
        self.time_since_cooldown = genset_controller_init_params['time_since_cooldown']['value']   
        self.time_since_start = genset_controller_init_params['time_since_start']['value']   
        self.average_active_power = genset_controller_init_params['average_active_power']['value']
        self.average_active_power = np.clip(self.average_active_power, 0, 0.7*self.genset_sim.prime_power_rating)
        
        self.controllable_status = self.check_controllable_status()
        self.controllable_power = self.check_controllable_power()  # Check if the genset is controllable (i.e. it is not in cooldown or warmup)
        self.available_power = self.update_available_power(reserve_available=True)  # Update the available power based on the state and the power setpoint

        ## Resetting the simulator
        self.genset_sim.reset(genset_device_init_params, real = False)  # Reset the simulator 

    def update_controller_state(self, observations):
        """
        Updates the simulator based on the observation from the true genset.

        Inputs:
            device_observations is a dictionary with the following keys:
                status: string, 'running', 'warmup', 'cooldown', or 'off'
                time_since_warmup: Time elapsed since warmup in minutes
                time_since_cooldown: Time elapsed since cooldown in minutes
                time_since_start: Time elapsed since the genset started in minutes
                average_active_power: Average active power of the genset in kW
        """
        device_observations = observations['device_observations']
        # update simulator state
        self.genset_sim.simulator_sensor_update(device_observations)  # Update the simulator with the observations from the true genset
        
        if 'controller_state' in observations:              # If the observations come from another controller, we update the controller state
            controller_state = observations['controller_state']
            # update controller parameters
            if controller_state['obs_type'] == 'state':
                self.status = controller_state['status']
                self.time_since_cooldown = controller_state['time_since_cooldown']
                self.time_since_warmup = controller_state['time_since_warmup']
                self.time_since_start = controller_state['time_since_start']
                self.average_active_power = controller_state['average_active_power']
                self.controllable_status = controller_state['controllable_status']
                self.controllable_power = controller_state['controllable_power']
                self.available_power = controller_state['available_power']            
            elif controller_state['obs_type'] == 'params':
                self.status = controller_state['status']['value']
                self.time_since_cooldown = controller_state['time_since_cooldown']['value']
                self.time_since_warmup = controller_state['time_since_warmup']['value']
                self.time_since_start = controller_state['time_since_start']['value']
                self.average_active_power = controller_state['average_active_power']['value']
                self.controllable_status = self.check_controllable_status()
                self.controllable_power = self.check_controllable_power()
                self.available_power = self.update_available_power(reserve_available=True)
                
    def simulator_dynamic_update(self, valid_action, reserve_available):
        # Update the genset simulator based on the action taken by the controller
        valid_action = self.generate_safe_action(valid_action, reserve_available)
        self.genset_sim.step(valid_action, reserve_available = reserve_available)
        observations = self.gather_observations()
        return observations

    def gather_observations(self):
        # Gather observations from the genset simulator
        device_observations = self.genset_sim.gather_observations()
        controller_state = {
            'status': self.status,
            'time_since_warmup': self.time_since_warmup,
            'time_since_cooldown': self.time_since_cooldown,
            'time_since_start': self.time_since_start,
            'average_active_power': self.average_active_power,
            'available_power': self.available_power,
            'controllable_status': self.controllable_status,
            'controllable_power': self.controllable_power,
            'obs_type': 'state'
        }
        observations = {'device_observations': device_observations, 'controller_state': controller_state}
        return observations

    def check_valid(self, processed_action):
        pass

    def generate_safe_action(self, action, reserve_available):
        # Generate a safe action based on the given action and safety parameters. Update the state machine and the available power in the controller state based on the action.
        status_change = action['status_change']
        power_setpoint = action['power_setpoint']
        self.status, safe_status_change = self.generate_status_change_and_update_state_machine(status_change)
        self.controllable_status = self.check_controllable_status()
        self.controllable_power = self.check_controllable_power()
        self.available_power = self.update_available_power(reserve_available)
        safe_setpoint = self.generate_safe_setpoint(power_setpoint)
        self.average_active_power = self.update_average_active_power(safe_setpoint)  # Update the average active power based on the setpoint (could be improved by taking the observation, but it messes things up re: updating the other simulators)

        safe_action = {'status_change': safe_status_change, 'power_setpoint': safe_setpoint}
        return safe_action

    def update_available_power(self, reserve_available):
        """
        Updates the available power based on the state
        """
        # Status limit
        if self.status == 'warmup':
            self.available_power = self.warmup_power
        elif self.status == 'cooldown':
            self.available_power = self.cooldown_power
        elif self.status == 'off':
            self.available_power = 0
        elif self.status == 'running':
            self.available_power = self.compute_running_available_power(reserve_available)       
        else:
            raise ValueError(f'{self.status} is not defined. self.status should be among running, warmup, cooldown, and off')   

        return self.available_power   

    def predict_available_power(self, status_action, reserve_available):
        """
        Predicts the available power based on the status action and the reserve available
        """

        if (self.status == 'running' and self.controllable_status) and status_action == 'stop':
            available_power = self.cooldown_power
        elif (self.status == 'off' or (self.status == 'cooldown' and self.controllable_status)) and status_action == 'start':
            available_power = self.warmup_power
        elif self.status == 'warmup' and self.time_since_warmup >= self.warmup_time - self.time_step:
            available_power = self.compute_running_available_power(reserve_available)  # If the genset is in last step of warmup, we can predict the available power based on the running available power
        else:
            available_power = self.available_power

        return available_power
    
    def predict_controllable_power(self, status_action):
        """
        Predicts if power is controllable based on the status action
        """
        if self.status == 'running' and self.controllable_status and status_action == 'stop':                                       # Getting to cooldown
            controllable_power = False
        elif self.status == 'warmup' and self.time_since_warmup >= self.warmup_time - self.time_step:                               # Getting to running
            controllable_power = True
        else:
            controllable_power = self.controllable_power

        return controllable_power
    
    def predict_controllable_status(self, status_action):
        """
        Predicts if the status is controllable based on the status action
        """
        if self.status == 'running' and self.controllable_status and status_action == 'stop':                                       # Getting to cooldown
            controllable_status = False
        elif self.status == 'warmup' and self.time_since_warmup >= self.warmup_time - self.time_step:                               # Getting to running
            controllable_status = True
        else:
            controllable_status = self.controllable_status

        return controllable_status

    
    def compute_running_available_power(self, reserve_available):
            # Capacity power limit
            if reserve_available:
                capacity_available_power = np.round(self.genset_sim.prime_power_rating * self.genset_sim.temp_overload_factor, 2)
            else:
                capacity_available_power = self.genset_sim.prime_power_rating
            
            # Average power limit
            max_average_active_power = self.average_active_power_limit * self.genset_sim.prime_power_rating
            average_power_limit = (self.average_active_power_window * max_average_active_power - (self.average_active_power_window - self.time_step) * self.average_active_power) / self.time_step   # Power limit over which the average power over the last active_power_window will reach the maximum average power

            available_power = np.minimum(capacity_available_power, average_power_limit)      # Total availalbe power does not account for the status of the genset. It is needed when setting the power setpoint      

            return available_power    
        
    def generate_safe_setpoint(self, power_setpoint):
        """
        Generate a safe setpoint for the genset based on the constraints
        """
        ## Update active power
        if self.status == 'warmup':
            power_setpoint = self.warmup_power
        elif self.status == 'cooldown':
            power_setpoint = self.cooldown_power
        elif self.status == 'running':               
            power_setpoint = np.maximum(np.minimum(power_setpoint, self.available_power), self.minimum_load)
        elif self.status == 'off':
            power_setpoint = 0
        else:
            raise ValueError(f'{self.status} is not defined. self.status should be among running, warmup, cooldown, and off')
        
        return power_setpoint
    
    def update_average_active_power(self, active_power):
        ## Updating the average active power

        if self.average_active_power_constraint:

            if self.status in ['running', 'warmup', 'cooldown']:
                if active_power < 0.3 * self.genset_sim.prime_power_rating:
                    active_power_for_average = 0.3 * self.genset_sim.prime_power_rating        # if the power is below 30% of the prime power, we consider it as 30% for the average
                else:
                    active_power_for_average = active_power
                
                average_active_power = (self.average_active_power * (self.average_active_power_window - self.time_step) + active_power_for_average * self.time_step) / self.average_active_power_window

            
            elif self.status == 'off':          # Average active power is to be updated while the genset is on only.
                average_active_power = self.average_active_power
                
        else:
            average_active_power = 0       # If the constraint is not active, we do not update the average active power

        return average_active_power


    def generate_status_change_and_update_state_machine(self, status_change):
        # Updates the state machine based on the action and returns the status change action to send to the genset. # Could be improved by taking the observation, but it messes things up re: updating the other simulators)
        if self.status == 'off':
            if status_change == 'start':
                new_status = 'warmup'
                valid_status_change = 'start'
            else:
                new_status = 'off'
                valid_status_change = 'none'


        elif self.status == 'warmup':
            self.time_since_warmup += self.time_step
            valid_status_change = 'none'
            if self.time_since_warmup >= self.warmup_time:          # Limits the value so as not to be OOD for the agent.
                self.time_since_warmup = 0
                new_status = 'running'
                self.time_since_start += self.time_step 
            else:
                new_status = 'warmup'
        
        elif self.status == 'cooldown':
            self.time_since_cooldown += self.time_step
            if self.controllable_status:
                self.time_since_cooldown = 0    # Reset the time since start as we turned off.
                if status_change == 'start':                                 # If we want to re-start, we can now.
                    new_status = 'warmup'
                    valid_status_change = 'none'
                else:                                                       # Otherwise, we turn off completely.
                    new_status = 'off'
                    valid_status_change = 'stop'
            else:
                valid_status_change = 'none'
                new_status = 'cooldown'

        elif self.status == 'running':
            self.time_since_start += self.time_step 
            if status_change == 'stop':
                if self.controllable_status:
                    new_status = 'cooldown'
                    self.time_since_start = 0
                else:
                    new_status = 'running'              # No change
                valid_status_change = 'none'
            else:
                new_status = 'running'
                valid_status_change = 'none'


            if self.time_since_start > self.minimum_run_time + self.time_step:          # Limits the value so as not to be OOD for the agent.
                self.time_since_start = self.minimum_run_time + self.time_step




        return new_status, valid_status_change
        
    
    def check_controllable_status(self):
        # Returns True if the genset status is controllable (i.e. it is not in cooldown or warmup)
        if (self.status == 'running' and self.time_since_start >= self.minimum_run_time - self.time_step) or (self.status == 'cooldown' and self.time_since_cooldown >= self.cooldown_time - self.time_step):
            controllable_status = True
        else:
            controllable_status = False

        return controllable_status
    
    def check_controllable_power(self):
        if self.status == 'running' or (self.status == 'warmup' and self.time_since_warmup >= self.warmup_time - self.time_step):  # In the last case, it means it will reach running next time step.
            controllable_power = True
        else:
            controllable_power = False

        return controllable_power

    def assess_off(self):
        # Returns True if the genset is off and does not produce anything
        return self.status == 'off' and self.genset_sim.active_power == 0
    

    def predict_step(self, action, reserve_available):
        # Predict the next step of the genset simulator based on the given action
        copy_self = copy.deepcopy(self)
        observations_pred = copy_self.simulator_dynamic_update(action, reserve_available)
        return observations_pred



