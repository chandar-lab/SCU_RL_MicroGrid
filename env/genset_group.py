import numpy as np
import pprint
import copy

try:
    from env.genset_controller import GensetController
    from env.genset import Genset
    from env.abstract_classes import System
except:
    from genset_controller import GensetController
    from genset import Genset
    from abstract_classes import System


class GensetGroup(System):
    """
    Group of gensets
    """

    def __init__(self, params, time_step, real):
        """
        Inputs:
            - params: dictionary with the following keys:
        """

        const_params = params['const_params']
        init_params = params['init_params']
        self.real = real
        self.isDevice = False

        self.time_step = time_step

        self.n_gensets = const_params['n_gensets']['value']

        self.reset(init_params, real)



    def reset(self, init_params, real):
        """
        Resets the genset group to the initial state
        """

        self.genset_ids = []
        self.gensets = {}
        self.genset_controllers = {}

        for id in range(self.n_gensets):
            genset_device_params = init_params['gensets'][id]['device']
            self.genset_ids.append(id)
            self.gensets[id] = Genset(genset_device_params, self.time_step, real=real)
            self.genset_controllers[id] = GensetController(init_params['gensets'][id], self.time_step)
            self.genset_controllers[id].update_controller_state({'device_observations': self.gensets[id].gather_observations()})

        self.update_lists()
        self.update_power()
        self.update_available_power(reserve_available=True)
        self.update_fuel_consumption()

        self.max_warmup_time = np.max([self.genset_controllers[id].warmup_time for id in self.genset_ids])
        self.max_cooldown_time = np.max([self.genset_controllers[id].cooldown_time for id in self.genset_ids])

        return self.gather_observations()
    

    def simulator_sensor_update(self, device_observations):
        """
         Only called if this is a simulator. Updates the state of the simulator based on real observations.
        """
        if self.real:
            raise ValueError("This function should only be called if the genset group is a simulator")

        for id in self.genset_ids:
            # Update the controllers with the device observations
            self.genset_controllers[id].update_controller_state(device_observations['gensets'][id])
            # Update the devices with the device observations
            self.gensets[id].simulator_sensor_update(device_observations['gensets'][id]['device_observations'])

        if device_observations['obs_type'] == 'state':
            self.genset_group_active_power = device_observations['genset_group_active_power']
            self.genset_group_fuel_consumption = device_observations['genset_group_fuel_consumption']
            self.genset_group_available_power = device_observations['genset_group_available_power']
            self.running_gensets_ids = device_observations['config_lists']['running_gensets_ids']
            self.warmup_gensets_ids = device_observations['config_lists']['warmup_gensets_ids']
            self.cooldown_gensets_ids = device_observations['config_lists']['cooldown_gensets_ids']
            self.off_gensets_ids = device_observations['config_lists']['off_gensets_ids']
        
        elif device_observations['obs_type'] == 'params':
            self.update_lists()
            self.update_power()
            self.update_available_power(1, reserve_available=True)
            self.update_fuel_consumption()
    

    def update_lists(self):
        """
        Updates the lists of gensets in the group
        """
        self.running_gensets_ids = []                   # Running, not in warmup or cooldown
        self.warmup_gensets_ids = []                    # In warmup
        self.cooldown_gensets_ids = []                  # In cooldown
        self.off_gensets_ids = []                       # Off
        for id in self.genset_ids:
            if self.genset_controllers[id].status == 'warmup':
                self.warmup_gensets_ids.append(id)
            elif self.genset_controllers[id].status == 'cooldown':
                self.cooldown_gensets_ids.append(id)
            elif self.genset_controllers[id].status == 'running':
                self.running_gensets_ids.append(id)
            elif self.genset_controllers[id].status == 'off':
                self.off_gensets_ids.append(id)
            else:
                raise ValueError("Invalid status for genset {}: {}".format(id, self.genset_controllers[id].status))


    def step(self, individual_actions, reserve_available, verbose = False):
        """
        Inputs:
            - action: dictionary with the following keys:
                - status_change: string, 'start_next', 'none', or 'stop_last'      # Action to take for the group of gensets
                - power_setpoint: float, in kW              # Total power setpoint for the group of gensets
        """

        ## Step the gensets
        if verbose:
            print("Genset group sending following individual actions: ")
            pprint.pp(individual_actions)

        for id in self.genset_ids:

            safe_individual_action = self.genset_controllers[id].generate_safe_action(individual_actions[id], reserve_available)
            self.gensets[id].step(safe_individual_action)
            device_observations = self.gensets[id].gather_observations()

            self.genset_controllers[id].update_controller_state({'device_observations': device_observations})

        # Update the group variables

        self.update_lists()        
        self.update_power()
        self.update_available_power(reserve_available)
        self.update_fuel_consumption()


        ## Return observations
        #observations = self.gather_observations()

        #return observations
    
    def update_available_power(self, reserve_available):
        self.genset_group_available_power = 0

        for id in range(self.n_gensets):
            self.genset_controllers[id].update_available_power(reserve_available)
            self.genset_group_available_power += self.genset_controllers[id].available_power
        
        return self.genset_group_available_power
        


    def update_power(self):
        """
        Sums the active power of all gensets in the group
        """
        self.genset_group_active_power = 0
        for id in self.genset_ids:
            self.genset_group_active_power += self.genset_controllers[id].genset_sim.active_power


    def update_fuel_consumption(self):
        """
        Sums the fuel consumption of all gensets in the group
        """
        self.genset_group_fuel_consumption = 0
        for id in self.genset_ids:
            self.genset_group_fuel_consumption += self.genset_controllers[id].genset_sim.fuel_consumption


    def gather_observations(self):
        """
        Returns a dictionary with the observations of the group of gensets
        In details:
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
        """
        
        observations = {
            'genset_group_active_power': self.genset_group_active_power,
            'genset_group_fuel_consumption': self.genset_group_fuel_consumption,
            'genset_group_available_power': self.genset_group_available_power,
            'gensets': {},
            'config_lists': {
                'running_gensets_ids': self.running_gensets_ids,
                'warmup_gensets_ids': self.warmup_gensets_ids,
                'cooldown_gensets_ids': self.cooldown_gensets_ids,
                'off_gensets_ids': self.off_gensets_ids,
            },
            'obs_type': 'state'
        }

        for id in self.genset_ids:
            observations['gensets'][id] = self.genset_controllers[id].gather_observations()

        return observations




