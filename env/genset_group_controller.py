import numpy as np
import copy
try:
    from env.abstract_classes import Controller
    from env.genset_group import GensetGroup
except:
    from abstract_classes import Controller
    from genset_group import GensetGroup


class GensetGroupController(Controller):
    def __init__(self, genset_group_params, time_step):

        genset_group_controller_const_params = genset_group_params['controller']['const_params']
        genset_group_init_params = genset_group_params['controller']['init_params']

        self.time_step = time_step

        
        # Initializing the controller parameters
        ## No controller parameters to initialize

        # Initializing the simulator
        self.genset_group_sim = GensetGroup(genset_group_params['device'], self.time_step, real = False)


    def reset(self, genset_group_controller_init_params, genset_group_device_init_params):
        """
        Resets the controller and its simulator based on the initial parameters. 
        """
        # There is no init controller parameters.
        # Reset the simulator with the initial parameters
        self.genset_group_sim.reset(genset_group_device_init_params)

    def update_controller_state(self, observations):
        # Implement the logic to update the simulator based on observations from the device group controller
        
        device_observations = observations['device_observations']
        
        # update simulator state
        self.genset_group_sim.simulator_sensor_update(device_observations)

        # There is no controller state to update.


    def simulator_dynamic_update(self, action, reserve_available = False):
        """
        Updates the simulator dynamics based on the given action and time step.

        Args:
            action (any): The action to be applied to the simulator.
            reserve_available (bool): Indicates if reserve power is available.
        Returns:
            tuple: A tuple containing the observations gathered after the update.
        """
        # Implement the logic to update the simulator dynamics based on the action and time step
        valid_action = self.generate_safe_action(action, reserve_available) 
        self.genset_group_sim.step(valid_action, reserve_available = reserve_available)

        observations = self.gather_observations()
        return observations

    def gather_observations(self):
        # Implement the logic to gather observations from the simulator
        device_observations = self.genset_group_sim.gather_observations()
        observations = {'device_observations': device_observations, 'constroller_state': None}
        return observations

    def check_valid(self, processed_action):
        pass

    def generate_safe_action(self, action, reserve_available, verbose=False):
        """
        Generates safe actions for the genset group controller.
        Args:
            action (dict): A dictionary containing the action parameters.
                - status_change (str): The status change command.
                - power_setpoint (int): The power setpoint value.
            reserve_available (bool): Indicates if reserve power is available.
        Returns:
            dict: A dictionary containing individual actions for each genset.
                - status_change (str): The status change command for each genset.
                - power_setpoint (int): The power setpoint value for each genset.
        """
        
        status_change = action['status_change']
        if 'power_setpoint' in action:
            power_setpoint = action['power_setpoint']
        else:
            power_setpoint = 0
        if verbose:
            print("Status change command: {}".format(status_change))

        
        self.genset_group_sim.update_lists()

        ## Initialize individual actions
        individual_actions = {}
        
        for id in self.genset_group_sim.genset_ids:
            individual_actions[id] = {'status_change': None, 'power_setpoint': 0}
            
        ## Change status of gensets
        individual_actions = self.get_individual_status_change(individual_actions, status_change, verbose)
        self.genset_group_sim.update_available_power(reserve_available)
        
        ## Get power setpoints
        individual_actions =  self.get_individual_power_setpoints(individual_actions, power_setpoint, reserve_available, verbose)

        return individual_actions

    def get_individual_status_change(self, individual_actions, status_change, verbose = False):
        """
        This functions updates the status of the gensets based on the status_change command.
        The status_change command can be 'start_next', 'none', or 'stop_last'.
        
        Inputs:
            - individual_actions: dictionary with the following keys for each genset:
                - status_change: string, 'start', 'none', or 'stop'
                - power_setpoint: float, in kW
            - status_change: string, 'start_next', 'none', or 'stop_last'
            - verbose: Boolean, if True, print some information
        Outputs:
            - individual_actions: dictionary with the following keys for each genset:
                - status_change: string, 'start', 'none', or 'stop'
                - power_setpoint: float, in kW
        """

        # Check validity of status_change        
        if status_change not in ['start_next', 'none', 'stop_last']:
            raise ValueError("status_change should be 'start_next', 'none', or 'stop_last'")


        if status_change == 'none':
            ### Do nothing
            return individual_actions

        ### The order of the list is enforced as a hard constraint. Genset 2 cannot be running if genset 1 is not.
        ### Because of that priority order, it is not possible to start a genset if there is one in cooldown, nor is it possible to stop a genset if there is one in warmup or that has not passed its minimum run time.
        if status_change == 'start_next':
            if len(self.genset_group_sim.cooldown_gensets_ids) > 0:                                  # If there are gensets in cooldown, you cannot restart it, nor can you start a new one as it would break the priority list. Nothing happens.
                starting_genset_id = self.genset_group_sim.cooldown_gensets_ids[0]                   # Except: if the last genset in cooldown is ready to start, it can start at the next time step.
                if self.genset_group_sim.genset_controllers[starting_genset_id].controllable_status:
                    individual_actions[starting_genset_id]['status_change'] = 'start'           # The genset will be ready to start at the next time step.    
                    self.genset_group_sim.cooldown_gensets_ids.remove(starting_genset_id)
                    self.genset_group_sim.warmup_gensets_ids.append(starting_genset_id)
                else:
                    if verbose:
                        print("There are genset in cooldown and not ready to start. Ignoring start_next action.")

            elif len(self.genset_group_sim.off_gensets_ids) > 0:                                     # Otherwise, if there are gensets off, start the first one
                starting_genset_id = self.genset_group_sim.off_gensets_ids[0]                        # Restart the first "off" genset
                individual_actions[starting_genset_id]['status_change'] = 'start'               
                self.genset_group_sim.off_gensets_ids.remove(starting_genset_id)                     
                self.genset_group_sim.warmup_gensets_ids.append(starting_genset_id)   
            else:
                if verbose:
                    print("All gensets are running or in warmup. Ignoring start_next action.")
                                                                        # There are no more gensets to start
        elif status_change == 'stop_last':                                           
            # You can only stop a genset if the last one ON on the priority list is "running" for longer than the minimum run time. Otherwise, ignore.
            if len(self.genset_group_sim.warmup_gensets_ids) > 0:
                if verbose:
                    print("There are gensets in warmup. Ignoring stop_last action.")   
            elif len(self.genset_group_sim.running_gensets_ids) > 0:
                stopping_genset_id = self.genset_group_sim.running_gensets_ids[-1]                     # Stop the last genset in running (the one that was turned on the longest time ago)
                if self.genset_group_sim.genset_controllers[stopping_genset_id].controllable_status:
                    individual_actions[stopping_genset_id]['status_change'] = 'stop' 
                    self.genset_group_sim.running_gensets_ids.remove(stopping_genset_id)
                    self.genset_group_sim.cooldown_gensets_ids.append(stopping_genset_id)
                else:
                    if verbose:
                        print("The last genset in running has not reached its minimum run time. Ignoring stop_last action.")
            else:
                if verbose:                             # There are no more gensets to stop
                    print("All running gensets are off or in cooldown. Ignoring stop_last action.")     
                
        return individual_actions

    def get_individual_power_setpoints(self, individual_actions, power_setpoint, reserve_available, verbose = False):
        """
        This function computes the power setpoints for each genset based on the power setpoint of the group and the available power of each genset.
        Inputs:
            - individual_actions: dictionary with the following keys for each genset:
                - status_change: string, 'start', 'none', or 'stop'
                - power_setpoint: float, in kW
            - power_setpoint: float, in kW. Total power setpoint for the group of gensets
            - verbose: Boolean, if True, print some information
        Outputs:
            - individual_actions: dictionary with the following keys for each genset:
                - status_change: string, 'start', 'none', or 'stop'
                - power_setpoint: float, in kW
        """
        ### Compute non-controllable and available power
        non_controllable_power = 0
        controllable_power = 0
        controllable_genset_ids = []
        controllable_genset_max_ratios = []

        for id in range(self.genset_group_sim.n_gensets):
            if not self.genset_group_sim.genset_controllers[id].predict_controllable_power(individual_actions[id]['status_change']):             # If not controllable, the genset will produce its available power.
                individual_actions[id]['power_setpoint'] = self.genset_group_sim.genset_controllers[id].predict_available_power(individual_actions[id]['status_change'], reserve_available)
                non_controllable_power += self.genset_group_sim.genset_controllers[id].predict_available_power(individual_actions[id]['status_change'], reserve_available)
            else:                                                                                                                                 # If controllable, the genset can give up to its available power 
                controllable_genset_ids.append(id)
                controllable_power += self.genset_group_sim.gensets[id].prime_power_rating                                                        # We don't use available power, because the equal ratio constraint is about prime power rating (not the available power, which can include overload)
                controllable_genset_max_ratios.append(float(self.genset_group_sim.genset_controllers[id].predict_available_power(individual_actions[id]['status_change'], reserve_available)) / self.genset_group_sim.gensets[id].prime_power_rating)       # In case the available power is lower than the prime power rating, we need to take that into account in the ratio. 
 
        maximum_power_fraction = np.min(controllable_genset_max_ratios) if controllable_genset_max_ratios else 0                    # This is the maximum power fraction that can be used by the controllable gensets. It is the minimum of the ratios of available power to prime power rating of each controllable genset.

        ### Distribute power setpoints
        needed_power = power_setpoint - non_controllable_power
        if needed_power < 0: #  power setpoint is lower than non controllable power. Nothing we can do.
            needed_power = 0

        needed_power_fraction = needed_power / controllable_power if controllable_power > 0 else 0

        power_fraction = np.minimum(needed_power_fraction, maximum_power_fraction)

        for controllable_id in controllable_genset_ids:
            genset = self.genset_group_sim.gensets[controllable_id]
            individual_actions[controllable_id]['power_setpoint'] = np.round(genset.prime_power_rating * power_fraction, 2)
        
        return individual_actions

    def predict_step(self, action, reserve_available = False, verbose = False):
        # Implement the logic to predict the next step of the genset group based on the action and time step
        copy_self = copy.deepcopy(self)
        observations_pred = copy_self.simulator_dynamic_update(action, reserve_available)
        return observations_pred