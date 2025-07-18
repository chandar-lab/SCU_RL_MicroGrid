try:
    from env.abstract_classes import Controller
    from env.battery import Battery
except:
    from abstract_classes import Controller
    from battery import Battery
import copy
import numpy as np


class BatteryController(Controller):
    def __init__(self, battery_params, time_step):
        """
        Inputs:

        battery_sim_const_params is a dictionary containing the constant parameters for the battery simulator. See Battery object for details.
        battery_sim_init_params is a dictionary containing the initial parameters for the battery simulator. See Battery object for details.
        safety_parameters is a dictionary containing the safety parameters for the battery
            soc_max_norm: Maximal SOC the battery can reach in the normal mode
            soc_min_norm: Minimal SOC the battery can reach in the normal mode
            soc_max_res: Maximal SOC the battery can reach in the reserve mode
            soc_min_res: Minimal SOC the battery can reach in the reserve mode
        device_observations is a dictionary containing the observations from the true battery. See simulator_sensor_update for more details.
        """

        battery_controller_const_params = battery_params['controller']['const_params']
        self.time_step = time_step

        self.battery_sim = Battery(battery_params['device'], self.time_step, real = False) # Battery simulator

    
        self.soc_max_norm = battery_controller_const_params['soc_max_norm']['value']
        self.soc_min_norm = battery_controller_const_params['soc_min_norm']['value']
        self.soc_max_res = battery_controller_const_params['soc_max_res']['value']
        self.soc_min_res = battery_controller_const_params['soc_min_res']['value']


        # Must be set to the device observations to initialize the simulator
        self.reset(battery_params['controller']['init_params'], battery_params['device']['init_params'])  # Resetting the controller and the simulator



    def reset(self, battery_controller_init_params, battery_device_init_params):
        """
        Resets the controller to a new episode. Updates the simulator based on the battery's initial parameters.
        """
        # No controller state to reset

        # Reinitializating the simulator:
        self.battery_sim.simulator_sensor_update(battery_device_init_params)


    def update_controller_state(self, observations):
        """
        Updates the simulator based on the observation from the true battery.
        """
        # update simulator parameters
        self.battery_sim.simulator_sensor_update(observations['device_observations'])


    def simulator_dynamic_update(self, action, reserve_available, verbose = False):
        """
        Simulates a time step based on the simulator dynamics. Returns the observations from the simulator.
        Inputs:
            action is a dictionary with the following keys:
                p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
                mode: mode of the battery (normal or reserve)
        Outputs:
            observations dictionaries (see gather_observations(self) for more info)
        """
        valid_action = self.generate_safe_action(action, reserve_available)
        self.battery_sim.step(valid_action, verbose = False)
        observations = self.gather_observations()
        return observations


    def gather_observations(self):
        """
        Returns the observations from the battery simulator

        Outputs:
            observations is a dictionary with the following keys:
                soc: state of charge of the battery, in %
                p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
                degradation_cost: degradation cost of the battery, based on last action
        """
        
        device_observations = self.battery_sim.gather_observations()

        controller_state = {}
        
        observations = {"device_observations": device_observations, "controller_state": controller_state}

        return observations
    

    def check_valid(self, processed_action):
        pass

    def generate_safe_action(self, action, reserve_available):
        """
        Generate a valid action according to the constraints of the battery.

        Constraints of the battery:
            - Maximum power charge/discharge: P_nom
            - Maximum SOC
            - Minimum SOC
            - Maximal charging/charging intensity (I_nom)

        Inputs:
            action is a dictionary with the following keys:
                p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
                mode: mode of the battery (normal or reserve)

        Outputs:
            valid_action is a dictionary with the following keys:
                p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
        """
        
        p_grid = action['p_grid']

        # Check if the power does make the SOC exceed the maximum SOC or go below the minimum SOC, nor that the intensity in the battery is too high (I_nom)
        soc_pow_p_min, soc_pow_p_max = self.get_power_limits(reserve_available)

        p_grid = max(soc_pow_p_min, min(soc_pow_p_max, p_grid))
        
        valid_action = {'p_grid': p_grid}        # Action to be taken by the battery controller

        return valid_action

    def get_power_limits(self, reserve_available):
        """
        Computes the power limits based on the state of charge and maximal intensity
        Inputs:
            - reserve_available: Boolean, True if the battery is in reserve mode, False if it is in normal mode
        Outputs:
            - min_p_grid: minimal power the battery can take on the grid side, in kW (negative value)
            - max_p_grid: maximal power the battery can provide on the grid side, in kW (positive value)
        """
        n = 60 / self.time_step                 # Nb of time steps in an hour

        ### Based on the state of charge: finding I limits to prevent an SOC outside the limits
        if reserve_available:
            delta_soc_max = self.soc_max_res - self.battery_sim.soc     # How much the SOC can increase
            delta_soc_min = self.battery_sim.soc - self.soc_min_res     # How much the SOC can decrease
        else:
            delta_soc_max = self.soc_max_norm - self.battery_sim.soc     # How much the SOC can increase
            delta_soc_min = self.battery_sim.soc - self.soc_min_norm     # How much the SOC can decrease        
        min_i_soc = -1 * delta_soc_max * self.battery_sim.Q_max * n # Negative value as it refers to the max "charging intensity" the battery can take
        max_i_soc = delta_soc_min * self.battery_sim.Q_max * n   # Positive value as it refers to the max "discharging intensity" the battery can provide

        ### Including nominal intensity
        min_i = max(min_i_soc, -self.battery_sim.I_nom)        # Final limit on charging intensity the battery can take (A, negative value)
        max_i = min(max_i_soc, self.battery_sim.I_nom)         # Final limit on discharging intensity the battery can provide (A, positive value)

        ### Computing the corresponding power limits on the DC (battery) side, in kW (negative for charging, positive for discharging)
        min_p_i_dc = (self.battery_sim.V_boc * min_i - (self.battery_sim.R + self.battery_sim.A/(self.battery_sim.Q_max * n)) * min_i**2)/1000        # DC charging power limit due to intensity, in kW (negative)               
        max_p_i_dc = (self.battery_sim.V_boc * max_i - (self.battery_sim.R + self.battery_sim.A/(self.battery_sim.Q_max * n)) * max_i**2)/1000        # DC discharging power limit due to intensity, in kW (positive)

        ## Converting to grid-side power limits
        min_p_i_grid = min_p_i_dc / self.battery_sim.eta_charge     # Charging, in kW. The power on the grid side is higher than the power stored in the battery
        max_p_i_grid = max_p_i_dc * self.battery_sim.eta_charge     # Discharging, in kW. The power on the grid side is lower than the power provided by the battery

        ## Including to the nominal power limit
        min_p_grid = np.clip(min_p_i_grid, -self.battery_sim.P_nom, self.battery_sim.P_nom)        # Final limit on charging power the battery can take on the grid side (kW, negative value)
        max_p_grid = np.clip(max_p_i_grid, -self.battery_sim.P_nom, self.battery_sim.P_nom)        # Final limit on discharging DC power the battery can provide on the grid side (kW, positive value)

        return min_p_grid, max_p_grid
    

    def predict_step(self, action, reserve_available, verbose = False):
        """
        Makes a fake step to predict the next observations.
        Action is a dictionary with the following keys
            p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
            mode: mode of the battery (normal or reserve)
        """

        copy_self = copy.deepcopy(self)
        observations_pred = copy_self.simulator_dynamic_update(action, reserve_available, verbose = False)
        return observations_pred