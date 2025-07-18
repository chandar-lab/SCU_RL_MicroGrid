import numpy as np
import pprint
import copy

try:
    from .utils import add_params_noise
    from env.abstract_classes import System

except:
    from utils import add_params_noise
    from abstract_classes import System


class Battery(System):
    def __init__(self, params, time_step, real) -> None:

        const_params = params['const_params']
        init_params = params['init_params']

        self.time_step = time_step
        self.isDevice = True

        self.real = real
        # Randomize parameters for the device if testing for robustness
        if self.real and const_params['noise_level']['value'] not in ['None', 'none']:
            noise_level = const_params['noise_level']['value']
            print(f'Adding {noise_level} level noise to Battery device')
            for param in ["P_nom", "Q_max", "eta_charge"]:
                print(f'Original {param}: {const_params[param]["value"]}')
                const_params[param] = add_params_noise(const_params[param]['value'], const_params['noise_level']['value'], const_params['noise_range_percentage']['value'])
                print(f'Noisy {param}: {const_params[param]["value"]}')
            

        self.battery_constant_parameters = const_params
        # Fixed parameters
        self.soc_min = const_params["soc_min"]['value']
        self.soc_max = const_params["soc_max"]['value']
        self.I_nom = const_params["I_nom"]['value']           # Nominal current, in A
        self.Q_max = const_params["Q_max"]['value']           # Maximal capacity, in Ah
        # self.E_max = battery_constant_parameters["E_max"]           # Nominal energy, in kWh  (useless if we do not approximate)
        self.eta_charge = const_params["eta_charge"]['value']   # Charging efficiency, in ratio (from 0 to 1)
        self.R = const_params["R"]['value']                   # Internal resistance, in Ohm
        self.A = const_params["A"]['value']                   # Linear coefficient for open circuit voltage, in V
        self.B = const_params["B"]['value']                   # Value at 0 for open circuit voltage, in V
        self.P_nom = const_params["P_nom"]['value']           # Nominal power, in kW
        
        self.degradation_cost_type = const_params["degradation_cost_type"]['value']
        self.buffer_size = const_params['buffer_size']['value']
        # Battery degradation cost
        if self.degradation_cost_type == "linear":
            self.degradation_cost_calculator = LinearizedBatteryDegradationCost(alpha_d=const_params['alpha_d']['value'])
        elif self.degradation_cost_type == "cycle_based":
            self.degradation_cost_calculator = CycleBasedBatteryDegradationCost(alpha_d=const_params['alpha_d']['value'], beta=const_params['beta']['value'])
        else:
            raise ValueError("Unknown degradation cost type")

        # Continuous parameters
        self.reset(init_params)

        self.degradation_cost_calculator.soc_sp = [self.soc] * 3


    def reset(self, init_params):
        # Initial parameters
        self.soc = init_params["soc"]["value"]      # Initial state of charge, taken from init_params
        self.V_boc = self.A * self.soc + self.B     # Initial open circuit voltage, in V
        
        soc_pow_p_min, soc_pow_p_max = self.get_real_power_limits()
        p_grid = max(soc_pow_p_min, min(soc_pow_p_max, init_params["p_grid"]["value"]))
        self.step({"p_grid": p_grid})                 # Assume make a step at p to compute electric values

    def step(self, action, verbose = False):
        """
        Update the battery state according to the action
        Inputs:
            - action: dictionary with the following keys
                - p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
        """

        n = 60 / self.time_step                 # Nb of time steps in an hour
        prev_soc = self.soc

        min_p_grid, max_p_grid = self.get_real_power_limits()   # so that SOC never gets under soc_min or over soc_max

        ## Applying the power limits
        self.p_grid = np.clip(action["p_grid"], min_p_grid, max_p_grid)

        # Applying efficiency to compute the power on the DC (battery) side
        if self.p_grid < 0:
            self.p_batt = self.p_grid * self.eta_charge         # Power on the DC (battery) side, in kW. If charging, the power stored in the battery is lower than the power on the grid side
            p_batt_w = self.p_batt * 1000                       # Power on the DC (battery) side, in W for calculations
        else:
            self.p_batt = self.p_grid / self.eta_charge         # in kW. If discharging, the power taken from the battery is higher than the power on the grid side
            p_batt_w = self.p_batt * 1000                       # Power on the DC (battery) side, in W for calculations

        # Updating the other electrical values, in the right order
        self.I = (self.V_boc - np.sqrt(self.V_boc**2 - 4 * (self.R + self.A/(self.Q_max * n)) * p_batt_w)) / (2 * (self.R + self.A/(self.Q_max * n)))
        self.soc = self.soc - self.I / (self.Q_max * n)
        self.V_boc = self.A * self.soc + self.B      # Open circuit voltage, in V
        self.V = self.V_boc - self.R * self.I        # Voltage on the DC (battery) side, in V

        # Computing the degradation cost
        self.degradation_cost = self.degradation_cost_calculator.step(prev_soc, self.soc - prev_soc, verbose=verbose)

        self.available_power_charge, self.available_power_discharge = self.get_real_power_limits()       # This is for the agent's observations. 

        #observations = self.gather_observations()

        #return observations
    
    def simulator_sensor_update(self, device_observations):
        # Only called if this is a simulator. Updates the state of the simulator based on real device observations.
        if self.real:
            raise ValueError("This function should only be called if the battery is a simulator")
        
        if device_observations['obs_type'] == 'state':
            # Update the state of the battery based on the device observations
            self.soc = device_observations['soc']
            self.p_grid = device_observations['p_grid']
            try:        # If the info is there, use it. Otherwise, soc and p_grid are enough                self.available_power_charge = device_observations['available_power_charge']
                self.available_power_discharge = device_observations['available_power_discharge']
                self.degradation_cost = device_observations['degradation_cost']
                self.V = device_observations['V']
                self.I = device_observations['I']
                self.V_boc = device_observations['V_boc']
                self.p_batt = device_observations['p_batt']
            except:
                pass
        elif device_observations['obs_type'] == 'params':
            # Update the state of the battery based on the device observations
            self.soc = device_observations['soc']['value']
            self.p_grid = device_observations['p_grid']['value']
            try:        # If the info is there, use it. Otherwise, soc and p_grid are enough
                self.available_power_charge = device_observations['available_power_charge']['value']
                self.available_power_discharge = device_observations['available_power_discharge']['value']
                self.degradation_cost = device_observations['degradation_cost']['value']
                self.V = device_observations['V']['value']
                self.I = device_observations['I']['value']
                self.V_boc = device_observations['V_boc']['value']
                self.p_batt = device_observations['p_batt']['value']
            except:
                pass
        else:
            raise ValueError("The observation type should be either 'state' or 'params'")
        
    def get_real_power_limits(self):
        """
        Computes the power limits based on the state of charge and maximal intensity
        Inputs:
        Outputs:
            - min_p_grid: minimal power the battery can take on the grid side, in kW (negative value)
            - max_p_grid: maximal power the battery can provide on the grid side, in kW (positive value)
        """
        n = 60 / self.time_step                 # Nb of time steps in an hour

        ### Based on the state of charge: finding I limits to prevent an SOC outside the limits
        delta_soc_max = self.soc_max - self.soc     # How much the SOC can increase
        delta_soc_min = self.soc - self.soc_min     # How much the SOC can decrease        

        min_i_soc = -1 * delta_soc_max * self.Q_max * n # Negative value as it refers to the max "charging intensity" the battery can take
        max_i_soc = delta_soc_min * self.Q_max * n   # Positive value as it refers to the max "discharging intensity" the battery can provide

        ### Including nominal intensity
        min_i = max(min_i_soc, -self.I_nom)        # Final limit on charging intensity the battery can take (A, negative value)
        max_i = min(max_i_soc, self.I_nom)         # Final limit on discharging intensity the battery can provide (A, positive value)

        ### Computing the corresponding power limits on the DC (battery) side, in kW (negative for charging, positive for discharging)
        min_p_i_dc = (self.V_boc * min_i - (self.R + self.A/(self.Q_max * n)) * min_i**2)/1000        # DC charging power limit due to intensity, in kW (negative)               
        max_p_i_dc = (self.V_boc * max_i - (self.R + self.A/(self.Q_max * n)) * max_i**2)/1000        # DC discharging power limit due to intensity, in kW (positive)

        ## Converting to grid-side power limits
        min_p_i_grid = min_p_i_dc / self.eta_charge     # Charging, in kW. The power on the grid side is higher than the power stored in the battery
        max_p_i_grid = max_p_i_dc * self.eta_charge     # Discharging, in kW. The power on the grid side is lower than the power provided by the battery

        ## Including to the nominal power limit
        min_p_grid = np.clip(min_p_i_grid, -self.P_nom, self.P_nom)        # Final limit on charging power the battery can take on the grid side (kW, negative value)
        max_p_grid = np.clip(max_p_i_grid, -self.P_nom, self.P_nom)        # Final limit on discharging DC power the battery can provide on the grid side (kW, positive value)

        return min_p_grid, max_p_grid


    def gather_observations(self):
        observations = {
            "soc": self.soc,
            "p_grid": self.p_grid,
            "available_power_charge": self.available_power_charge,
            "available_power_discharge": self.available_power_discharge,
            "degradation_cost": self.degradation_cost,
            "V": self.V,
            "I": self.I,
            "V_boc": self.V_boc, 
            "p_batt": self.p_batt,
            'obs_type': 'state'
        }

       
        if self.degradation_cost_type == "cycle_based":
            reversed_R = self.degradation_cost_calculator.R if self.degradation_cost_calculator.R else []
            if len(reversed_R) < self.buffer_size:
                reversed_R = np.array([[-1]*(self.buffer_size-len(reversed_R)) + reversed_R])
            else:
                reversed_R = reversed_R[-self.buffer_size]
            observations["soc_tp_buffer"] = reversed_R
        else:
            observations["soc_tp_buffer"] = [1]*self.buffer_size

        return observations

    def print_state(self):
        print(
            "SOC: {}%, V_boc: {} V, V: {} V, I: {} A, P_batt: {} kW, P_grid: {} kW".format(self.soc * 100, self.V_boc, self.V, self.I, self.p_batt, self.p_grid)
        )   


class LinearizedBatteryDegradationCost:
    """
    Class representing the linearized battery degradation cost.

    Attributes:
        alpha_d (float): The degradation coefficient.
    """

    def __init__(self, alpha_d):
        """
        Initializes a new instance of the BatteryDegradationCost class.

        Args:
            initial_soc (float): The initial state of charge (SoC).
        """
        self.alpha_d = alpha_d

    def step(self, soc_t, b_t, verbose=False):

        # Calculate the degradation cost
        degradation_cost = self.alpha_d * abs(b_t)
        return degradation_cost




class CycleBasedBatteryDegradationCost:
    """
    Represents a class for calculating the degradation cost of a battery based on cycle-based degradation models.
    Inspired from https://www.sciencedirect.com/science/article/pii/S0026271421000780 and https://ieeexplore.ieee.org/document/9789478/ 
    Attributes:
        alpha_d (float): The degradation coefficient.
        beta (float): The degradation exponent.
        F (list): A list of three values representing the input signal.
        R (list): A list of discretized switching points.
        j (int): An index for tracking the current position in the R list.
        w (float): The width of the discretization window.

    Methods:
        hysteresis_filter(F):
        rainflow4p(R):
            Check if a cycle is found based on the rainflow counting algorithm.
        update_switching_points(x, F, R, verbose=False):
            Update the switching points based on the input signal.
        calculate_degradation_cost(soc_t, b_t):
            Calculate the degradation cost based on the current state of charge (SoC) and battery usage.
        step(soc_t, b_t, verbose=False):
            Update the switching points, calculate the degradation cost, and return the degradation cost.
    """

    def __init__(self, alpha_d, beta):

        self.alpha_d = alpha_d
        self.beta = beta
        self.F = None
        self.R = []
        self.j = 0
        self.w = 0.01
    
    def hysteresis_filter(self, F):
        """
        Apply hysteresis filter to the input signal.
        Parameters:
        - F (list): A list of three values representing the input signal.
        Returns:
        - F (list): The updated list of three values after applying the hysteresis filter.
        - tpFound (bool): A boolean value indicating whether a turning point was found.
        """


        def shift(F):
            F[0] = F[1]
            F[1] = F[2]
            return F

        def skip(F):
            F[1] = F[2]
            return F

        tpFound = False
        if F[2] < F[1]:
            if F[0] >= F[1]:
                F = skip(F)
            else:
                tpFound = True
                F = shift(F)
        elif F[2] > F[1]:
            if F[0] <= F[1]:
                F = skip(F)
            else:
                tpFound = True
                F = shift(F)
        else:
            if F[0] > F[1]:
                F[1] = min(F[1], F[2])
            else:
                F[1] = max(F[1], F[2])

        return F, tpFound


    def rainflow4p(self, R):
        cycleFound = False
        if min(R[-4], R[-1]) <= min(R[-3], R[-2]) and max(R[-3], R[-2]) <= max(R[-4], R[-1]):
            cycleFound = True
        return cycleFound
        

    def update_switching_points(self, x, F, R, verbose=False):
        
        F[2] = discretize(x)
        F, tpFound = self.hysteresis_filter(F)
        if tpFound:
            if verbose:
                print('TP found')
            R.append(discretize(F[0]))

        R.append(discretize(x))

        while len(self.R) >= 4 and self.rainflow4p(R):
            if verbose:
                print('rainflow condition satisfied')
            R[-3] = R[-1]
            del R[-2:]

        del R[-1]
        
        return F, R

    def calculate_degradation_cost(self, soc_t, b_t):
            """
            Calculates the degradation cost based on the current state of charge (SoC) and battery usage.

            Args:
                soc_t (float): The current state of charge (SoC).
                b_t (float): The battery usage.

            Returns:
                float: The degradation cost.
            """
            # Calculate the degradation cost
            degradation_cost = (
                self.alpha_d * np.exp(self.beta * abs(soc_t + b_t - self.R[-1]))
                - self.alpha_d * np.exp(self.beta * abs(soc_t - self.R[-1]))
            )
            if degradation_cost < 0: # If the degradation cost is negative, we compute it relative to the cost of discretization window
                degradation_cost_normalized = self.alpha_d * (np.exp(self.beta * abs(self.w)) - 1)
                degradation_cost = abs(b_t) * degradation_cost_normalized / self.w
            
            return degradation_cost

    def step(self, soc_t, b_t, verbose=False):
        """
        Updates the switching points, calculates the degradation cost, and returns the degradation cost.

        Args:
            soc_t (float): The current state of charge (SoC).
            b_t (float): The battery usage.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            float: The degradation cost.
        """
                # Update the switching points and calculate the degradation cost

        if isinstance(soc_t, np.ndarray):
            soc_t = soc_t.item()
        if self.F is None:
            self.F = [soc_t for _ in range(3)]
            self.R = [soc_t]
        
        if verbose:
          print(f"SoC: {soc_t}, b_t: {b_t}, SoC after: {soc_t+b_t}")
        self.F, self.R = self.update_switching_points(soc_t+b_t, self.F, self.R, verbose=verbose)
        if verbose:
          print(f"Buffer R: {self.R}")
        

        degradation_cost = self.calculate_degradation_cost(soc_t, b_t)
        if verbose:
          print(f"Degradation cost: {round(degradation_cost, 5)}")
          print('')

        return degradation_cost
    

def discretize(x):
    return np.round(x, 2).item()