import numpy as np
import pprint
import copy
import os
import sys
from noise import pnoise1
import datetime
import pandas as pd
import time
from sortedcontainers import SortedDict

try:
    from env.abstract_classes import System
except:
    from abstract_classes import System

class WindTurbine(System):
    def __init__(self, params, time_step, real) -> None:
        """
        Inputs:
            - const_params: Dictionary with the constant parameters of the wind turbine, with the following keys:
                - pred_time_step: Prediction time step in minutes
                - nb_pred_time_steps: Number of prediction time steps
                - mode: Mode of the wind turbine (e.g., perlin)
                - perlin_params: Parameters of the Perlin noise
            - init_params: Dictionary with the initial parameters of the wind turbine, with the following keys:
                - date_time: Date time of the initial observation (automatically set by microgrid, no need to add it in the config)
                - setpoint: initial setpoint
        """
        const_params = params['const_params']
        init_params = params['init_params']

        self.time_step = time_step

        self.real = real
        self.isDevice = True

        self.nominal_power = const_params['nominal_power']['value']  # In kW. Needed for the microgrid controller.

        if not init_params['active']['value']:
            self.active = False
            try:
                self.nb_pred_time_steps = const_params['nb_pred_time_steps']['value'] # Number of prediction time steps
            except:
                self.nb_pred_time_steps = 10

        else:
            self.active = True        
            self.mode = const_params['mode']['value']        # Perlin or data or dummy
            if self.mode == "perlin":
                self.perlin_generators = {}     # One per day of the year (each day changes the base). Will be built on the fly
                const_params['perlin_params']['amplitude'] = {'value': self.nominal_power - const_params['perlin_params']['average']['value'], 'type': 'float'}
                self.perlin_params = const_params['perlin_params']
            elif self.mode == "data":
                self.production_df, self.predict_dict = initialize_wind_data(self.nominal_power)
            elif self.mode == "dummy":
                self.dummy_power = 0
                self.dummy_power_next = 0
            else:
                raise ValueError("Invalid turbine mode {}. Must be perlin, data or dummy.".format(self.mode))
            
            self.pred_time_step = const_params['pred_time_step']['value']  # In minutes
            self.nb_pred_time_steps = const_params['nb_pred_time_steps']['value'] # Number of prediction time steps

        self.reset(init_params)
        

    def reset(self, init_params):
        if self.active:
            if self.mode == "perlin":
                self.perlin_generators = {}
            self.available_power = self.get_available_wind_power(init_params['date_time']['value'])
            self.available_power_next = self.get_available_wind_power(init_params['date_time']['value']+datetime.timedelta(minutes=1), next_step=True)
            self.power = np.minimum(self.available_power, init_params['turbine_setpoint']['value'])
            self.available_wind_power_pred = self.get_available_wind_power_prediction(init_params['date_time']['value'])
            self.turbine_setpoint = init_params['turbine_setpoint']['value']
            if self.mode == "dummy":
                self.dummy_power = self.power
                self.dummy_power_next = self.power
            observations = self.gather_observations()
        else:
            observations = self.gather_inactive_observations()
        return observations

    def step(self, action, next_step = False, verbose = False):
        """
        Step the wind turbine.
        Inputs:
            - action: Dictionary with the action of the wind turbine, with the following
                - turbine_setpoint: power setpoint, in kW
            - date_time: Date time of the step
            - verbose: Boolean to print the observations
        Outputs:
            - observations: Dictionary with the observations of the wind turbine, with the following
                - available_wind_power: Available wind power, in kW
                - turbine_setpoint: power setpoint, in kW
                - wind_power: Wind power, in kW
                - available_wind_power_pred: Available wind power prediction, list, in kW
        """
        turbine_setpoint = action['turbine_setpoint']
        date_time = action['date_time']
        if self.active:
            self.available_power = self.get_available_wind_power(date_time, next_step=next_step)
            self.available_power_next = self.get_available_wind_power(date_time+datetime.timedelta(minutes=1), next_step=True)
            self.turbine_setpoint = turbine_setpoint
            self.power = np.minimum(self.available_power, np.maximum(0, self.turbine_setpoint))   # We cannot produce more than the available power, we should not produce more than the setpoint, we cannot produce negative power
            self.available_wind_power_pred = self.get_available_wind_power_prediction(date_time)
 

    def predict_step(self, action):
        """
        Makes a fake step to predict the next observations
        """
        date_time = action['date_time']
        observations_pred = {
            'available_wind_power': self.get_available_wind_power(date_time),
            'wind_power': np.minimum(self.get_available_wind_power(date_time), np.maximum(0, action['turbine_setpoint'])),
            'available_wind_power_pred': self.get_available_wind_power_prediction(date_time)
        }

        return observations_pred

    def gather_observations(self):
        if self.active:
            observations = {}
            observations['active'] = True
            observations['available_wind_power'] = self.available_power
            observations['available_wind_power_next'] = self.available_power_next
            observations['wind_power'] = self.power
            observations['available_wind_power_pred'] = self.available_wind_power_pred
            observations['obs_type'] = 'state' 

            return observations
        else:
            return self.gather_inactive_observations()

    def gather_inactive_observations(self):
        observations = {}
        observations['active'] = False
        observations['available_wind_power'] = 0
        observations['wind_power'] = 0
        try:
            observations['available_wind_power_pred'] = [0] * self.nb_pred_time_steps
        except:
            observations['available_wind_power_pred'] = [0] * 10
        observations['obs_type'] = 'state'
        
        return observations
    

    def simulator_sensor_update(self, device_observations, next_step=False):
        """
        Updates the simulator based on the observation from the true wind turbine.

        Inputs:
            device_observations is a dictionary with the following keys:
                wind_speed: wind speed in m/s
                power_output: power output of the wind turbine in kW
                temperature: temperature in degrees Celsius
                humidity: humidity in percentage
                wind_direction: wind direction in degrees
        """
        if self.real:
            raise ValueError("This function should only be called if the battery is a simulator")
        
        # update simulator parameters
        if device_observations['obs_type'] == 'state' and device_observations['active']:
            if next_step:
                self.available_power = device_observations['available_wind_power_next']
                self.dummy_power = device_observations['available_wind_power_next']
            else:
                self.available_power = device_observations['available_wind_power']
                self.dummy_power = device_observations['available_wind_power']

            self.available_power_next = device_observations['available_wind_power_next']
            self.dummy_power_next = device_observations['available_wind_power_next']    
            self.power = device_observations['wind_power']
            self.available_wind_power_pred = device_observations['available_wind_power_pred']       

        elif device_observations['obs_type'] == 'params' and device_observations['active']['value']:

            self.available_power = self.get_available_wind_power(device_observations['date_time']['value'])
            self.available_power_next = self.get_available_wind_power(device_observations['date_time']['value']+datetime.timedelta(minutes=1), next_step=True)
            self.power = np.minimum(self.available_power, device_observations['turbine_setpoint']['value'])
            if self.mode == 'dummy':
                self.dummy_power = self.power
                self.dummy_power_next = self.power
            self.available_wind_power_pred = self.get_available_wind_power_prediction(device_observations['date_time']['value'])

        
        elif device_observations['obs_type'] not in ['state', 'params']:
            raise ValueError("The observation type should be either state or params") 


    def get_available_wind_power(self, date_time, next_step=False):
        if self.active:
            if self.mode == "perlin":
                hour = date_time.hour
                minute = date_time.minute

                minute_of_day = hour * 60 + minute
                day_of_year = date_time.timetuple().tm_yday         # To be used as a base for the Perlin noise generator

                if day_of_year not in self.perlin_generators.keys():
                    self.perlin_generators[day_of_year] = self.make_perlin_generator(self.perlin_params, day_of_year)

                wind_power = self.perlin_generators[day_of_year].calculate_wind(minute_of_day)

                return wind_power
            elif self.mode == "data":
                # Calculate the index in the data as the minute of the year
                index = (date_time.timetuple().tm_yday - 1) * 24 * 60 + date_time.hour * 60 + date_time.minute
                # Get the value at the calculated index
                wind_power = self.production_df.iloc[index]['AvailableWindPower']
                return wind_power
            elif self.mode == "dummy":
                wind_power = self.dummy_power if not next_step else self.dummy_power_next
                return wind_power
            else:
                raise ValueError('Invalid wind turbine mode {} '.format(self.mode) + 'Available modes: perlin, data, dummy')
        
        else:
            return 0
        
    def set_dummy_power(self, dummy_power, dummy_power_next):
        self.dummy_power = dummy_power
        self.dummy_power_next = dummy_power_next
        self.available_power = dummy_power
        self.available_power_next = dummy_power_next

    def get_available_wind_power_prediction(self, date_time):
        wind_power_prediction = []

        if self.mode == "perlin":
            # Sample from the Perlin noise generator
            for i in range(self.nb_pred_time_steps):
                future_date_time = date_time + datetime.timedelta(minutes = (i+1) * self.pred_time_step)
                wind_power_prediction.append(self.get_available_wind_power(future_date_time))
        elif self.mode == "data":
            date_time = date_time.replace(second=0, microsecond=0)
            # Find the closest date in the prediction data
            closest_date = find_closest_prev_date(date_time, self.predict_dict)
            # Get the full per-minute prediction for the closest date (for the next 10 hours)
            wind_power_prediction_data = self.predict_dict[closest_date]
            # Filter to get only the steps we need (every pred_time_step minutes, for nb_pred_time_step steps, starting at the current time + one time step)
            # Get index of current datetime in the prediction data
            if len(wind_power_prediction_data.index[(wind_power_prediction_data['DateTime'] == date_time + datetime.timedelta(minutes=self.pred_time_step))].values) > 0:
                x = wind_power_prediction_data.index[(wind_power_prediction_data['DateTime'] == date_time + datetime.timedelta(minutes=self.pred_time_step))].values[0]
            else:
                x = 0

            wind_power_prediction = wind_power_prediction_data.loc[x : x+self.nb_pred_time_steps*self.pred_time_step-1 : self.pred_time_step, ['AvailableWindPowerForecast']].values.ravel().tolist()  # Get the prediction of the next nb_pred_time_step steps
            if len(wind_power_prediction) < self.nb_pred_time_steps:        # Can happen if we are at the end of the year
                #Appending values to wind_power_prediction
                wind_power_prediction = wind_power_prediction + [wind_power_prediction[-1]] * (self.nb_pred_time_steps - len(wind_power_prediction))

        elif self.mode == "dummy":
            return []
            #raise NotImplementedError("The get_available_wind_power_prediction function should have been overridden or not called, as the wind turbine mode is 'dummy'")                                                                 
        else:
            raise ValueError('Wind turbine mode {} not implemented. '.format(self.mode) + 'Available modes: perlin, data, dummy')

        return wind_power_prediction

    def make_perlin_generator(self, perlin_params, day_of_year):
        base = day_of_year

        perlin_params['amplitude'] = perlin_params['amplitude']             # In kW          # Could be changed to a function of the day of the year
        perlin_params['average'] = perlin_params['average']                 # In kW          # Could be changed to a function of the day of the year
        perlin_generator = PerlinWindPowerGenerator(perlin_params, base)

        return perlin_generator


# Perlin noise generator
# Inspired from:

class SinglePerlinGenerator:
    """
    Single Perlin generator. Can be used to generate a Perlin noise. For some documentation: https://kmdouglass.github.io/posts/perlin-noise/
    """
    def __init__(self, octaves, persistence, lacunarity, repeat, base):
        """
        Inputs:
            - octaves: Number of octaves in the perlin noise
            - persistence: Persistence
            - lacunarity: Lacunarity
            - repeat: Repeat
            - base: Base
        """
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.repeat = repeat
        self.base = base

    def noise(self, x):
        return pnoise1(x, octaves=self.octaves, persistence=self.persistence, lacunarity=self.lacunarity, repeat=self.repeat, base=self.base)


class PerlinWindPowerGenerator:
    """
    Wind power generator with Perlin noise. Inspired from  https://github.com/maivincent/marl-demandresponse-original/blob/main/utils.py#L1203 .
    We sum several Perlin noises with different octaves and normalize them. I am not sure to understand exactly what is being done here; however, it works.
    We then multiply the result by the amplitude and add the average to get wind power.
    """
    def __init__(self, perlin_params, base):
        """
        Inputs:
            - perlin_params: Dictionary with the parameters of the Perlin noise, with the following keys:
                - amplitude: Amplitude of the noise
                - nb_octaves: Number of octaves
                - octaves_step: Octaves step
                - period: Period
                - persistence: Persistence
                - lacunarity: Lacunarity
                - repeat: Repeat
                - average: Average (offset of the noise)
            - base: Base (equivalent to the seed of the Perlin noise)
        """
        self.amplitude = perlin_params['amplitude']['value']             # In kW
        self.nb_octaves = perlin_params['nb_octaves']['value']            # 5
        self.octaves_step = perlin_params['octaves_step']['value']        # 4
        self.period = perlin_params['period']['value']                    # 24*60nb_octaves = 5, octaves_step = 4, period = 60*24,  base = 10, persistence = 0.8, lacunarity = 2.5
        self.persistence = perlin_params['persistence']['value']          # 0.8
        self.lacunarity = perlin_params['lacunarity']['value']            # 2.5
        self.repeat = perlin_params['repeat']['value']                    # 1024
        self.average = perlin_params['average']['value']                  # In kW

        self.base = base

        self.noise_list = []


        # Building the Perlin noise generators
        for i in range(self.nb_octaves):
            self.noise_list.append(
                SinglePerlinGenerator(octaves=2**i * self.octaves_step, persistence=self.persistence, lacunarity=self.lacunarity, repeat=self.repeat, base=base)
            )

    def calculate_wind(self, x):
        noise = 0

        # Calling all the Perlin noise generators
        for j in range(self.nb_octaves - 1):
            noise += self.noise_list[j].noise(x / self.period) / (2**j)
        noise += self.noise_list[-1].noise(x / self.period) / (2**self.nb_octaves - 1)

        wind_power = np.maximum(0, self.amplitude * noise + self.average)   # We cannot produce negative power
        return wind_power



def initialize_wind_data(nominal_power):
    """
    Checks if the data file exists, loads it.
    """
    # Import parquet file
    norm_avail_wind_power_data_path1 = os.path.join('..', 'data', 'norm_avail_wind_power_data.parquet')
    norm_avail_wind_power_data_path2 = os.path.join('data', 'norm_avail_wind_power_data.parquet')
    norm_avail_wind_power_pred_path1 = os.path.join('..', 'data', 'norm_avail_wind_power_forecast.parquet')
    norm_avail_wind_power_pred_path2 = os.path.join('data', 'norm_avail_wind_power_forecast.parquet')

    if os.path.exists(norm_avail_wind_power_data_path1) and os.path.exists(norm_avail_wind_power_pred_path1):
        print("Wind turbine parquet data found, importing...")
        available_wind_data = pd.read_parquet(norm_avail_wind_power_data_path1)
        predict_wind_data = pd.read_parquet(norm_avail_wind_power_pred_path1)

    elif os.path.exists(norm_avail_wind_power_data_path2) and os.path.exists(norm_avail_wind_power_pred_path2):
        print("Wind turbine parquet data found, importing...")
        available_wind_data = pd.read_parquet(norm_avail_wind_power_data_path2)
        predict_wind_data = pd.read_parquet(norm_avail_wind_power_pred_path2)

    else:
        raise FileNotFoundError("Wind turbine data files not found.")

    # Denormalize the data
    available_wind_data['AvailableWindPower'] = available_wind_data['AvailableWindPower'] * nominal_power
    predict_wind_data['AvailableWindPowerForecast'] = predict_wind_data['AvailableWindPowerForecast'] * nominal_power

    # Group predictions by 'DateTimeUpdated'
    grouped = predict_wind_data.groupby('DateTimeUpdated')

    # Create a SortedDict to store the grouped data
    predict_wind_dict = SortedDict({name: group[['DateTime', 'AvailableWindPowerForecast']].reset_index(drop=True) for name, group in grouped})

    print("Wind turbine data initialized.")

    return available_wind_data, predict_wind_dict


def find_closest_prev_date(target_date, sorted_dict):
    keys = sorted_dict.keys()
    pos = sorted_dict.bisect_left(target_date)
    if pos == 0:
        return keys[0]
    if pos == len(keys):
        return keys[-1]
    return keys[pos - 1]

