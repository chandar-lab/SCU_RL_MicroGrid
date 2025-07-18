import numpy as np
import datetime
import copy
import os
import pandas as pd
import pprint

try:
    from env.abstract_classes import System
except:
    from abstract_classes import System


class Demand(System):
    def __init__(self, params, time_step, real):
        """
        Inputs:
            - params: dictionary with the following keys
                - data_file_path: string
                - date_time: datetime object
                - pred_time_step: int (in minutes)
                - nb_pred_time_steps: int (number of time steps to predict in the future)
                - normalisation_factor: float (to multiply the demand by)
                - mode: data or dummy (data: demand is based on data file. Dummy: functions are expected to be overriden)
        """

        const_params = params['const_params']
        init_params = params['init_params']

        self.time_step = time_step
        self.real = real
        self.isDevice = True


        self.mode = const_params['mode']['value']

        if self.mode == "data":
            self.pred_time_step = const_params['pred_time_step']['value']
            self.nb_pred_time_steps = const_params['nb_pred_time_steps']['value']
            self.normalisation_factor = const_params['normalisation_factor']['value']
            self.forecast_model = const_params['forecast_model']['value']
            self.demand_data = initialize_demand_data()
            if self.forecast_model == 'forecast':
                self.forecast_data = initialize_demand_forecast_data()

            elif self.forecast_model != 'ground_truth':
                raise ValueError(f'forecast model {self.forecast_model} not recognized')
        elif self.mode == "dummy":
            self.dummy_demand = 0
            self.dummy_demand_next = 0
        else:
            raise ValueError("Invalid demand mode {}. Must be data or dummy.".format(self.mode))
        
        self.reset(init_params)

    def reset(self, init_params):
        self.demand = self.get_demand(init_params['date_time']['value'])
        self.demand_next = self.get_demand(init_params['date_time']['value'] + datetime.timedelta(minutes=self.time_step), next_step=True)
        self.demand_prediction = self.get_demand_prediction(init_params['date_time']['value'])


    def step(self, action):
        date_time = action['date_time']
        self.demand = self.get_demand(date_time)
        self.demand_next = self.get_demand(date_time + datetime.timedelta(minutes=self.time_step), next_step=True)     
        self.demand_prediction = self.get_demand_prediction(date_time)


    def get_demand(self, date_time, next_step = False):
        if self.mode == "data":
            or_demand = self.demand_data.iloc[(date_time.timetuple().tm_yday - 1) * 24 * 60 + date_time.hour * 60 + date_time.minute][['Load']].to_numpy()[0]
            demand = or_demand * self.normalisation_factor       
        elif self.mode == "dummy":
            demand = self.dummy_demand if not next_step else self.dummy_demand_next
        return demand
    
    def simulator_sensor_update(self, device_observations, next_step=False):
        """
        Updates the simulator based on the observation from the true demand.
        Inputs:
            device_observations is a dictionary with the following keys:
                - demand: float (current demand)
                - demand_next: float (next demand)
                - demand_pred: list of floats (predicted demand for the next nb_pred_time_steps time steps)
        """
        if device_observations['obs_type'] == 'state':
            if next_step:
                self.demand = device_observations['demand_next']
                self.dummy_demand = device_observations['demand_next']
            else:
                self.demand = device_observations['demand']
                self.dummy_demand = device_observations['demand']

            self.demand_next = device_observations['demand_next']
            self.dummy_demand_next = device_observations['demand_next']
            self.demand_prediction = device_observations['demand_pred']

        elif device_observations['obs_type'] == 'params':
            self.demand = self.get_demand(device_observations['date_time']['value'])
            self.demand_next = self.get_demand(device_observations['date_time']['value'] + datetime.timedelta(minutes=self.time_step), next_step=True)
            if self.mode == 'dummy':
                self.dummy_demand = self.demand
                self.dummy_demand_next = self.demand_next
            self.demand_prediction = self.get_demand_prediction(device_observations['date_time']['value'])


    def get_demand_prediction(self, date_time):
        if date_time is None:
            date_time = self.date_time

        if self.mode == "data":
            if self.forecast_model == 'ground_truth':
                nb_minutes_in_a_year = 366 * 24 * 60 if (date_time.year % 4 == 0) and (date_time.year % 100 != 0 or date_time.year % 400 == 0) else 365 * 24 * 60 
                x = ((date_time.timetuple().tm_yday - 1) * 24 * 60 + date_time.hour * 60 + date_time.minute)
                x_array = np.arange(x + self.pred_time_step, x + (self.nb_pred_time_steps + 1) * self.pred_time_step, self.pred_time_step) % (nb_minutes_in_a_year)
                demand_array = self.demand_data.iloc[x_array][['Load']].to_numpy()
            
            elif self.forecast_model == 'forecast':
                if date_time.month == 12 and date_time.day == 31 and date_time.hour >= 11 and date_time.minute >= 50:
                    print("Warning: The data is limited until 11h50 AM of the 31st of December. After that, we will just predict the same thing for that day.")
                    date_time = datetime.datetime(date_time.year, 12, 31, 11, 50, 0)            # The data is limited until 11h50 AM of the 31st of December. After that, we will just predict the same thing for that day.
                x = int(date_time.minute%5)
                truncated_dt = date_time - datetime.timedelta(minutes=date_time.minute%5, seconds=date_time.second, microseconds=date_time.microsecond)
                ls = []
                for  t in range(self.nb_pred_time_steps):
                    ls.append(f'+{x + self.pred_time_step * (t + 1)}m')

                demand_array = self.forecast_data.loc[str(truncated_dt), ls].to_numpy().reshape(-1, 1)

            else:
                raise ValueError(f'forecast model {self.forecast_model} not recognized')
            demand_prediction = list(demand_array[:, 0] * self.normalisation_factor)                                                                        
        elif self.mode == "dummy":
            return list([None])   
        
        return demand_prediction
    
    def set_dummy_demand(self, dummy_demand, demand_next):
        self.dummy_demand = dummy_demand
        self.dummy_demand_next = demand_next
        self.demand = dummy_demand
        self.demand_next = demand_next

    def gather_observations(self):
        observations = {}
        observations['demand'] = self.demand
        observations['demand_next'] = self.demand_next
        observations['demand_pred'] = self.demand_prediction
        observations['obs_type'] = 'state'
        return observations


def initialize_demand_data():
    """
    Initializes the demand data.  
    Outputs:
        - demand: pandas DataFrame with the demand data, normalized
    """
    
    norm_demand_path1 = os.path.join('..', 'data', 'norm_demand.parquet')
    norm_demand_path2 = os.path.join('data', 'norm_demand.parquet')

    if os.path.exists(norm_demand_path1):
        print("Demand parquet data found, importing...")
        norm_demand_data = pd.read_parquet(norm_demand_path1)

    elif os.path.exists(norm_demand_path2):
        print("Demand parquet data found, importing...")
        norm_demand_data = pd.read_parquet(norm_demand_path2)

    else:
        raise FileNotFoundError("Demand data files not found. Please check the file paths.")
    
    return norm_demand_data

def initialize_demand_forecast_data():
    """
    Initializes the demand forecast data.  
    We don't denormalize here as it takes ~40s, and we do not need all the data at once during an episode.
    Denormalization is done in the step function.
    Outputs:
        - norm_demand_forecast: pandas DataFrame with the demand forecast data, normalized
    """

    norm_demand_forecast_path1 = os.path.join('..', 'data', 'norm_demand_forecast.parquet')
    norm_demand_forecast_path2 = os.path.join('data', 'norm_demand_forecast.parquet')

    if os.path.exists(norm_demand_forecast_path1):
        print("Demand forecast parquet data found, importing...")
        norm_demand_forecast = pd.read_parquet(norm_demand_forecast_path1)

    elif os.path.exists(norm_demand_forecast_path2):
        print("Demand forecast parquet data found, importing...")
        norm_demand_forecast = pd.read_parquet(norm_demand_forecast_path2)
    else:
        raise FileNotFoundError("Demand forecast data file not found. Please check the file paths.")
    
    return norm_demand_forecast