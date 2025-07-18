try:
    from env.genset_group import GensetGroup
    from env.battery import Battery
    from env.genset_group_controller import GensetGroupController
    from env.battery import Battery
    from env.battery_controller import BatteryController
    from env.windturbine_controller import WindTurbineController
    from env.demand import Demand
    from env.demand_controller import DemandController
    from env.external_world import ExternalWorld
    from env.windturbine import WindTurbine 
    from env.abstract_classes import System
except:
    from genset_group import GensetGroup
    from battery import Battery
    from genset_group_controller import GensetGroupController
    from battery import Battery
    from battery_controller import BatteryController
    from demand_controller import DemandController
    from windturbine_controller import WindTurbineController
    from demand import Demand
    from external_world import ExternalWorld
    from windturbine import WindTurbine 
    from abstract_classes import System


class MicroGrid(System):
    def __init__(self, params, time_step, real, verbose = False):
        const_params = params['const_params']
        init_params = params['init_params']
        self.preliminary_init_params = params['init_params']
        self.real = real
        self.isDevice = False 
        self.time_step = time_step

        self.reset(init_params, real)


    def reset(self, init_params, real):
        ## Initializing external world
        self.external_world = ExternalWorld(init_params['external_world'], self.time_step)
        init_world_obs = self.external_world.gather_observations()
        self.init_date_time = init_world_obs['date_time']


        ## Initializing genset group
        self.genset_group = GensetGroup(init_params['genset_group']['device'], self.time_step, real=real)
        self.genset_group_controller = GensetGroupController(init_params['genset_group'], self.time_step)
        self.genset_group_controller.update_controller_state({'device_observations': self.genset_group.gather_observations()})  # Initialize the genset group controller with the genset group observations

        ## Initializing battery
        self.battery = Battery(init_params['battery']['device'], self.time_step, real=real)
        self.battery_controller = BatteryController(init_params['battery'], self.time_step)
        self.battery_controller.update_controller_state({'device_observations': self.battery.gather_observations()})  # Initialize the battery controller with the battery observations


        ## Initializing demand
        init_params['demand']['device']['init_params']['date_time'] = {'value': self.init_date_time, 'type': 'datetime'}

        if not hasattr(self, 'demand'):
            self.demand = Demand(init_params['demand']['device'], self.time_step, real=real)
        else:
            self.demand.reset(init_params['demand']['device']['init_params'])  # If demand exists, we do not make a new instance, because data loading for the demand device (not sim) leads to long instanciation time.

        self.demand_controller = DemandController(init_params['demand'], self.time_step)
        self.demand_controller.update_controller_state({'device_observations': self.demand.gather_observations()})  # Initialize the demand controller with the demand observations

        ## Initializing wind turbine 
        if 'wind_turbine' in init_params.keys() and init_params['wind_turbine']['device']['init_params']['active']['value']:
            init_params['wind_turbine']['device']['init_params']['date_time'] = {'value': self.init_date_time, 'type': 'datetime'}
            if not hasattr(self, 'wind_turbine'):
                self.wind_turbine = WindTurbine(init_params['wind_turbine']['device'], time_step=self.time_step, real=real)
            else:
                self.wind_turbine.reset(init_params['wind_turbine']['device']['init_params'])  # If wind_turbine exists, we do not make a new instance, because data loading for the turbine device (not sim) leads to long instanciation time.
            self.wind_turbine_controller = WindTurbineController(init_params['wind_turbine'], self.time_step)
            self.wind_turbine_controller.update_controller_state({'device_observations': self.wind_turbine.gather_observations()})       # Initialize the wind turbine controller with the wind turbine observations
        else:
            inactive_wind_turbine_params = {'device': {'init_params':{'active': {'value': False}}, 'const_params': {'nominal_power': {'value': 0}}}, 'controller': {'const_params': {}, 'init_params': {}}}
            self.wind_turbine = WindTurbine(inactive_wind_turbine_params['device'], time_step=self.time_step, real=real)
            self.wind_turbine_controller = WindTurbineController(inactive_wind_turbine_params, self.time_step)

        return self.gather_observations()

    def simulator_sensor_update(self, device_observations):
        """
         Only called if this is a simulator. Updates the state of the simulator based on real device observations.
        """        
        if self.real:
            raise ValueError("This function should only be called if the microgrid is a simulator")
        
        # Update the controllers with the device observations
        self.battery_controller.update_controller_state(device_observations['battery'])
        self.genset_group_controller.update_controller_state(device_observations['genset_group'])
        self.wind_turbine_controller.update_controller_state(device_observations['wind_turbine']) 
        self.demand_controller.update_controller_state(device_observations['demand'])

        # Update the devices with the device observations
        self.demand.simulator_sensor_update(device_observations['demand']['device_observations'])
        self.battery.simulator_sensor_update(device_observations['battery']['device_observations'])
        self.genset_group.simulator_sensor_update(device_observations['genset_group']['device_observations'])
        self.wind_turbine.simulator_sensor_update(device_observations['wind_turbine']['device_observations'])
        
        setattr(self.external_world, 'date_time', device_observations['external_world']['date_time'])

    
    def step(self, action, reserve_available = True, verbose = False):
        """
        Implements one step of the microgrid environment
        Inputs:
            - action: dictionary with the following keys
                - genset_group: genset group action:
                    - status_change: "start_next" -> start next genset in the priority list; stop_last -> stop last genset in the priority list; none -> do nothing
                - battery: battery action
                    - p_grid: power to charge/discharge the battery on the grid side, in kW. p_grid > 0 -> discharging the battery; p_grid < 0 -> charging the battery
        Returns:
            - observations: dictionary with the following keys
                - genset_group: genset group observations:
                    - genset_group_active_power: in kW
                    - genset_group_fuel_consumption: in l/h
                    - gensets: dictionary of genset with ids (int) as keys 
                        - ids : dictionary with the following keys for each genset:
                            - running: Boolean
                            - warmup: Boolean
                            - cooldown: Boolean
                            - overload: Boolean
                            - time_since_warmup: int,  in minutes
                            - time_since_cooldown: int, in minutes
                            - time_since_start: int, in minutes
                            - active_power: float, in kW
                            - available_active_power: float, in kW
                            - average_active_power: float, in kW
                            - fuel_consumption: float, in l/h
                - battery: battery observations:
                    - soc: state of charge: float, in ratio from 0 to 1)
                    - p_grid: power charging/discharging the battery on the grid side: float, in kW (positive for discharging, negative for charging)
                    - available_power_charge: float, in kW
                    - available_power_discharge: float, in kW
                    - degradation_cost: float, in monetary unit
                    - soc_sp: list, state of charge switching points, in ratio from 0 to 1
                - demand: demand observations:
                    - demand: float, in kW
                    - demand_pred: predicted demand, list of floats, in kW
                - wind_turbine: wind turbine observations:
                    - available_wind_power: Available wind power, in kW
                    - turbine_setpoint: power setpoint, in kW
                    - wind_power: Wind power, in kW
                    - available_wind_power_pred: Available wind power prediction, list, in kW
                - external_world: external world observations:
                    - date_time: datetime object
                    - time_step: int, in minutes
                - microgrid: microgrid observations:
                    - balance: float, in kW
        """
        if verbose:
            print("---- MICROGRID STEP ----")
            print("Received action: {}".format(action))

        external_world_obs = self.external_world.step(verbose = verbose)
        date_time = external_world_obs['date_time']

        action['wind_turbine']['date_time'] = date_time
        action['demand']['date_time'] = date_time

        # Compute safe actions
        battery_action = action['battery']
        valid_battery_action = self.battery_controller.generate_safe_action(battery_action, reserve_available=reserve_available)
        valid_genset_group_action = self.genset_group_controller.generate_safe_action(action['genset_group'], reserve_available=reserve_available)
        valid_wind_turbine_action = self.wind_turbine_controller.generate_safe_action(action['wind_turbine'])

        # Step the elements
        self.demand.step(action['demand'])
        self.battery.step(valid_battery_action, verbose = verbose)  
        self.genset_group.step(valid_genset_group_action, reserve_available, verbose = verbose)
        self.wind_turbine.step(valid_wind_turbine_action, verbose = verbose)

        # Update the controllers with the device observations
        self.demand_controller.update_controller_state({'device_observations': self.demand.gather_observations()})
        self.battery_controller.update_controller_state({'device_observations': self.battery.gather_observations()})
        self.genset_group_controller.update_controller_state({'device_observations': self.genset_group.gather_observations()})
        self.wind_turbine_controller.update_controller_state({'device_observations': self.wind_turbine.gather_observations()})


        #observations = self.gather_observations()
        #return observations

    def gather_observations(self):
        external_world_obs = self.external_world.gather_observations()
        date_time = external_world_obs['date_time']
        genset_group_obs = self.genset_group_controller.gather_observations()
        battery_obs = self.battery_controller.gather_observations()
        demand_obs = self.demand_controller.gather_observations()
        wind_turbine_obs = self.wind_turbine_controller.gather_observations()

        #microgrid_obs = {
        #    'balance': battery_obs['device_observations']['p_grid'] + genset_group_obs['device_observations']['genset_group_active_power'] + wind_turbine_obs['device_observations']['wind_power'] - demand_obs['device_observations']['demand'],
        #}
        
        observations = {
            'external_world': external_world_obs,
            'genset_group': genset_group_obs,
            'battery': battery_obs,
            'demand': demand_obs,
            'wind_turbine': wind_turbine_obs,
            'balance': battery_obs['device_observations']['p_grid'] + genset_group_obs['device_observations']['genset_group_active_power'] + wind_turbine_obs['device_observations']['wind_power'] - demand_obs['device_observations']['demand']
        }

        return observations
    
