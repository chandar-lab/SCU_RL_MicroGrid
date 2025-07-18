try:
    from env.abstract_classes import Controller
    from env.microgrid import MicroGrid
except:
    from abstract_classes import Controller
    from microgrid import MicroGrid
import copy
import numpy as np

class MicrogridInitializationError(Exception):
    pass

class MicroGridController(Controller):
    def __init__(self, microgrid_params, time_step, verbose=False):


        microgrid_controller_const_params = microgrid_params['controller']['const_params']
        microgrid_controller_init_params = microgrid_params['controller']['init_params']

        self.check_initialization = microgrid_controller_const_params['check_initialization']['value']
        self.init_check_steps = microgrid_controller_const_params['init_check_steps']['value']
        self.init_check_loops = microgrid_controller_const_params['init_check_loops']['value']
        self.wind_turbine_priority = microgrid_controller_const_params['wind_turbine_priority']['value']
        self.check_balance = microgrid_controller_const_params['check_balance']['value']
        self.check_reserve = microgrid_controller_const_params['check_reserve']['value']
        self.adjust_battery = microgrid_controller_const_params['adjust_battery']['value']
        self.conservativeness_coeff = microgrid_controller_const_params['conservativeness_coeff']['value']
        self.min_available_wind_power = microgrid_controller_const_params['min_available_wind_power']['value']
        self.max_available_wind_power_drop_step = microgrid_controller_const_params['max_available_wind_power_drop_step']['value']
        self.max_demand = microgrid_controller_const_params['max_demand']['value']
        self.max_demand_increase_step = microgrid_controller_const_params['max_demand_increase_step']['value']

        self.time_step = time_step


        # Initialization of the simulator

        sim_microgrid_params = copy.deepcopy(microgrid_params)
        if 'wind_turbine' in sim_microgrid_params['device']['init_params']:
            sim_microgrid_params['device']['init_params']['wind_turbine']['device']['const_params']['mode']['value'] = 'dummy'
        sim_microgrid_params['device']['init_params']['demand']['device']['const_params']['mode']['value'] = 'dummy'
        self.microgrid_sim = MicroGrid(sim_microgrid_params['device'], time_step = self.time_step, real = False)

        self.copy_external_world = copy.deepcopy(self.microgrid_sim.external_world)


    def reset(self, microgrid_controller_init_params, microgrid_device_init_params):
        """
        Resets the controller and its simulator based on the initial parameters. 
        """
        # There is no init controller parameters to reinitialize

        # Reset the simulator with initial parameters
        self.microgrid_sim.reset(microgrid_device_init_params, real = False)

        # Check initialization validity and return an error if not initialized correctly
        
        if self.check_initialization == True:         
            verified = self.verify_initialization_validity()
            if not verified:
                raise MicrogridInitializationError("The microgrid is not initialized correctly. Check the initial parameters to ensure it is possible to recover from the initial state.")
        else:
            verified = True

        return verified

    def check_valid(self, processed_action):
        pass

    def update_controller_state(self, observations):

        # Update the simulator state
        self.microgrid_sim.simulator_sensor_update(observations['device_observations'])

        # Update the state of the copy of external world (useful for the controller)
        setattr(self.copy_external_world, 'date_time', observations['device_observations']['external_world']['date_time'])

    def simulator_dynamic_update(self, action, reserve_available=True, verbose = False):
        valid_action = self.generate_safe_action(action, verbose = verbose)
        self.microgrid_sim.step(valid_action, reserve_available, verbose = verbose)
        observations = self.gather_observations()
        return observations

    def gather_observations(self):

        device_observations = self.microgrid_sim.gather_observations()
        controller_state = {}

        observations = {'device_observations': device_observations, 'controller_state': controller_state}
        return observations

    def generate_safe_action(self, action, verbose=False):
        
        # Update external world
        external_world_obs = self.copy_external_world.step(verbose = verbose)
        date_time = external_world_obs['date_time']

        action['demand'] = {'date_time': date_time}
        action['wind_turbine'] = {'date_time': date_time}

        # Ensure recoverability
        if self.check_balance or self.check_reserve:
            action = self.shield_ensure_recoverability(action, verbose = verbose)     # This will try to avoid having to use the reserve in the next time steps.
            if verbose:
                print("Action after recoverability shield: {}".format(action))



        # Get the demand and the reserve
        obs = self.microgrid_sim.gather_observations()
        demand = obs['demand']['device_observations']['demand_next']

        # Compute the turbine setpoint to complete the demand after predicting the batteries production and considering the minimal genset production.
        if self.microgrid_sim.wind_turbine.active:
            action = self.compute_turbine_setpoint(action, demand,  reserve_available = True, verbose = verbose)     # Assume reserve available here.
            if verbose:
                print("   Action after computing turbine setpoint: {}".format(action))
        else:
            action['wind_turbine'] = {'turbine_setpoint': 0}


       # Compute the genset setpoint to complete the demand after predicting the batteries production. First, try to avoid using overload.
        action = self.compute_genset_setpoint(action, demand, reserve_available = False, verbose = verbose)
        if verbose:
            print("   Action after computing genset setpoint without reserve_available: {}".format(action))


        # Shield 1 - Check if the genset setpoint is enough to complete the demand, adjust the battery if necessary. Allow using battery reserve if necessary.
        if self.adjust_battery == True:
            action = self.shield_check_battery_adjustment(action, demand, reserve_available = True, verbose = verbose)
            if verbose:
                print("   Action after shield step 1: {}".format(action))

        # Compute the genset setpoint to complete the demand after predicting the new batteries production. Now, allow using overload if necessary and available to the microgrid.
        action = self.compute_genset_setpoint(action, demand, reserve_available = True, verbose = verbose)
        if verbose:
            print("   Action after computing genset setpoint: {}".format(action))

        valid_action = action

        return valid_action

    def compute_genset_setpoint(self, action, demand, reserve_available, verbose = False):
        """
        Set the genset setpoint to complete the demand after predicting the batteries and wind turbine production
        """
        shielded_action = copy.deepcopy(action)
        battery_obs_pred = self.microgrid_sim.battery_controller.predict_step({'p_grid': action['battery']['p_grid']}, reserve_available)
        battery_expected_power = battery_obs_pred['device_observations']['p_grid']

        # Compute the maximum power available from the genset group
        max_power_action = {'status_change': shielded_action['genset_group']['status_change'],
                             'power_setpoint': np.inf}
        pred_genset_group_controller_obs = self.microgrid_sim.genset_group_controller.predict_step(max_power_action, reserve_available, verbose = verbose)
        max_genset_power = pred_genset_group_controller_obs['device_observations']['genset_group_active_power']

        if self.microgrid_sim.wind_turbine.active:
            wind_turbine_obs_pred = self.microgrid_sim.wind_turbine_controller.predict_step(action['wind_turbine'], next_step = True)
            wind_turbine_expected_power = wind_turbine_obs_pred['device_observations']['wind_power']
        else:
            wind_turbine_expected_power = 0

        # Compute the power needed from the genset
        genset_power_setpoint = min(demand - battery_expected_power - wind_turbine_expected_power, max_genset_power)

        shielded_action['genset_group']['power_setpoint'] = genset_power_setpoint
        if verbose:
            print("The genset setpoint is set at {} kW to complete the wind turbine production and the battery power command.".format(genset_power_setpoint))

        return shielded_action

    def compute_turbine_setpoint(self, action, demand, reserve_available, verbose = False):
        """
        Set the turbine setpoint to complete the demand after predicting the batteries production and considering the minimal genset group production given its configuration. 
        If wind_turbine_priority, the wind turbine will be greedily used before a discharging battery.
        """
        shielded_action = copy.deepcopy(action)
        if verbose:
            print("compute_turbine_setpoint - The original action is {}.".format(shielded_action))
            print("compute_turbine_setpoint - The demand is {} kW.".format(demand))

        battery_obs_pred = self.microgrid_sim.battery_controller.predict_step({'p_grid': action['battery']['p_grid']}, reserve_available)
        battery_expected_power = battery_obs_pred['device_observations']['p_grid']

        if verbose:
            print("compute_turbine_setpoint - The expected battery power is {} kW.".format(battery_expected_power))

        low_genset_action = shielded_action['genset_group']
        low_genset_action['power_setpoint'] = 0
        min_genset_group_obs_pred = self.microgrid_sim.genset_group_controller.predict_step(low_genset_action, reserve_available=reserve_available)
        min_genset_group_expected_power = min_genset_group_obs_pred['device_observations']['genset_group_active_power']
        if verbose:
            print("compute_turbine_setpoint - The expected minimal genset group power is {} kW.".format(min_genset_group_expected_power))

        if battery_expected_power <= 0 or self.wind_turbine_priority == False:               # If battery is charging, wind power setpoint must consider it. If battery is discharging, without wind_turbine_priority, wind power setpoint must also consider it. 
            wind_turbine_power_setpoint = demand - battery_expected_power - min_genset_group_expected_power
        elif battery_expected_power >= 0 and self.wind_turbine_priority:
            wind_turbine_power_setpoint = demand - min_genset_group_expected_power          # If battery is discharging, and we give priority to wind turbine power, we do not consider the battery's setpoint and we are greedy regarding the wind power. The battery setpoint will be later corrected.
            if verbose:
                print("Priority is given to windturbine over battery discharging.")
        shielded_action['wind_turbine'] = {'turbine_setpoint' : wind_turbine_power_setpoint, 'date_time': action['wind_turbine']['date_time']}
        if verbose:
            print("The wind turbine setpoint is set at {} kW to complete the battery power command.".format(wind_turbine_power_setpoint))

        return shielded_action


    def shield_ensure_recoverability(self, action, verbose = False):
        """
        Prevent the turning OFF or force the turning ON of a genset if waiting for one more time step to turn on would lead to a negative balance in the next warmup + cooldown period, assuming agent discharging the battery. 
        This shield assumes that the predictions are ground truth, but does not account for power reserve. The generators should be able to provide without going in overload (as this is kept for the reserve).
        """

        shielded_action = copy.deepcopy(action)
        balanced_sl, reserve_balanced_sl = True, True
        if not self.check_balance and not self.check_reserve:
            return shielded_action

        start_next_action = {
            'genset_group': {'status_change': 'start_next'},    # Start the next genset at every time step
            'battery': {'p_grid': np.inf}                       # Because the shield does not control the battery, we must assume the worst: the agent will discharge the battery as much as possible
            # 'wind_turbine': {'turbine_setpoint': 0}              # The wind turbine will not be used to complete the demand
        }
            
        nb_steps = self.microgrid_sim.genset_group.max_warmup_time + self.microgrid_sim.genset_group.max_cooldown_time

        if shielded_action['genset_group']['status_change'] == 'stop_last':          # If the action was to turn off, check if it will lead to a negative balance even if we go all in with turning ON later
            next_actions = [shielded_action] + [start_next_action] * (nb_steps-1)
            balanced_sl = self.check_pos_balance_n_steps(nb_steps, next_actions, reserve_available = False, verbose = verbose) if self.check_balance else True
            reserve_balanced_sl = self.check_pos_balance_n_steps(nb_steps, next_actions, reserve_available = True, verbose = verbose) if self.check_reserve else True

            if (not balanced_sl) or (not reserve_balanced_sl):
                shielded_action['genset_group']['status_change'] = 'none'
                if verbose:
                    print("Shield - 'stop_last' action would not be recoverable in the next {} steps due to: balanced = {}, reserve_balanced = {}. The action was overridden to 'none'.".format(nb_steps, balanced_sl, reserve_balanced_sl))
            else:
                if verbose:
                    print("Shield - 'stop_last' action would be recoverable in the next {} steps. The action was allowed.".format(nb_steps))
                return shielded_action
            

        if shielded_action['genset_group']['status_change'] == 'none':         # If the action was to do nothing, check if it will lead to a negative balance even if we go all in with turning ON later
            next_actions = [shielded_action] + [start_next_action] * (nb_steps-1)
            balanced_n = self.check_pos_balance_n_steps(nb_steps, next_actions, reserve_available = False, verbose = verbose) if self.check_balance else True
            reserve_balanced_n = self.check_pos_balance_n_steps(nb_steps, next_actions, reserve_available = True, verbose = verbose) if self.check_reserve else True
                        
            if (not balanced_n) or (not reserve_balanced_n):
                shielded_action['genset_group']['status_change'] = 'start_next'
                if verbose:
                    print("Balanced: ", balanced_n)
                    print("Reserve balanced: ", reserve_balanced_n)
                    print("Shield - 'none' action would not be recoverable in the next {} steps due to: balanced = {}, reserve_balanced = {}. The action was overridden to 'start_next'.".format(nb_steps, balanced_n, reserve_balanced_n))
            else:
                if verbose:
                    print("Balanced: ", balanced_n)
                    print("Reserve balanced: ", reserve_balanced_n)
                    print("Shield - 'none' action would be recoverable in the next {} steps. The action was allowed.".format(nb_steps))
        
        return shielded_action

    def shield_check_battery_adjustment(self, action, demand, reserve_available = True, verbose = False):
        """
        If necessary, adjust the battery action based on the predicted genset power production
        """

        old_p_grid = action['battery']['p_grid']
        shielded_action = copy.deepcopy(action)      
        
        # Compute the expected power from the genset
        balance_pred, genset_group_expected_power, _, wind_turbine_power = self.compute_expected_balance(action, demand, reserve_available)

        # Adjust the battery action if the balance is lower than zero
        if balance_pred != 0:          
            shielded_action['battery']['p_grid'] = demand - genset_group_expected_power - wind_turbine_power
            if verbose:
                print("Shield - The battery power was readjusted from {} to {} kW to prevent a balance of {}.".format(old_p_grid, shielded_action['battery']['p_grid'], balance_pred))

        return shielded_action

    def verify_initialization_validity(self, verbose = False):
        """
        Check if the microgrid's initial state is ensured to be recoverable relying only on the shield. If not, try to reset the microgrid until it can.
        """
        i = 0

        start_next_action = {
            'genset_group': {'status_change': 'start_next'},     # Start the next genset at every time step
            'battery': {'p_grid': np.inf}                        # Because the shield does not control the battery, we must assume the worst: the agent will discharge the battery as much as possible
        }

        next_actions = [start_next_action] * self.init_check_steps

        balanced = self.check_pos_balance_n_steps(self.init_check_steps, next_actions, reserve_available = False, verbose = verbose) if self.check_balance else True
        reserve_balanced = self.check_pos_balance_n_steps(self.init_check_steps, next_actions, reserve_available = True, verbose = verbose) if self.check_reserve else True

        return balanced and reserve_balanced
    
    def check_pos_balance_n_steps(self, nb_steps, next_actions, reserve_available, verbose = False):
        """
        Verify that the balance in the next n steps is not negative, based on predictions and a list of actions.
        genset_overload_available should be set to False when the recoverability should not rely on the genset overload capacity.
        """
        observations_list = self.predict_n_steps(nb_steps, next_actions, reserve_available, verbose = verbose)
        
        balanced = True
        
        for i in range(nb_steps):
            balance_i = observations_list[i]['device_observations']['balance']

            if verbose:
                print("Reserve balance at step {}: {}".format(i, balance_i))

            if balance_i < -0.05:      # If the balance is negative, the microgrid is not recoverable. Give a 0.05 margin to avoid numerical errors from rounding the gensets group.        
                balanced = False
                break
        if balanced:
            if verbose:
                print("The balance of the microgrid is ensured, the balance is 0 or positive in the next {} steps.".format(nb_steps))
            return True
        else:
            if verbose:
                print("The balance of the microgrid is not respected, the balance is negative in the next {} steps.".format(nb_steps))
            return False

    def compute_expected_balance(self, action, demand, reserve_available = True):

        #external_world_obs = self.copy_external_world.gather_observations()

        battery_obs_pred = self.microgrid_sim.battery_controller.predict_step(action['battery'], reserve_available)
        battery_expected_power = battery_obs_pred['device_observations']['p_grid']

        genset_group_obs_pred = self.microgrid_sim.genset_group_controller.predict_step(action['genset_group'], reserve_available)
        genset_group_expected_power = genset_group_obs_pred['device_observations']['genset_group_active_power']

        if self.microgrid_sim.wind_turbine.active:
            wind_turbine_obs_pred = self.microgrid_sim.wind_turbine_controller.predict_step(action['wind_turbine'], next_step = True)
            wind_turbine_expected_power = wind_turbine_obs_pred['device_observations']['wind_power']
        else:
            wind_turbine_expected_power = 0

        balance_pred = battery_expected_power + genset_group_expected_power + wind_turbine_expected_power - demand

        return balance_pred, genset_group_expected_power, battery_expected_power, wind_turbine_expected_power


    def predict_n_steps(self, n, actions, reserve_available = True, verbose = False):
        """
        Makes n fake steps to predict the next observations
        """
        if verbose:
            print("---------------------- PREDICTING STEPS ----------------------")
        observations_pred_list = []

        copy_self = copy.deepcopy(self)
        copy_self.check_balance = False
        copy_self.check_reserve = False

        for i in range(n):
            # Set demand and available_wind_power to predetermined values if in parameters - otherwise, predict_n_steps will use the actual future values in its prediction       
                                      
            ## Demand
            if reserve_available:       # If on reserve mode, check worst case scenario. 
                demand = np.minimum(self.microgrid_sim.demand.demand_next + (i+1)*self.conservativeness_coeff*self.max_demand_increase_step, self.max_demand)     # Increase the maximum possible demand at each step
                demand_next = np.minimum(self.microgrid_sim.demand.demand_next + (i+2)*self.conservativeness_coeff*self.max_demand_increase_step, self.max_demand)     # Increase the maximum possible demand at each step
            else:                       # Otherwise, check using current demand (next because we are looking at incoming time step).
                demand = self.microgrid_sim.demand.demand_next
                demand_next = demand
            # Override function in the microgrid copy_self
            copy_self.microgrid_sim.demand.set_dummy_demand(demand, demand_next)
            copy_self.microgrid_sim.demand_controller.demand_sim.set_dummy_demand(demand, demand_next)

            ## Wind turbine
            if self.microgrid_sim.wind_turbine.active:
                if reserve_available:       # If on reserve mode, check worst case scenario.
                    available_wind_power = np.maximum(self.microgrid_sim.wind_turbine.available_power_next - (i+1)*self.conservativeness_coeff*self.max_available_wind_power_drop_step*self.microgrid_sim.wind_turbine.nominal_power, self.min_available_wind_power)
                    available_wind_power_next = np.maximum(self.microgrid_sim.wind_turbine.available_power_next - (i+2)*self.conservativeness_coeff*self.max_available_wind_power_drop_step*self.microgrid_sim.wind_turbine.nominal_power, self.min_available_wind_power)
                else:                       # Otherwise, check using current wind power (next because we are looking at incoming time step).
                    available_wind_power = self.microgrid_sim.wind_turbine.available_power_next
                    available_wind_power_next = available_wind_power

                copy_self.microgrid_sim.wind_turbine.set_dummy_power(available_wind_power, available_wind_power_next)
                copy_self.microgrid_sim.wind_turbine_controller.windturbine_sim.set_dummy_power(available_wind_power, available_wind_power_next)
            
            
            if verbose:
                print(" ----- Prediction step: {} -----".format(i))
                print("Time: ", self.microgrid_sim.external_world.date_time)
                print("Demand: ", demand, "Next demand: ", demand_next)

            observations_pred = copy_self.simulator_dynamic_update(actions[i], reserve_available, verbose = verbose)
            observations_pred_list.append(observations_pred)
        
        if verbose:
            print("---------------------- PREDICTION DONE ----------------------")
        return observations_pred_list

    def predict_step(self, action, reserve_available, verbose = False):
        copy_self = copy.deepcopy(self)
        copy_self.check_balance = False
        copy_self.check_reserve = False
        observations_pred = copy_self.simulator_dynamic_update(action, reserve_available, verbose = verbose)
        return observations_pred
    