
from abc import ABC, abstractmethod
import copy
try:
    from .utils import add_params_noise
except:
    from utils import add_params_noise

class System(ABC):
    def __init__(self, params, time_step, real):

        const_params = params['const_params']
        init_params = params['init_params']
        self.time_step = time_step

        # Randomize parameters for the system if relevant
        self.real = real  # True if the system is a real system, False if it is a simulator
        #self.device = TO FILL  # True if the system is a device, False if it is a system with subcomponents (e.g., a microgrid with devices)

        if self.real and const_params['noise_level']['value'] not in ['None', 'none']:
            noise_level = const_params['noise_level']['value']
            print(f'Adding {noise_level} noise to DEVICE')
            for param in ["prime_power_rating"]:
                print(f'Original {param}: {const_params[param]["value"]}')
                const_params[param]['value'] = add_params_noise(const_params[param]['value'], const_params['noise_level']['value'], const_params['noise_range_percentage']['value'])
                print(f'Noisy {param}: {const_params[param]["value"]}')        
        
        # Set fixed parameters
        ## self.param1 = const_params['param1']['value']  # Example of a constant parameter

        # Initialize the instance
        self.reset(init_params, real)

    def step(self, action):
        # Update the device state based on the action taken
        pass

    def reset(self, init_params, real):
        # Reset the device to its initial state
        self.real = real

        # Set self initial parameters
        ## self.param2 = init_params['param2']['value']  # Example of an initial parameter

        # Do other necessary reset actions

        # Gather observations
        obs = self.gather_observations()

        return obs

    def gather_observations(self):
        # Gather the current state of the device
        state = {
            'obs_type': 'state',
            # Add other relevant state information here
            ## 'state1': self.state1,  # Example of a state dimension to be observed
        }
        return state

    def simulator_sensor_update(self, device_observations):
        """
        Updates the simulator based on the observation from the true device.
        This is only useful in case this is not a "device" but a digital twin/simulator.
        Inputs:
            device_observations is a dictionary with the current state of the device.
        """
        if self.real:
            raise ValueError("This function should only be called if the device is a simulator")

        # Update the simulator parameters based on the observations
        if device_observations['obs_type'] == 'state':
            # Update the simulator state based on the device observations
            pass
        elif device_observations['obs_type'] == 'params':
            # Update the simulator state based on the initial parameters (need to get the ['value'] of each parameter)
            pass




class Controller(ABC):
    def __init__(self, device_params, time_step):
        controller_const_params = device_params['controller']['const_params']
        self.time_step = time_step

        # Initializing the simulator
        self.device_sim = Device(device_params['device'], self.time_step, device=False)  # Device simulator

        # Constant parameters for the controller
        ## self.param1 = controller_const_params['param1']['value']  # Example of a constant parameter

        self.reset(device_params['controller']['init_params'], device_params['device']['init_params'])  # Resetting the controller and the simulator

    def reset(self, controller_init_params, device_init_params):
        """
        Resets the controller to a new episode. Updates the simulator based on the device's initial parameters.
        """
        # Re-initialize the controller state if necessary
        ## self.param2 = controller_init_params['param2']['value']  # Example of an initial parameter

        # Resetting the simulator:
        self.device_sim.simulator_sensor_update(device_init_params, device = False)

    def update_controller_state(self, observations):
        """
        Updates the simulator based on the observation from the true device.
        Inputs:
            observations is a dictionary with the current state of the device.
        """

        # Update simulator state
        self.device_sim.simulator_sensor_update(observations['device_observations'])

        # Update controller state if necessary
        if 'controller_state' in observations:
            controller_state = observations['controller_state']
            if controller_state['obs_type'] == 'state':
                # Update the controller state based on the device observations
                ## self.state1 = controller_state['state1']  # Example of a controller state to be updated
                pass
            elif controller_state['obs_type'] == 'params':
                # Update the controller state based on the initial parameters
                ## self.state1 = controller_state['state1']['value']  # Example of a controller state to be updated
                pass

    def simulator_dynamic_update(self, valid_action):
        """
        Updates the simulator based on the action taken by the controller.
        Inputs:
            action is a dictionary with the action to be taken by the controller.
        """

        valid_action = self.generate_safe_action(valid_action)
        self.device_sim.step(valid_action)
        obs = self.device_sim.gather_observations()
        return obs
    
    def gather_observations(self):
        """
        Gathers the observations from the device simulator.
        Outputs:
            observations is a dictionary with the current state of the device.
        """
        device_observations = self.device_sim.gather_observations()

        controller_state = {
            'obs_type': 'state',
            # Add other relevant controller state information here
            ## 'state1': self.state1,  # Example of a controller state to be observed
        }

        observations = {'device_observations': device_observations, 'controller_state': controller_state}
        return observations
    
    def generate_safe_action(self, action):
        """
        Generates a safe action based on the input action.
        This function can be overridden in subclasses to implement specific safety checks.
        Inputs:
            action is a dictionary with the action to be taken by the controller.
        Outputs:
            valid_action is a dictionary with the safe action to be taken by the controller.
        """
        # For now, we assume all actions are valid
        valid_action = action
        return valid_action
    
    def predict_step(self, action):
        """
        Predicts the next step of the device simulator based on the given action.
        Inputs:
            action is a dictionary with the action to be taken by the controller.
        Outputs:
            predicted_observations is a dictionary with the predicted state of the device.
        """
        # Copy the current state of the simulator
        copy_self = copy.deepcopy(self)
        predicted_observations = copy_self.simulator_dynamic_update(action)
        return predicted_observations