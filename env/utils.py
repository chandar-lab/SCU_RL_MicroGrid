import random
import numpy as np

def add_params_noise(param, noise_level, noise_range_percentage):

    if noise_range_percentage  > 0.33:
        raise ValueError("Invalid noise range percentage. It cannot be larger than 0.33")
    
    range_value = noise_range_percentage * abs(param)

    if noise_level is None or noise_level == 'none':
        noise = 0  # no noise
    elif noise_level == 'low':
        noise = random.uniform(-0.5 * range_value, 0.5 * range_value)  # Low noise level
    elif noise_level == 'medium':
        noise = random.uniform(-1 * range_value, 1 * range_value)  # Medium noise level
    elif noise_level == 'high':
        noise = random.uniform(-2 * range_value, 2 * range_value)  # High noise level
    else:
        raise ValueError("Invalid noise level. Choose from 'low', 'medium', or 'high'.")
    
    return param + noise


def set_init_param(init_param, data_type, random_parameters=None):
    """
    Set initial parameter based on type and randomization parameters.
    
    Args:
        init_param: The initial parameter value or 'random'/'random_list'
        data_type: The data type ('integer', 'float', 'boolean', 'string', 'date')
        random_parameters: Dictionary containing randomization parameters:
            - min/max: For numeric types when init_param is 'random'
            - elements: List of possible values when init_param is 'random_list'
    
    Returns:
        The initialized parameter value
    """
    # Default random_parameters if None
    if random_parameters is None:
        random_parameters = {}
        
    # Standardize data type
    if data_type == 'integer':
        data_type = 'int'
    elif data_type == 'boolean':
        data_type = 'bool'
    
    # Handle random initialization
    if init_param == 'random':
        if data_type == 'int':
            min_val = random_parameters.get('min')
            max_val = random_parameters.get('max')
            param = np.random.randint(min_val, max_val + 1)
        elif data_type == 'float':
            min_val = random_parameters.get('min')
            max_val = random_parameters.get('max')
            param = np.random.uniform(min_val, max_val)
        elif data_type == 'bool':
            param = bool(np.random.choice([True, False]))
        elif data_type == 'date':
            from datetime import datetime
            from dateutil import parser
            
            min_date = random_parameters.get('min')
            max_date = random_parameters.get('max')
            
            # Convert string dates to timestamps
            if isinstance(min_date, str):
                min_date = parser.parse(min_date).timestamp()
            if isinstance(max_date, str):
                max_date = parser.parse(max_date).timestamp()
                
            # Sample random timestamp and convert back to datetime
            random_ts = np.random.uniform(min_date, max_date)
            param = datetime.fromtimestamp(random_ts).strftime('%Y-%m-%d %H:%M:%S')
        else:
            raise ValueError(f"Unsupported data type for random sampling: {data_type}")
            
    # Handle random sampling from a list
    elif init_param == 'random_list':
        elements = random_parameters['elements']
        if not elements:
            raise ValueError("Elements list must be provided for random_list initialization")
        param = np.random.choice(elements)
        
        # Convert the param to the specified type if needed
        if data_type == 'int':
            param = int(param)
        elif data_type == 'float':
            param = float(param)
        elif data_type == 'string':
            # For string type, ensure the value remains a string without conversion
            param = str(param)       
        elif data_type == 'bool':
            if isinstance(param, str):
                param = param.lower() == 'true'
            else:
                param = bool(param)
    # Use the provided value directly
    else:
        param = init_param
        
        # Convert the param to the specified type if needed
        if data_type == 'int':
            param = int(param)
        elif data_type == 'float':
            param = float(param)
        elif data_type == 'string':
            # For string type, ensure the value remains a string without conversion
            param = str(param)     
        elif data_type == 'bool':
            if isinstance(param, str):
                param = param.lower() == 'true'
            else:
                param = bool(param)
           
    return param



def sample_init_parameters(params):
    """
    Set the initial parameters of the microgrid components by processing the randomization.
    
    Recursively walks through the parameter dictionary and replaces any 'random' values
    with appropriate samples based on the specified type, min/max values or list of options.
    
    Args:
        params: Dictionary containing microgrid parameters from config.yaml
    
    Returns:
        Dictionary with same structure but with all 'random' values sampled according to their specifications
    """

    if not isinstance(params, dict):
        return params
    
    if isinstance(params, list):
        for i, item in enumerate(params):
            if isinstance(item, dict):
                sample_init_parameters(item)  # Recursive call for each item in the list
                params[i] = item
        return params  # Return the modified list

    for key, value in list(params.items()):  # Iterate over a copy of the keys
        # If the value is a dictionary but not a parameter specification (has 'value' key)
        if isinstance(value, dict) and 'value' not in value:
            sample_init_parameters(value)  # Call recursively, modifying in place
        # If the value is a list
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    sample_init_parameters(item)
            params[key] = value # Call recursively for each element of the list        # If the value is a parameter specification
        elif isinstance(value, dict) and 'value' in value:
            param_value = value['value']
            param_type = value.get('type', 'float')
            
            # Build the random_parameters dictionary with the appropriate keys
            random_parameters = {}
            if 'min' in value:
                random_parameters['min'] = value['min']
            if 'max' in value:
                random_parameters['max'] = value['max']
            if 'elements' in value:
                random_parameters['elements'] = value['elements']
                
            # Process the parameter and update ONLY the 'value' field while preserving the dictionary
            sampled_value = set_init_param(param_value, param_type, random_parameters)
            value['value'] = sampled_value  # Update only the 'value' key
            params[key] = value  # Keep the original dictionary structure
        else:
            # For non-dictionary values, copy as-is (no change needed)
            pass

    return params


def check_genset_running(params):
    """
    Ensure the controller parameters are set correctly for the genset running status.
    """
    for i in range(params['device']['init_params']['genset_group']['device']['const_params']['n_gensets']['value']):
        if params['device']['init_params']['genset_group']['device']['init_params']['gensets'][i]['device']['init_params']['running']['value']:  # If it is running
            params['device']['init_params']['genset_group']['device']['init_params']['gensets'][i]['controller']['init_params']['status']['value'] = np.random.choice(['running', 'warmup', 'cooldown'], p=[0.6, 0.2, 0.2])
        else:           # If it is off
            params['device']['init_params']['genset_group']['device']['init_params']['gensets'][i]['controller']['init_params']['status']['value'] = 'off'

    return params

def generate_init_parameters(params):
    params = sample_init_parameters(params)  # Sample the initial parameters
    params = check_genset_running(params)  # Check the genset running status

    return params