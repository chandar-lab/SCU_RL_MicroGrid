import os
import ast
import numpy as np
import torch
import random
import yaml

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    # Parse the additional parameters
    additional_args = {}
    for arg in args:
        if arg.startswith('--'):
            # key = arg.lstrip('--').split('=')[0]
            # value = arg.split('=')[-1]
            if '=' in arg:
                key, value = arg.lstrip('--').split('=', 1)
            elif '' in arg:
                key, value = arg.lstrip('--').split(' ', 1)
            else:
                raise ValueError('arguments should either be key=value or key value')
            # value = next(iter(unknown_args[unknown_args.index(arg)+1:]), None) # TODO: Adpatation of ' ' instead of '=' in this CLI
            if value is not None and not value.startswith('--'):
                additional_args[key] = value

    for key, value in additional_args.items():
        keys = key.split('.')
        try:
            value = ast.literal_eval(value)
        except:
            value = value
        update_nested_dict(config, keys, value)

def update_nested_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

    

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(seed)

