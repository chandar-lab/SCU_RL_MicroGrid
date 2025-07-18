import sys
import os

import argparse
import wandb
import gymnasium as gym
import yaml
import numpy as np
from utils import load_config, update_config, set_seed
import matplotlib.pyplot as plt

sys.path.append("/nfs/yw8782/Downloads/rl_commander/agents")
sys.path.append("/nfs/yw8782/Downloads/rl_commander/env")
from agents.baselines import RandomAgent, NeutralAgent, TurnoffDischargeAgent
from agents.RLAgents import RLAgent
from env.env_microgrid import MicroGridEnv
import glob
# from utils import count_cycles
import json

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"),
        help="Path to config file"
    )
    parser.add_argument("--save_dir", type=str, default="logs/",
                        help="Path to save dir")
    parser.add_argument("--load_model", type=str, default="random",)
    parser.add_argument("--disable_wandb", action='store_true',
                        help="Disable Weights and Biases (wandb)")

    # Add more arguments as needed
    args, unknown_args = parser.parse_known_args()

    # Load configuration
    config = load_config(args.config)

    run = wandb.init(
        project="AI4CC_workshop_evaluate",
        config=config,
        sync_tensorboard=True,
        mode="disabled" if args.disable_wandb else "online")

    if unknown_args:
        update_config(config, args, unknown_args)

    set_seed(config['seed'])

    load_model = args.load_model #config.get('load_model', 'random').lower()
    config['environment']['return_dict'] = load_model in ['neutral', 'turn_off', 'random']


    # Create the Gym environment
    env = gym.make(config['eval_environment']['env_name'],
                   env_params=config['eval_environment'])
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Load evaluation results if exists
    rootDir = '/gpfs/groups/gc055/commanderRL/wandb'

    evals = {load_model: {}}

    # Load model or create random agent
    if load_model == 'random':
        model = RandomAgent(env.action_space)
    elif load_model == 'neutral':
        model = NeutralAgent(env.action_space)
    elif load_model == 'turn_off':
        model = TurnoffDischargeAgent(env.action_space)
    else:
        agent = RLAgent(config['agent']['algorithm'], env,
                        tensorboard_log=args.save_dir, verbose=1, device="cpu")
        model = agent.load(load_model)
        run_id = load_model.split('/')[-2]
        for dirName, subdirList, fileList in os.walk(rootDir):
            if dirName.split('-')[-1] == run_id:
                print(f"Found the run directory: {dirName}")
                training_config_path = dirName + '/files/wandb-metadata.json'
                break

        with open(training_config_path, "r") as f:
            training_config = json.load(f)['args']
        training_config = {item.split('=')[0][2:]:item.split('=')[1] for item in training_config}
        wandb.config.update(training_config, allow_val_change=True)

        evals[load_model] = {'config': training_config}
    print(f"intialized {load_model} agent ... ")


    # Run a sample trajectory
    obs, env_info = env.reset()
    terminated, truncated = False, False
    total_fuel_consumption, total_battery_degradation, rewards = [], [], []
    soc_list = []
    
    while not terminated and not truncated:
        # Take action using the model
        action, agent_info = model.predict(obs)
        # Sync observations and actions to WandB
        wandb.log({
            'external_world': env_info['external_world'],
            'microgrid': env_info['microgrid'],
            'demand': env_info['demand'],
            'battery': env_info['battery'],
            'genset_group': env_info['genset_group'],
            'wind_turbine': env_info['wind_turbine'],
            })
 
        # Step the environment
        obs, reward, terminated, truncated, env_info = env.step(action)
        total_fuel_consumption.append(env_info['genset_group']['genset_group_fuel_consumption'])
        total_battery_degradation.append(env_info['battery']['degradation_cost'])
        soc_list.append(env_info['battery']['soc'])
        rewards.append(reward)
    
    # print(count_cycles(np.array(soc_list)))
    evals[load_model].update({'total_fuel_consumption': total_fuel_consumption,
                   'total_battery_degradation': total_battery_degradation,
                     'soc_list': soc_list,
                     'reward': rewards})
    
    file_name = 'evaluation_results_jan.json'
    if os.path.exists(f'{rootDir}/{file_name}'):
        with open(f'{rootDir}/{file_name}', 'r') as f:
            all_evals = json.load(f)
    else:
        all_evals = {}


    all_evals.update(evals)
    # Save evaluation results
    with open(f'{rootDir}/{file_name}', 'w') as f:
        json.dump(all_evals, f)
    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()