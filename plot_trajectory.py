import sys
import os

import argparse
import wandb
import gymnasium as gym
import yaml
import numpy as np
from utils import load_config, update_config, set_seed
import matplotlib.pyplot as plt

sys.path.append("/nfs/yw8782/Downloads/rlc_experiments/agents")
sys.path.append("/nfs/yw8782/Downloads/rlc_experiments/env")
from agents.baselines import RandomAgent, TurnoffDischargeAgent, TurnoffIdleAgent, TurnoffChargeAgent, NeutralDischargeAgent, NeutralIdleAgent, NeutralChargeAgent, TurnonDischargeAgent, TurnonIdleAgent, TurnonChargeAgent, GreedyAgent
from agents.RLAgents import RLAgent
from env.env_microgrid import MicroGridEnv
import glob
import json

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--render-config", type=str, default="configs/render.yaml",
                        help="Path to render config file")
    parser.add_argument("--save_dir", type=str, default="logs/",
                        help="Path to save dir")
    parser.add_argument("--disable_wandb", action='store_true',
                        help="Disable Weights and Biases (wandb)")
    parser.add_argument("--render", action='store_true',
                        help="Renders the environment with Pygame")

    # Add more arguments as needed
    args, unknown_args = parser.parse_known_args()

    # Load configuration
    config = load_config(args.config)

    run = wandb.init(
        project="rlc_exp_deploy",
        config=config,
        sync_tensorboard=True,
        mode="disabled" if args.disable_wandb else "online")

    if unknown_args:
        update_config(config, unknown_args)

    set_seed(config['seed'])
    # List all possible baseline_agents
    baseline_agents = {
        'random': RandomAgent,
        'turn_off_discharge': TurnoffDischargeAgent,
        'turn_off_idle': TurnoffIdleAgent,
        'turn_off_charge': TurnoffChargeAgent,
        'neutral_discharge': NeutralDischargeAgent,
        'neutral_idle': NeutralIdleAgent,
        'neutral_charge': NeutralChargeAgent,
        'turn_on_discharge': TurnonDischargeAgent,
        'turn_on_idle': TurnonIdleAgent,
        'turn_on_charge': TurnonChargeAgent,
        'greedy': GreedyAgent
    }
    # Get model
    load_model = config.get('load_model', 'random')

    config['eval_environment']['return_dict'] = load_model in baseline_agents.keys()     # If the agent is a baseline, return the dict


    # Create the Gym environment
    env = gym.make(config['eval_environment']['env_name'],
                   env_params=config['eval_environment'])
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Load evaluation results if exists


    # Load model or create random agent
    if load_model in baseline_agents.keys():
        eval_iterations = 10
        model = baseline_agents[load_model](env)
        conservativeness_coeff = config['eval_environment']['microgrid']['shield']['conservativeness_coeff']
        load_model = load_model+f'-{conservativeness_coeff}'
        evals = {load_model : {'config': f'{conservativeness_coeff}', 'results': {}}}
    else:
        rootDir = '/'.join(load_model.split('/')[:-2] + ['wandb'])
        eval_iterations = 10

        agent = RLAgent(config['agent']['algorithm'], env,
                        tensorboard_log=args.save_dir, verbose=1, device="cpu", policy_kwargs=config['agent'])
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

        if 'environment.microgrid.shield.check_balance' in training_config:
            env.microgrid_controller.shield_params['check_balance'] = training_config['environment.microgrid.shield.check_balance']
        if 'environment.microgrid.shield.check_reserve' in training_config:
            env.microgrid_controller.shield_params['check_reserve'] = training_config['environment.microgrid.shield.check_reserve']

        evals = {load_model: {'config': training_config, 'results': {}}}
    print(f"intialized {load_model} agent ... ")

    for i in range(eval_iterations):
        # Run a sample trajectory
        obs, env_info = env.reset()
        date_time = str(env_info['external_world']['date_time'])

        while date_time in evals[load_model]['results'].keys():
            obs, env_info = env.reset()
            date_time = str(env_info['external_world']['date_time'])

        terminated, truncated = False, False
        total_fuel_consumption, total_battery_degradation, rewards = [], [], []
        neg_balance, genset_overload, rew_shield, balanced, reserve_balanced = [], [], [], [], []
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
            neg_balance.append(env_info['neg_balance'])
            genset_overload.append(env_info['genset_overload'])
            rew_shield.append(env_info['rew_shield'])
            balanced.append(int(env_info['balanced']))
            reserve_balanced.append(int(env_info['reserve_balanced']))

        reserve_balanced_and_balanced = [1 if balanced[i] == 1 and reserve_balanced[i] == 1 else 0 for i in range(len(balanced))]
        balanced_only_list = [1 if balanced[i] == 1 and reserve_balanced[i] == 0 else 0 for i in range(len(balanced))]
        reserve_balanced_only_list = [1 if reserve_balanced[i] == 1 and balanced[i] == 0 else 0 for i in range(len(balanced))]
        
        evals[load_model]['results'].update({date_time: {'total_fuel_consumption': sum(total_fuel_consumption),
                    'total_battery_degradation': sum(total_battery_degradation),
                        'reward': sum(rewards),
                        'neg_balance': sum(neg_balance),
                        'genset_overload': sum(genset_overload),
                        'rew_shield': sum(rew_shield),
                        'balanced': sum(balanced),
                        'reserve_balanced': sum(reserve_balanced),
                        'reserve_balanced_and_balanced': sum(reserve_balanced_and_balanced),
                        'balanced_only_list': sum(balanced_only_list),
                        'reserve_balanced_only_list': sum(reserve_balanced_only_list)}})
    
    file_name = os.path.join(os.path.dirname(__file__), "results", "experiment_results.json")
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            all_evals = json.load(f)
    else:
        all_evals = {}


    all_evals.update(evals)
    # Save evaluation results
    with open(file_name, 'w') as f:
        json.dump(all_evals, f)
    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
