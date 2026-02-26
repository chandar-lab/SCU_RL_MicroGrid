# main.py
import sys
import argparse
import torch
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
agents_path = os.path.join(base_dir, 'agents')
env_path = os.path.join(base_dir, 'env')

sys.path.append(agents_path)
sys.path.append(env_path)
from agents.RLAgents import RLAgent, LagrangianUpdateCallback
from env.env_microgrid import MicroGridEnv
from utils import load_config, update_config, set_seed
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RL Training with Stable Baselines3")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"), help="Path to config file")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "logs"), help="Path to save dir")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable Weights and Biases (wandb)")
    
    # Add more arguments as needed
    args, unknown_args = parser.parse_known_args()
    
    new_logger = configure(args.save_dir, ["stdout", "csv", "tensorboard"])

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the first available GPU
        print('Using GPU:', torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')  # Use CPU
        print('Using CPU')

    # Load configuration
    config = load_config(args.config) 

    if unknown_args:
        update_config(config, unknown_args)
    
    # Making sure prediction steps are the same for wind turbine and demand predictions (TODO: is this really necessary?)
    assert config['environment']['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['pred_time_step']['value'] == config['environment']['microgrid']['device']['init_params']['demand']['device']['const_params']['pred_time_step']['value']

    # Making sure all time steps are the same for train and eval environments
    assert config['eval_environment']['microgrid']['device']['init_params']['demand']['device']['const_params']['pred_time_step']['value'] == config['environment']['microgrid']['device']['init_params']['demand']['device']['const_params']['pred_time_step']['value']
    assert config['eval_environment']['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['pred_time_step']['value'] == config['environment']['microgrid']['device']['init_params']['wind_turbine']['device']['const_params']['pred_time_step']['value']

    run = wandb.init(
        project="RLC-experiments",
        dir=args.save_dir,
        config=config,
        sync_tensorboard=True, # keep this true to sync wandb logs
        mode="disabled" if args.disable_wandb else "online")

    save_dir = f"{args.save_dir}{run.id}"
    # Create the Gym environment
    env = gym.make(config['environment']['env_name'], env_params=config['environment'])
    # env = Monitor(env)

    eval_env = gym.make(config['eval_environment']['env_name'], env_params=config['eval_environment'])
    # eval_env = Monitor(env)
    set_seed(config['seed'])

    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir,
                                log_path=save_dir, eval_freq=config['runner']['eval_freq'],
                                n_eval_episodes=config['runner']['n_eval_episodes'],
                                deterministic=True, render=False)
    wandb_callback = WandbCallback(model_save_path=save_dir, 
                                   model_save_freq=config['runner']['eval_freq'], 
                                   gradient_save_freq=0,
                                   verbose=2,)
    
    # Initialize the model
    model = RLAgent(config['agent']['algorithm'], env, tensorboard_log=save_dir, verbose=1, seed=config['seed'], device=device, batch_size=config['runner']['batch_size'],  stats_window_size=config['runner']['n_eval_episodes'], policy_kwargs=config['agent'])

    # model.set_logger(new_logger) # Comment out this line if you want to use WandbCallback
    # Train the model

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config['runner']['n_eval_episodes'])
    wandb.log({"eval/mean_reward": mean_reward, "eval_steps": 0})

    # for i in range(config['runner']['total_timesteps']//config['runner']['eval_freq']):
    callbacks = [wandb_callback, eval_callback]
    if config["agent"]["algorithm"] == "LSAC":
        lag_cfg = config["agent"].get("lagrangian", {})
        callbacks.append(
            LagrangianUpdateCallback(
                lambda_lr=lag_cfg.get("lambda_lr", 1e-3),
                cost_limit=lag_cfg.get("cost_limit", 0.05),
                lambda_lrs=lag_cfg.get("lambda_lrs", {}),
                cost_limits=lag_cfg.get("cost_limits", {}),
            )
        )
    model.learn(config['runner']['total_timesteps'], callback=callbacks)
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config['runner']['n_eval_episodes'])
    # wandb.log({"eval/mean_reward": mean_reward, "eval_steps": config['runner']['total_timesteps']})
    # print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    env.close()

    wandb.save(f"{save_dir}/*.zip")
    print('Final model saved')

if __name__ == "__main__":
    main()
