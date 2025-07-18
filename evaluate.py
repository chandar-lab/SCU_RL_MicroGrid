import sys
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from main import load_config
import wandb
import os

agents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
if agents_path not in sys.path:
    sys.path.append(agents_path)
from agents.baselines import RandomAgent, SimpleAgent
from agents.RLAgents import RLAgent

def main():
    # Parse command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default='random')
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"), help="Path to config file")

    args = parser.parse_args()

    config = load_config(args.config)

    wandb.init(
    project="commander_RL_evaluation",
    config=config,
    sync_tensorboard=True)

    # Create the Gym environment
    env = gym.make("MicroGridEnv-v0", env_params=config['environment'])


    # Load model or create random agent
    if args.load_model.lower() == 'random':
        model = RandomAgent(env.action_space)
    elif args.load_model.lower() == 'simple':
        model = SimpleAgent(env.action_space)
    else:
        agent = RLAgent(config['agent']['algorithm'], env, tensorboard_log=args.save_dir, verbose=1, device="cpu")
        model = agent.load(args.load_model)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    main()