import gymnasium as gym
from .env_microgrid import MicroGridEnv
from gymnasium.envs.registration import register

register(
    id='MicroGridEnv-v0',
    entry_point='env.env_microgrid:MicroGridEnv',  # Replace with the actual path
    max_episode_steps=525600,  # Equivalent to one year
    kwargs={'env_params': {}}
)