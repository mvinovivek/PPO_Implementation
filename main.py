import gymnasium as gym
from ppo import PPO

hyperparameters = {
    'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 200,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'lr': 3e-4,
    'clip': 0.2,
    'render': True,
    'render_every_i': 10
}

env = gym.make('Pendulum-v1')
ppo = PPO(env, hyperparameters)
ppo.learn(100)
