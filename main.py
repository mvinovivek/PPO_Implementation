import gymnasium as gym
from ppo import PPO

hyperparameters = {
    "timesteps_per_batch": 1000,
    "max_timesteps_per_episode": 500,
    "gamma": 0.99,
    "n_updates_per_iteration": 10,
    "lr": 3e-4,
    "clip": 0.2,
    "render": True,
    "render_every_i": 10,
    "save_intervals": False,
    "save_freq": 2000,
}

env = gym.make("BipedalWalker-v3")
ppo = PPO(env, hyperparameters)
ppo.learn(10000)
ppo.save_model()
