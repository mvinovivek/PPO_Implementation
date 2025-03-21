import gymnasium as gym
from ppo import PPO

env = gym.make('Pendulum-v1')
ppo = PPO(env)
ppo.learn(10000)
