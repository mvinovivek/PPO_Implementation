import torch
import gymnasium as gym
from networks import FeedForwardNetwork
from torch.distributions import Categorical


actor_model = "ppo_actor.pth"

env = gym.make("BipedalWalker-v3", render_mode="human")
obs_dim = env.observation_space.shape[0]

if type(env.action_space) is gym.spaces.Box:
    act_dim = env.action_space.shape[0]
else:
    act_dim = env.action_space.n

# Build our policy the same way we build our actor model in PPO
policy = FeedForwardNetwork(obs_dim, act_dim)

# Load in the actor model saved by the PPO algorithm
policy.load_state_dict(torch.load(actor_model))

obs, _ = env.reset()
done = False

t = 0
ep_len = 0
ep_ret = 0

if type(env.action_space) is gym.spaces.Discrete:
    # This works for discrete actions only
    while not done:
        t += 1
        mean = policy(obs)
        dist = Categorical(logits=mean)
        action = dist.sample().detach().numpy()
        obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        ep_ret += rew
        ep_len += 1
        env.render()
        print(ep_len)

else:
    # # This works for continuos actions only
    while not done:
        t += 1
        action = policy(obs)
        obs, rew, terminated, truncated, _ = env.step(action.detach().numpy())
        done = terminated | truncated
        ep_ret += rew
        ep_len += 1
        env.render()
        print(ep_len)
