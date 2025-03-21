from networks import FeedForwardNetwork
import time
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import gymnasium as gym


class PPO:

    def __init__(self, env: gym.Env, hyperparameters: dict):
        """
        Initializing the PPO Model

        Parameters :
        1. env - Any Gym Environment
        2. Hyperparameters

        Returns : None
        """

        assert type(env.observation_space) is gym.spaces.Box
        assert type(env.action_space) is gym.spaces.Box

        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNetwork(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNetwork(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim, ), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,
            "i_so_far": 0,
            "batch_lens": [],
            "batch_rews": [],
            "actor_losses": [],
        }

    def learn(self, total_timesteps):
        """
        Train the actor and the critic networks. The Main algorithm of PPO is implemented in this function

        Parameters:
        1. total_timesteps - The total number of timesteps the algorithms should be trained for

        Returns: None
        """
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout())

            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            V, _ = self.evaluate(batch_obs, batch_acts)
            advantage_at_k = batch_rtgs - V.detach()

            advantage_at_k = (advantage_at_k - advantage_at_k.mean()) / (
                advantage_at_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, cur_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(cur_log_probs - batch_log_probs)
                surr1 = ratios * advantage_at_k
                surr2 = (torch.clamp(ratios, 1 - self.clip, 1 + self.clip) *
                         advantage_at_k)

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger["actor_losses"].append(actor_loss.detach())

            self._log_summary()
            # Saving the model
            if int(t_so_far % self.save_freq) == 0:
                self.save_model(timestep=t_so_far)

    def save_model(self, timestep: int = None):
        if timestep is None:
            torch.save(self.actor.state_dict(), "./ppo_actor.pth")
            torch.save(self.critic.state_dict(), "./ppo_critic.pth")
            print("Saved Model!")

        else:
            torch.save(self.actor.state_dict(),
                       f"./ppo_actor_at_{timestep}.pth")
            torch.save(self.critic.state_dict(),
                       f"./ppo_critic_at_{timestep}.pth")
            print(f"Saved Model at Timestep :{timestep}")

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def _init_hyperparameters(self, hyperparameters: dict):
        # Algorithm Hyperparameters
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        # Miscellaneous parameters
        self.render = True
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None

        for param, val in hyperparameters.items():
            exec("self." + param + "=" + str(val))

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in ep_rews:
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def _log_summary(self):
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))
        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]])
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger["actor_losses"]])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print("------------------------------------------------------",
              flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
