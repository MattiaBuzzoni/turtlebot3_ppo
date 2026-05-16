"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""
import datetime
import functools
import logging
import os
import platform
import time

import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.distributions import MultivariateNormal

from src.agents.agent import Agent
from src.agents.ppo.net_actor import NetActor
from src.agents.ppo.net_critic import NetCritic
from src.agents.ppo.scripts import configs
from src.agents.ppo.tools.attr_dict import AttrDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO(Agent):
    """This is the PPO class we will use as our model in main.py."""

    def __init__(self, env_id, env_args, log_dir, debug_mode):
        super().__init__(env_id, env_args, log_dir, debug_mode)    
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                    None
		"""
        # Make sure the environment is compatible with our code
        # assert(type(env.observation_space) == gym.spaces.Box)
        # assert(type(env.action_space) == gym.spaces.Box)
        self._env_id = env_id
        self._args = env_args
        self._log_dir = log_dir
        self._debug = debug_mode

        self._config = AttrDict(getattr(configs, self._env_id)())
        
        # Extractenvironment information
        self._env = self._create_environment(self._config)
        self._obs_dim = self._env.observation_space.shape[0]
        self._act_dim = self._env.action_space.shape[0]

        # Initialize actor and critic networks
        self._actor = NetActor(self._obs_dim, self._act_dim).to(device)
        self._critic = NetCritic(self._obs_dim, 1).to(device)

        # Initialize optimizers for actor and critic
        self._actor_optim = Adam(self._actor.parameters(), lr=self._config.lr)
        self._critic_optim = Adam(self._critic.parameters(), lr=self._config.lr)

        # Initialize the covariance matrix used to query the actor for actions
        #self._cov_var = torch.full(size=(self._act_dim,), fill_value=0.8).to(device)
        #self._cov_mat = torch.diag(self._cov_var).to(device)

    
    def _create_environment(self, config):
        """Constructor for an instance of the environment."""
        env = gym.make(config.env, **self._env_args)

        return env

    def _learn(self, config):
        """Train the actor and critic networks. Here is where the main PPO algorithm resides."""
        print(f"Learning... Running {config.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{config.timesteps_per_batch} timesteps per batch for a total of {config.total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        value_func = []
        while t_so_far < config.total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self._rollout(t_so_far, config)  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Calculate advantage at k-th iteration
            self.V, _ = self._evaluate(batch_obs, batch_acts, config)
            value_func.append(self.V.detach().mean())
            A_k = batch_rtgs - self.V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(config.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                self.V, curr_log_probs = self._evaluate(batch_obs, batch_acts, config)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - config.clip, 1 + config.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(self.V, batch_rtgs)
                # weihgts = self.actor.parameters()
                # w1_res1_actor0 = weihgts.gi_frame.f_locals['self'].rb1.fc1.weight

                # Calculate gradients and perform backward propagation for actor network
                self._actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self._actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self._critic_optim.zero_grad()
                critic_loss.backward()
                self._critic_optim.step()
            
            # Save our model if it's time
            # create model folder
            path = f'src/policies/{self._env_id}'
            os.makedirs(path, exist_ok=True)

            if i_so_far % config.save_freq == 0:
                epoch = i_so_far // config.save_freq
                torch.save(self._actor.state_dict(), f'{path}/ppo_actor_{epoch}.pth')
                torch.save(self._critic.state_dict(), f'{path}/ppo_critic_{epoch}.pth')

    def _rollout(self, t_so_far, config):
        """
			This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Reset the environment. Note that obs is short for observation.
        obs, info = self._env.reset()
        done = False
        episode_reward = 0
        one_round = 0
        ep_rews = []
        # while t < self.timesteps_per_batch:
        for t in range(config.timesteps_per_batch):

            # Track observations in this batch
            batch_obs.append(obs)
            # Calculate action and make a step in the env.
            # Note that rew is short for reward.
            action, log_prob = self._get_action(obs, t_so_far, one_round, config)
            # old state as input because of reward function
            obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated

            episode_reward += reward
            # Track recent reward, action, and action log probability
            ep_rews.append(reward)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            # t += 1 		# Increment timesteps ran this batch so far
            one_round += 1
            if done or one_round >= config.max_timesteps_per_episode:
                batch_lens.append(one_round)
                batch_rews.append(ep_rews)
                ep_rews = []
                if one_round != 0:
                    print('Step: %3i' % one_round, '| Avg. Reward:{:.2f}'.format(episode_reward / one_round),
                          '| Time step: %i' % (t_so_far + np.sum(batch_lens)))
                episode_reward = 0
                one_round = 0
                done = False
                obs, info = self._env.reset()
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            # If render is specified, render the environment
            # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
            # 	self.env.render()
            # If the environment tells us the episode is terminated, break

        # Track episodic lengths and rewards
        batch_rews.append(ep_rews)
        # if one_round != 0:
        # 	print('Step: %3i' % one_round, '| avg_reward:{:.2f}'.format(episode_reward / one_round),
        # 		  '| Time step: %i' % (t_so_far + np.sum(batch_lens)), '|', result)
        #
        # 	self.logger['Episode_Rewards'].append(episode_reward / one_round)

        episode_rewards = []

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
        batch_rtgs = self._compute_rtgs(batch_rews, config).to(device)

        # ALG STEP 4
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def _compute_rtgs(self, batch_rews, config):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * config.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def _get_action(self, obs, t_so_far, one_round, config):
        """
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
        self.t_step = one_round
        # Query the actor network for a mean action
        mean, log_std = self._actor(obs)
        std = log_std.exp()

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        #if self.t_step == 0 and t_so_far > 50000 and self._cov_mat[0][0] >= 0.1:
        #    self._cov_mat *= 0.995
        dist = MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(std))

        # Sample an action from the distribution
        action = dist.sample()
        action = torch.stack([
            torch.clamp(action[0], -1, 1),
            torch.clamp(action[1], -1, 1)
        ])
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().cpu().numpy(), log_prob.detach()

    def _evaluate(self, batch_obs, batch_acts, config):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        self.V = self._critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean, log_std = self._actor(batch_obs)
        std = log_std.exp()

        mean = torch.stack([
            torch.clamp(mean[:, 0], -1, 1),
            torch.clamp(mean[:, 1], -1, 1)
        ], dim=1)
        
        dist = MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(std))
        log_probs = dist.log_prob(batch_acts)
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return self.V, log_probs


    def train(self):
        """Create configuration and launch the training."""
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        full_logdir = os.path.expanduser(os.path.join(self._log_dir, '{}-{}'.format(timestamp, self._env_id)))

        self._learn(self._config)

