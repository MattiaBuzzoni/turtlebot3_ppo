"""Running pre-trained agent."""
import logging
import os
import time
import torch


from src.agents.ppo import ppo
from src.agents.ppo.scripts import utility
from src.utils.cli import flags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyPlayer:
    def __init__(self, env_id: str, robot: str, debug: bool, args: dict, log_dir, agent):
        self._args = args
        self._debug = debug
        self._log_dir = log_dir
        self._env_id = env_id
        self._robot = robot

        self._args['robot_model'] = self._robot
        self._args['debug'] = self._debug

        if self._debug:
            self._args['render'] = True

        self._args['policy'] = True
        self._agent = agent(self._env_id, self._args, self._log_dir, self._debug)
        self._actor = self._agent._actor

    def play(self):
        policy_id = f"{self._env_id}"
        policy_path = flags.ENV_ID_TO_POLICY[policy_id][0]
        self._actor.load_state_dict(torch.load(policy_path))

        with torch.no_grad():
            sum_rewards = 0
            observation, _ = torch.tensor(self._agent._env.reset(), dtype=torch.float32).to(device)

            while True:
                action = self._actor(observation)
                observation, reward, terminated, truncated, _ = self._agent._env.step(action)
                done = terminated or truncated
                time.sleep(0.002)
                sum_rewards += reward
                logging.info(f"Reward={sum_rewards}")

                if done:
                    break