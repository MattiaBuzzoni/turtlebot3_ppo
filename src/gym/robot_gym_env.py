"""Implement the abstract class of Robot agnostic gym environment."""

from __future__ import annotations

import time
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod

from gymnasium.utils import seeding
from src.core.simulation import Simulation

class RobotGymEnv(gym.Env, ABC):
    
    metadata = {"render.modes": ["human", "gui", "rgb_array"], "video.frames_per_second": 100}

    def __init__(
        self, 
        robot_model, 
        mark,
        terrain_id=None,
        terrain_type='plane',
        on_rack=False,
        render=False,
        record_video=False,
        debug=False,
        policy=False,
    ):
        """Initialize the gym environment."""
        self._render = render
        self._debug = debug
        self._policy = policy
        self._last_frame_time = 0

        # Start simulation
        self.world_object = {}
        self._simulation = Simulation(robot_model=robot_model,
                                      debug=debug,
                                      terrain_id=terrain_id,
                                      terrain_type=terrain_type,
                                      record_video=record_video,
                                      render=render,
                                      mark=mark)
        
        self._on_rack = on_rack
        self.seed()
        
    @property
    def simulation(self):
        return self._simulation
    
    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def _build_action_space(self):
        pass

    @abstractmethod
    def _build_observation_space(self):
        pass

    def close(self):
        self._simulation.robot.terminate()

    def reset(self, **kwargs):
        print("reset simulation")
        self.simulation.reset()

        if self.simulation.terrain.terrain_type != "plane":
            self.simulation.terrain.update_terrain()

        if "position" in kwargs:
            position = kwargs["position"]
        else:
            position = self._simulation.robot.get_constants().START_POS

        # Get terrain offset
        z_offset = self.simulation.terrain.get_terrain_z_offset()
        position = [position[0], position[1], position[2] + z_offset]

        if "orientation" in kwargs:
            orientation = [0, 0, kwargs["orientation"]]
        else:
            orientation = self._simulation.robot.get_constants().INIT_ORIENTATION

        self._simulation.pybullet_client.resetBasePositionAndOrientation(
            self._simulation.robot.get_robot_id,
            position,
            self._simulation.pybullet_client.getQuaternionFromEuler(orientation)
        )

        return self.get_observation()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]
    
    def step(self, action, **kwargs):
        """Step forward the simulation, given the action."""

        self.simulation.apply_step_action(action)
        if "update_lds" in kwargs:
            self._simulation.robot.update_lidar()
        observation = self.get_observation()
        reward = self.reward()
        done, info = self.termination()

        return np.array(observation), reward, done, info

    def render(self, mode="rgb_array", close=False):
        return self._simulation.render(mode)

    def _sleep_at_reset(self):
        if self._render:
            # Sleep, otherwise the computation takes less time than real time
            # which will make the visualization like a fast-forward video
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.simulation.env_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.simulation.robot.get_base_position()      

            # Also keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self.simulation.pybullet_client.getDebugVisualizerCamera()[8:11]
            self.simulation.pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            self.simulation.pybullet_client.configureDebugVisualizer(
                self.simulation.pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)  
            
    def is_contact(self):
        """Decide wheter the robot is crash on a wall."""
        contacts = self.simulation.robot.get_contacts()
        return contacts

    def termination(self):
        if self.is_contact():
            print("CRASHING!")
        return self.is_contact(), {}
    