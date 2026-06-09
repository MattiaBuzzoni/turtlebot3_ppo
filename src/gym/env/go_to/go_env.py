""" This Gym Environment implements the "Go to target" task. """

import math 
import random 
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.core import sim_constants
from src.gym import robot_gym_env
from src.utils import pybullet_data

class GoEnv(robot_gym_env.RobotGymEnv):

    """The gym environment for the TurtleBot3."""

    def __init__(
        self,
        robot_model,
        mark,
        obstacles_list=None,
        target_position=None,
        terrain_id=None,
        terrain_type='plane',
        show_plot=False,
        on_rack=False,
        render=False,
        record_video=False,
        debug=False,
        policy=False,
    ):
        
        super(GoEnv, self).__init__(
            robot_model=robot_model,
            on_rack=on_rack,
            render=render,
            debug=debug,
            terrain_id=terrain_id,
            terrain_type=terrain_type,
            record_video=record_video,
            mark=mark,
            policy=policy            
        )

        self._debug = debug
        self._show_plot = show_plot
        self._target_position = target_position
        self._simulation.pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)
        self._obstacle_list = np.array(obstacles_list) if obstacles_list is not None else np.array([])
        self._random_target = True if self._target_position is None else False

        # Maximum episode time in seconds
        self._max_time = 1000

        self.prev_pos = ((0., 0.), 0.)
        self.pos = ((0., 0.), 0.)
        #self.prev_vel = ((0., 0.), 0.)
        #self.vel = ((0., 0.), 0.)
        
        self._done = False
        self._plot = None 
        self._observation = []

        # Build Actions space
        self._build_action_space()

        # Build Observations space
        self._build_observation_space()

        if self._debug:
            self._ui = self._setup_ui_parameters()
            self._build_world(True)

    def _build_action_space(self):
        self.action_space = spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                                       high=np.array([1., 1.], dtype=np.float32))
        
    def _build_observation_space(self):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )

    def _setup_ui_parameters(self):
        ui = {
            "sim_ui": self.simulation.setup_ui_parameters(),
            # # terrain ui params
            # "terrain": self.simulation.terrain.setup_ui_params(self.simulation.pybullet_client),
            # robot camera
            "cam_ui": self.setup_equipment_ui_params()            
        }

        return ui
    
    def setup_equipment_ui_params(self):
        cam_ui = self.simulation.pybullet_client.addUserDebugParameter("Show robot cam", 0, -1, 0)
        return cam_ui

    def parse_equipment_ui_params(self):
        if "cams" in self.simulation.robot.equipment:
            cam_flag = self.simulation.pybullet_client.readUserDebugParameter(self._ui["cam_ui"])
            if cam_flag % 2 != 0:
                return True
            return False
        return False
    
    def get_odometry(self):
        pos = self.simulation.robot.get_base_position()
        orn = self.simulation.robot.get_base_roll_pitch_yaw()

        x, y, _ = pos

        yaw = orn[2]

        dx = self._target_position[0] - x
        dy = self._target_position[1] - y

        rel_theta = math.atan2(dy, dx)
        diff_angle = (yaw - rel_theta + math.pi) % (2 * math.pi) - math.pi

        return yaw, rel_theta, diff_angle
    
    def update_position_velocity(self, current_pos, current_orient, robot_id, pybullet_client):
        new_xy, new_yaw = self.format_position(current_pos, current_orient)
        self.prev_pos = self.pos
        # self.prev_vel = self.vel
        self.pos = new_xy, new_yaw
        # self.vel = self.get_velocity(pybullet_client, robot_id)

    @staticmethod
    def format_position(position, orientation):
        x, y, _ = position
        _, _, yaw = orientation
        return (x, y), yaw

    @staticmethod
    def get_velocity(pb_client, robot):
        linear, angular = pb_client.getBaseVelocity(robot)
        vx, vy, _ = linear
        _, _, wz = angular
        return (vx, vy), wz

    def get_info(self):
        (x, y), yaw = self.pos
        return {"x": x,
                "y": y,
                "yaw": yaw}
    
    def _distance_to_target(self):
        pos = self.simulation.robot.get_base_position()
        distance = math.hypot(pos[0] - self._target_position[0], 
                              pos[1] - self._target_position[1])
        
        return distance
    
    def _prev_distance_to_target(self):
        (px, py), _ = self.prev_pos
        distance = math.hypot(px - self._target_position[0], 
                              py - self._target_position[1])
        
        return distance
    
    def _on_target(self):
        if self._distance_to_target() <= 0.1:
            return True
        
        return False
    
    def get_observation(self):
        self.update_position_velocity(self.simulation.robot.get_base_position(),
                                      self.simulation.robot.get_base_roll_pitch_yaw(),
                                      self.simulation.robot._wheeled,
                                      self.simulation.pybullet_client)
        yaw, rel_theta, diff_angle = self.get_odometry()

        (x, y), _ = self.pos
        # (vx, vy), _ = self.vel

        scan_range = []
        scans = self.simulation.robot.update_lidar()
        batches = scans.reshape((36, 10))
        min_values = np.min(batches, axis=1)

        scan_range = np.clip(min_values, 0, 3.5)
        scan_range = np.nan_to_num(scan_range, nan=0.0)

        target_dist = self._distance_to_target()


        observation = np.concatenate([
            scan_range,        
            [x, y],                
            [target_dist],        
            [diff_angle],
            [yaw, rel_theta],
            ]).astype(np.float32)

        self._observation = np.array(observation)

        return self._observation
    
    def step(self, action, **kwargs):
        return super(GoEnv, self).step(action, **kwargs)
    
    def reward(self):
        reward = 0.
        pos = self.simulation.robot.get_base_position()
        orn = self.simulation.robot.get_base_roll_pitch_yaw()
        yaw = orn[2]

        # Goal Distance Rate
        d_goal = self._prev_distance_to_target() - self._distance_to_target()

        # Goal Heading
        dx = self._target_position[0] - pos[0]
        dy = self._target_position[1] - pos[1]
        G_h = math.atan2(dy, dx) - yaw

        # Goal-oriented Reward Function
        A = np.mod((0.5 * (G_h + np.pi)), 2 * np.pi)
        if d_goal > 0.5 or d_goal <= 0:
            reward = -10
        elif 0 < d_goal <= 0.5:
            d_goal = d_goal * 100
            reward = 200. * d_goal * (1 - 4 * np.abs(0.5 - np.mod((A/np.pi), 1)))

        return reward 

    def reset(self, seed=None, options=None):
        if self._debug:
            # Create world object
            self._build_world()
            # Setup sim camera position
            # self.simulation.set_camera(3.8, 0, -30)
        self._build_world()

        new_xy, new_yaw = self.format_position(self.simulation.robot.get_base_position(),
                                               self.simulation.robot.get_base_roll_pitch_yaw())
        self.pos = new_xy, new_yaw
        self.prev_pos = new_xy, new_yaw

        self._done = False

        return super(GoEnv, self).reset()
    
    def _build_world(self, start_pos=None):
        if "target" not in self.world_object:
            urdf_root = pybullet_data.getDataPath()
            self.world_object["target"] = self.simulation.pybullet_client.loadURDF(
                f"{urdf_root}/world/objects/target/target.urdf",
                useFixedBase=True
            )

        if self._random_target:
            x = round(np.random.uniform(-2.5, 2.5), 2)
            y = round(np.random.uniform(-2.5, 2.5), 2)
            if 1. > x > 0:
                x = 1.
            if -1. < x < 0:
                x = -1.
            if 1. > y > 0:
                y = 1.
            if -1. < y < 0:
                y = -1.
            self._target_position = [x, y]
            if self._debug:
                print(f"Target: {x},{y}")
        
        if start_pos is None:
            x, y = self._target_position
            target_pos = [x, y, 0.]
        else:
            target_pos = [0., 0., 0.]
        self.simulation.pybullet_client.resetBasePositionAndOrientation(self.world_object["target"],
                                                                        target_pos,
                                                                        [0, 0, 0, 1])
        
    def termination(self):
        if super(GoEnv, self).termination()[0]:
            return True, {}
        
        done = False
        max_steps = self._max_time / (sim_constants.SIMULATION_TIME_STEP * sim_constants.ACTION_REPEAT)
        if self._on_target():
            done = True
            print("PATH DONE!")
        elif self.simulation.step_counter > max_steps:
            done = True
            print("TIME LIMIT")

        info = self.get_info()

        return done, info
