import os
import sys

import numpy as np
import math
from math import pi
import random
import time
import pybullet as p
import pybullet_data

# import tf.transformations
# from tf.transformations import euler_from_quaternion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from launch.ppo_stage import TurtleBot3Simulation

diagonal_dis = math.sqrt(2) * (3.8 + 3.8)
len_batch = 6


class Env():
    def __init__(self, is_training):        
        
        self.urdf_path = "../../turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf"
        self.tb3s = TurtleBot3Simulation(self.urdf_path)
        self.position, self.orientation = p.getBasePositionAndOrientation(self.tb3s.env.robot.body_id)
        self.start_pos = self.position  # memorizzo posizione iniziale
        self.start_orn = self.orientation  # memorizzo orientamento iniziale
        self.goal_position = [0., 0., 0.01]
        self.past_distance = 0.
        self.sum1 = 0
        self.sum2 = 0
        p.setTimeStep(1.0 / 240.0)
        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    # def close(self):
    #     """
    #     Close environment. No other method calls possible afterwards.
    #     """
    #     print("=== Closing PyBullet simulation ===")
    #     p.disconnect()
    #     time.sleep(10)

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position[0]- self.position[0], self.goal_position[1] - self.position[1])
        self.past_distance = goal_distance

        return goal_distance
    
    def get_odometry(self):
        self.position, self.orientation = p.getBasePositionAndOrientation(self.tb3s.env.robot.body_id)

        x, y, _ = self.position
        euler = p.getEulerFromQuaternion(self.orientation)
        yaw = math.degrees(euler[2])
        
        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position[0] - x, 1)
        rel_dis_y = round(self.goal_position[1] - y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)
        diff_angle = (yaw - rel_theta)
        if 0 <= diff_angle <= 180 or -180 <= diff_angle < 0:
            diff_angle = round(diff_angle, 2)
        elif diff_angle < -180:
            diff_angle = round(360 + diff_angle, 2)
        else:
            diff_angle = round(-360 + diff_angle, 2)

        # print(diff_angle)
        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getState(self, scan):
        self.get_odometry()

        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        done = False
        arrive = False

        for i in range(len(scan)):
            if scan[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = math.hypot(self.goal_position[0] - self.position[0], self.goal_position[1] - self.position[1])
        if current_distance <= self.threshold_arrive:
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_position[0] - self.position[0], self.goal_position[1] - self.position[1])
        distance_rate = (self.past_distance - current_distance)

        reward = 500.*distance_rate
        self.past_distance = current_distance

        if done:
            reward = -150.
            self.tb3s.env.robot.set_linear_angular_velocity(0, 0)

        if arrive:
            reward = 100.
            self.tb3s.env.robot.set_linear_angular_velocity(0, 0)

            for body_id in self.tb3s.env.target_ids:
                p.removeBody(body_id)
            self.tb3s.env.target_ids = []

            # Build the target
            try:
                self.goal_position[0] = random.uniform(-3.6, 3.6)
                self.goal_position[1] = random.uniform(-3.6, 3.6)
                while 1.6 <= self.goal_position[0] <= 2.4 and -1.4 <= self.goal_position[1] <= 1.4 \
                        or -2.4 <= self.goal_position[0] <= -1.6 and -1.4 <= self.goal_position[1] <= 1.4 \
                        or -1.4 <= self.goal_position[0] <= 1.4 and 1.6 <= self.goal_position[1] <= 2.4 \
                        or -1.4 <= self.goal_position[0] <= 1.4 and -2.4 <= self.goal_position[1] <= -1.6:
                    self.goal_position[0] = random.uniform(-3.6, 3.6)
                    self.goal_position[1] = random.uniform(-3.6, 3.6)
                self.tb3s.env.target_ids = self.tb3s.env.create_target_zones([self.goal_position[0], self.goal_position[1], self.goal_position[2]], self.tb3s.target_config)
            except Exception as e:
                print("failed to build new target!")
            self.goal_distance = self.getGoalDistace()

            arrive = False

        return reward
    
    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        self.tb3s.env.robot.set_linear_angular_velocity(linear_vel/4, ang_vel)

        for _ in range(24):
            p.stepSimulation()
            # time.sleep(1 / 240)

        data = None
        while data is None:
            try:
                data = self.tb3s.get_lidar_observation_vector()
            except:
                pass

        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]
        # state = Pick(state, len_batch)
        for pa in past_action:
            state.append(pa)
        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive)
        return np.asarray(state), reward, done, arrive
    
    def reset(self):
        # Reset the env #
        p.resetBasePositionAndOrientation(self.tb3s.env.robot.body_id, self.start_pos, self.start_orn)
        p.resetBaseVelocity(self.tb3s.env.robot.body_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
        self.past_distance = 0
        self.position, self.orientation = p.getBasePositionAndOrientation(self.tb3s.env.robot.body_id)

        for body_id in self.tb3s.env.target_ids:
            p.removeBody(body_id)
        self.tb3s.env.target_ids = []

        try:
            self.goal_position[0] = random.uniform(-3.6, 3.6)
            self.goal_position[1] = random.uniform(-3.6, 3.6)
            while 1.7 <= self.goal_position[0] <= 2.3 and -1.2 <= self.goal_position[1] <= 1.2 \
                    or -2.3 <= self.goal_position[0] <= -1.7 and -1.2 <= self.goal_position[1] <= 1.2 \
                    or -1.2 <= self.goal_position[0] <= 1.2 and 1.7 <= self.goal_position[1] <= 2.3 \
                    or -1.2 <= self.goal_position[0] <= 1.2 and -2.3 <= self.goal_position[1] <= -1.7:
                self.goal_position[0] = random.uniform(-3.6, 3.6)
                self.goal_position[1] = random.uniform(-3.6, 3.6)
            self.tb3s.env.target_ids = self.tb3s.env.create_target_zones([self.goal_position[0], self.goal_position[1], self.goal_position[2]], self.tb3s.target_config)
        except Exception as e:
            print("failed to build the target")
        data = None
        while data is None:
            try:
                data = self.tb3s.get_lidar_observation_vector()
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]
        # state = Pick(state, len_batch)
        state.append(0)
        state.append(0)
        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
