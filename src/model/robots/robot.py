"""Generic class for wheeled mobile robots."""

from __future__ import annotations

import numpy as np

from src.model.equipment import camera
from src.model.equipment import lidar
from src.model.equipment.lidar import Lidar
from src.utils import pybullet_data

class Robot:

    def __init__(
        self, 
        pybullet_client,
        mark,
        simulation,
        z_offset=0.  
    ):
        
        self._pybullet_client = pybullet_client
        self._simulation = simulation
        self._z_offset = z_offset
        self._mark = mark
        self._marks = self.get_marks()
        self._constants = self.get_constants()
        self._motor_name = self._marks.MARK_PARAMS[self._mark]['motor_names']
        # ...

        # Load robot URDF
        self._wheeled = self._load_urdf()
        self._build_joint_name_to_dict()
        self._build_motor_id_list()

        # ...

        self.reset_pose()

        # ...

        # Robot equipment
        self._load_equipment()

    @property 
    def pybullet_client(self):
        return self._pybullet_client
    
    @property 
    def get_robot_id(self):
        return self._wheeled
    
    @property
    def equipment(self):
        return self._equip
    
    def _get_motor_names(self):
        return self._motor_name
    
    def _build_motor_id_list(self):
        self._motor_id_list = [
            self._joint_name_to_id[motor_name]
            for motor_name in self._get_motor_names()
        ]
    
    def set_up_discrete_action_space(self):
        pass

    def set_up_continuous_action_space(self):
        pass

    def get_base_position(self):
        """Get the position of the robot's base."""
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self._wheeled))

        return position
    
    def get_base_roll_pitch_yaw(self):
        """Get the orientation of the robot's base."""
        _, orient = (self._pybullet_client.getBasePositionAndOrientation(self._wheeled))
        orient = self._pybullet_client.getEulerFromQuaternion(orient)

        return orient
    
    def get_base_roll_pitch_yaw_rate(self):
        """Get the rate of orientation change of the minitaur's base in euler angle."""
        angular_velocity = self._pybullet_client.getBaseVelocity(self._wheeled)[1]
        orientation = self.get_base_orientation()

        return self.transform_angular_velocity_to_local_frame(angular_velocity, orientation)
    
    def get_base_velocity(self):
        """Get the linear velocity of robot's base."""
        velocity, _ = self._pybullet_client.getBaseVelocity(self._wheeled)
        
        return velocity
    
    def get_base_orientation(self):
        pos, orn = self._pybullet_client.getBasePositionAndOrientation(
            self._wheeled
        )

        return orn
    
    def transform_angular_velocity_to_local_frame(self, angular_velocity, orientation):
        """Transform the angular velocity from world frame to robot's frame."""
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0],
                                                                        orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)
    
    def apply_action(self, motor_commands):
        """Apply motion command to TurtleBot3."""
        linear_velocity  =  motor_commands[0] * self._constants.MAX_LIN_VEL
        angular_velocity =  motor_commands[1] * self._constants.MAX_ANG_VEL

        L = self._constants.WHEEL_SEPARATION
        R = self._constants.WHEEL_RADIUS

        v_left = (linear_velocity - angular_velocity * L / 2.0) / R
        v_right = (linear_velocity + angular_velocity * L / 2.0) / R

        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self._wheeled,
            jointIndices=self._motor_id_list,
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocities=[v_left, v_right],
            forces=[self._constants.MAX_FORCE, self._constants.MAX_FORCE]
        )
        
    def receive_observation(self):
        self._joint_state = self._pybullet_client.getJointStates(self._wheeled, self._motor_id_list)
    
    
    def _load_urdf(self):
        x, y, z = self._constants.START_POS
        start_position = [x, y, z + self._z_offset]
        start_orientation = self._pybullet_client.getQuaternionFromEuler(
        self._constants.INIT_ORIENTATION 
    )

        return self._pybullet_client.loadURDF(
            f"{pybullet_data.getDataPath()}/{self._marks.MARK_PARAMS[self._mark]['urdf_name']}", 
            start_position, start_orientation
        )

    def _build_joint_name_to_dict(self):
        num_joints = self._pybullet_client.getNumJoints(self._wheeled)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self._wheeled, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8").replace("${namespace}", "")] = joint_info[0]
    
    def reset_pose(self):
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self._wheeled,
                jointIndex=joint_id,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
        for name in self._motor_name:
            self._pybullet_client.resetJointState(
                self._wheeled, self._joint_name_to_id[name], 0, targetVelocity=0)
            
    def get_contacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self._wheeled)

        for contact in all_contacts:
            body_b = contact[2]  
            if body_b != self._wheeled and body_b != 0:
                return True

        return False
            
    def _load_equipment(self):
        self._equip = {}
        if 'hardware' in self._marks.MARK_PARAMS[self._mark]:
            if 'camera' in self._marks.MARK_PARAMS[self._mark]['hardware']:
                self._equip = camera.parse_cams(self._marks, self._mark, self._equip)
            if 'lidar' in self._marks.MARK_PARAMS[self._mark]['hardware']:
                self._equip = lidar.parse_lds(self._marks, self._mark, self._equip)

    def update_camera(self):
        if "cams" in self._equip:
            self._equip["cams"][self._equip["default_cam"]].get_camera_image(self._pybullet_client)

    def update_lidar(self):
        if "lds" in self._equip:
            lds_pos, lds_orn = self._equip["lds"][self._equip["default_lds"]].get_lidar_pose(self._wheeled, self._pybullet_client)
            hit_results = self._equip["lds"][self._equip["default_lds"]].scan(lds_pos, lds_orn, self._pybullet_client)
            
            return self._equip["lds"][self._equip["default_lds"]].get_distances(hit_results)

        return lds_pos, lds_orn

    def get_default_camera(self):
        return self._equip["cams"][self._equip["default_cam"]]  

    def get_default_lidar(self):
        return self._equip["lds"][self._equip["default_lds"]]     
    
    def terminate(self):
        pass
