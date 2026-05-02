"""
PyBullet TurtleBot3 Simulation Environment
==================================================

This module provides a clean, modular interface for TurtleBot3 simulation using PyBullet.
Designed to be easily integrated with OpenAI Gym environments.

Author: [Mattia Buzzoni]
Date: [5 Giugno 2025]
Version: 1.0
"""

import os

import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class LidarConfig:
    """Configuration parameters for LIDAR sensor."""
    angle_resolution_deg: float = 5.9
    ray_direction_range_deg: Tuple[float, float] = (-90, 90)
    ray_length: float = 3.5
    offset: float = 0.1
    draw_debug_lines: bool = True


@dataclass
class RobotConfig:
    """Configuration parameters for robot."""
    urdf_path: str
    base_position: List[float] = None
    base_orientation: List[float] = None
    left_wheel_joint: int = 1
    right_wheel_joint: int = 2
    lidar_link_idx: int = 0
    
    def __post_init__(self):
        if self.base_position is None:
            self.base_position = [0, 0, 0.01]
        if self.base_orientation is None:
            self.base_orientation = [0, 0, 0, 1]


@dataclass
class EnvironmentConfig:
    """Configuration parameters for simulation environment."""
    gui_enabled: bool = True
    gravity: List[float] = None
    plane_size: List[float] = None
    grid_enabled: bool = True
    grid_size: float = 15.0
    grid_step: float = 1.25
    background_color: List[float] = None
    
    def __post_init__(self):
        if self.gravity is None:
            self.gravity = [0, 0, -9.81]
        if self.plane_size is None:
            self.plane_size = [30, 30, 0.01]
        if self.background_color is None:
            self.background_color = [0.7, 0.7, 0.7]


class LidarSensor:
    """
    Professional LIDAR sensor implementation with debug visualization.
    
    Provides ray-casting functionality for distance measurements and obstacle detection.
    Supports configurable angle resolution, range, and debug visualization.
    """
    
    def __init__(self, config: LidarConfig):
        """
        Initialize LIDAR sensor with given configuration.
        
        Args:
            config: LidarConfig object containing sensor parameters
        """
        self.config = config
        self.debug_line_ids: List[int] = []
        self._angle_resolution_rad = math.radians(config.angle_resolution_deg)
        self._ray_range_rad = [math.radians(config.ray_direction_range_deg[0]),
                               math.radians(config.ray_direction_range_deg[1])]
        
    @staticmethod
    def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Compute 3D rotation matrix from Euler angles.
        
        Args:
            roll: Rotation around x-axis (radians)
            pitch: Rotation around y-axis (radians)
            yaw: Rotation around z-axis (radians)
            
        Returns:
            3x3 rotation matrix
        """
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])
        return R_z @ R_y @ R_x
    
    def scan(self, position: np.ndarray, orientation: Tuple[float, float, float]) -> List[Tuple]:
        """
        Perform LIDAR scan from given position and orientation.
        
        Args:
            position: 3D position of LIDAR sensor
            orientation: Euler angles (roll, pitch, yaw) in radians
            
        Returns:
            List of hit results from PyBullet rayTestBatch
        """
        R = self._rotation_matrix(*orientation)
        num_rays = int((self._ray_range_rad[1] - self._ray_range_rad[0]) / self._angle_resolution_rad)
        
        ray_froms = []
        ray_tos = []
        
        for i in range(num_rays):
            angle = self._ray_range_rad[0] + i * self._angle_resolution_rad
            direction_local = np.array([math.cos(angle), math.sin(angle), 0])
            direction_world = R @ direction_local
            
            ray_from = position + self.config.offset * direction_world
            ray_to = ray_from + self.config.ray_length * direction_world
            
            ray_froms.append(ray_from)
            ray_tos.append(ray_to)
        
        hit_results = p.rayTestBatch(ray_froms, ray_tos)
        
        if self.config.draw_debug_lines:
            self._update_debug_lines(ray_froms, ray_tos, hit_results)
        
        return hit_results
    
    def _update_debug_lines(self, ray_froms: List[np.ndarray], 
                           ray_tos: List[np.ndarray], 
                           hit_results: List[Tuple]) -> None:
        """Update debug visualization lines for LIDAR rays."""
        num_rays = len(ray_froms)
        
        if not self.debug_line_ids:
            # Create new debug lines
            for i in range(num_rays):
                hit_obj_uid = hit_results[i][0]
                start = ray_froms[i]
                
                if hit_obj_uid != -1:
                    hit_pos = hit_results[i][3]
                    color = [0.35, 0.35, 0.88]  # Blue for hits
                    line_id = p.addUserDebugLine(start, hit_pos, color)
                else:
                    end = ray_tos[i]
                    color = [0.57, 0.57, 0.75]  # Light gray for no hits
                    line_id = p.addUserDebugLine(start, end, color)
                
                self.debug_line_ids.append(line_id)
        else:
            # Update existing debug lines
            for i in range(min(num_rays, len(self.debug_line_ids))):
                hit_obj_uid = hit_results[i][0]
                start = ray_froms[i]
                
                if hit_obj_uid != -1:
                    hit_pos = hit_results[i][3]
                    color = [0.35, 0.35, 0.88]
                    p.addUserDebugLine(start, hit_pos, color, 
                                     replaceItemUniqueId=self.debug_line_ids[i])
                else:
                    end = ray_tos[i]
                    color = [0.57, 0.57, 0.75]
                    p.addUserDebugLine(start, end, color, 
                                     replaceItemUniqueId=self.debug_line_ids[i])
    
    def get_distances(self, hit_results: List[Tuple]) -> np.ndarray:
        """
        Extract distances from LIDAR hit results.
        
        Args:
            hit_results: Results from PyBullet rayTestBatch
            
        Returns:
            Array of distances to obstacles (max range for no hits)
        """
        distances = []
        for hit_result in hit_results:
            if hit_result[0] != -1:  # Hit detected
                distances.append(hit_result[2] * self.config.ray_length)
            else:
                distances.append(float('Inf'))
        return np.array(distances)


class WorldBuilder:
    """
    Professional world building utilities for PyBullet simulation.
    
    Provides methods to create common simulation elements like walls, planes,
    grids, and visual markers with consistent styling and configuration.
    """
    
    @staticmethod
    def create_plane(size: List[float], position: List[float] = None, 
                    color: List[float] = None) -> int:
        """
        Create a textured ground plane.
        
        Args:
            size: [length, width, height] dimensions
            position: [x, y, z] position (default: [0, 0, 0])
            color: [r, g, b, a] color (default: gray)
            
        Returns:
            PyBullet body ID
        """
        if position is None:
            position = [0, 0, 0]
        if color is None:
            color = [0.6, 0.6, 0.6, 1]
        
        half_extents = [s/2 for s in size]
        
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color
        )
        
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
    
    @staticmethod
    def create_wall(center: List[float], half_extents: List[float], 
                   color: List[float] = None) -> int:
        """
        Create a wall obstacle.
        
        Args:
            center: [x, y, z] center position
            half_extents: [x, y, z] half dimensions
            color: [r, g, b, a] color (default: light gray)
            
        Returns:
            PyBullet body ID
        """
        if color is None:
            color = [0.9, 0.9, 0.9, 1]
        
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                         rgbaColor=color)
        
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center
        )
    
    @staticmethod
    def create_grid(size: float, step: float, height: float = 0.01, 
                   color: List[float] = None) -> None:
        """
        Create a visual grid on the ground plane.
        
        Args:
            size: Grid extends from -size to +size in both X and Y
            step: Distance between grid lines
            height: Z-height of grid lines
            color: [r, g, b] color (default: dark gray)
        """
        if color is None:
            color = [0.45, 0.45, 0.45]
        
        # Lines parallel to X-axis
        for i in np.arange(-size, size + step, step):
            p.addUserDebugLine(
                lineFromXYZ=[-size, i, height],
                lineToXYZ=[size, i, height],
                lineColorRGB=color,
                lineWidth=0.5
            )
        
        # Lines parallel to Y-axis
        for i in np.arange(-size, size + step, step):
            p.addUserDebugLine(
                lineFromXYZ=[i, -size, height],
                lineToXYZ=[i, size, height],
                lineColorRGB=color,
                lineWidth=0.5
            )
    
    @staticmethod
    def create_target_zones(center: List[float], zones: List[Tuple[float, List[float]]]) -> List[int]:
        """
        Create concentric target zones for navigation goals.
        
        Args:
            center: [x, y, z] center position
            zones: List of (radius, [r, g, b, a]) tuples
            
        Returns:
            List of PyBullet body IDs
        """
        body_ids = []
        base_z = center[2]
        offset_step = 0.00001
        
        for i, (radius, color) in enumerate(zones):
            # Note: This assumes you have a 'disk.obj' file in your working directory
            # You may need to create this or use a different approach
            try:
                mesh_visual = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=os.path.join(current_dir, '..', 'launch', 'disk_obj.obj'),
                    meshScale=[radius, radius, 0.001],
                    rgbaColor=color
                )
                
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=mesh_visual,
                    basePosition=[center[0], center[1], base_z + i * offset_step]
                )
                body_ids.append(body_id)
                
            except Exception as e:
                print(f"Warning: Could not create target zone {i}: {e}")
                # Fallback to cylinder
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=radius,
                    length=0.001,
                    rgbaColor=color
                )
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[center[0], center[1], base_z + i * offset_step]
                )
                body_ids.append(body_id)
        
        return body_ids


class Robot:
    """
    Professional robot controller with differential drive capabilities.
    
    Handles robot loading, joint control, and state management.
    Provides a clean interface for robot motion control and sensor integration.
    """
    
    def __init__(self, config: RobotConfig):
        """
        Initialize robot with given configuration.
        
        Args:
            config: RobotConfig object containing robot parameters
        """
        self.config = config
        self.body_id: Optional[int] = None
        self.lidar: Optional[LidarSensor] = None
        self._max_force = 1.0
        
    def load(self) -> bool:
        """
        Load robot URDF into simulation.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.config.urdf_path):
                raise FileNotFoundError(f"URDF file not found: {self.config.urdf_path}")
            
            self.body_id = p.loadURDF(
                self.config.urdf_path,
                basePosition=self.config.base_position,
                baseOrientation=self.config.base_orientation
            )
            
            # self._print_joint_info()
            return True
            
        except Exception as e:
            print(f"Error loading robot: {e}")
            return False
    
    '''def _print_joint_info(self) -> None:
        """Print information about robot joints for debugging."""
        if self.body_id is None:
            return
        
        num_joints = p.getNumJoints(self.body_id)
        print(f"Robot has {num_joints} joints:")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.body_id, i)
            print(f"  Joint {i}: {joint_info[1].decode('utf-8')} ({joint_info[2]})")'''
    
    def set_wheel_velocities(self, left_vel: float, right_vel: float) -> None:
        """
        Set wheel velocities for differential drive.
        
        Args:
            left_vel: Left wheel velocity (rad/s)
            right_vel: Right wheel velocity (rad/s)
        """
        if self.body_id is None:
            print("Warning: Robot not loaded")
            return
        
        p.setJointMotorControl2(
            bodyIndex=self.body_id,
            jointIndex=self.config.left_wheel_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=self._max_force
        )
        
        p.setJointMotorControl2(
            bodyIndex=self.body_id,
            jointIndex=self.config.right_wheel_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=self._max_force
        )
    
    def set_linear_angular_velocity(self, linear: float, angular: float, 
                                  wheelbase: float = 0.16) -> None:
        """
        Set robot motion using linear and angular velocities.
        
        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
            wheelbase: Distance between wheels (m)
        """
        # Convert to wheel velocities
        left_vel = (linear - angular * wheelbase / 2.0) / 0.033  # wheel radius = 0.033m
        right_vel = (linear + angular * wheelbase / 2.0) / 0.033
        
        self.set_wheel_velocities(left_vel, right_vel)
    
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get robot's current pose.
        
        Returns:
            Tuple of (position, orientation) as numpy arrays
        """
        if self.body_id is None:
            return np.zeros(3), np.zeros(4)
        
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos), np.array(orn)
    
    def get_lidar_pose(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Get LIDAR sensor pose.
        
        Returns:
            Tuple of (position, euler_orientation)
        """
        if self.body_id is None:
            return np.zeros(3), (0, 0, 0)
        
        lidar_state = p.getLinkState(self.body_id, self.config.lidar_link_idx)
        lidar_pos = np.array(lidar_state[0])
        lidar_orn = p.getEulerFromQuaternion(lidar_state[1])
        
        return lidar_pos, lidar_orn
    
    def attach_lidar(self, lidar_config: LidarConfig) -> None:
        """
        Attach LIDAR sensor to robot.
        
        Args:
            lidar_config: LidarConfig object
        """
        self.lidar = LidarSensor(lidar_config)
    
    def scan_lidar(self) -> Optional[np.ndarray]:
        """
        Perform LIDAR scan and return distances.
        
        Returns:
            Array of distances or None if LIDAR not attached
        """
        if self.lidar is None:
            return None
        
        lidar_pos, lidar_orn = self.get_lidar_pose()
        hit_results = self.lidar.scan(lidar_pos, lidar_orn)
        return self.lidar.get_distances(hit_results)


class SimulationEnvironment:
    """
    Professional PyBullet simulation environment manager.
    
    Provides a clean interface for setting up and managing PyBullet simulations.
    Designed to be easily integrated with OpenAI Gym environments.
    """
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize simulation environment.
        
        Args:
            config: EnvironmentConfig object containing simulation parameters
        """
        self.config = config
        self.physics_client = None
        self.robot: Optional[Robot] = None
        self.world_objects: List[int] = []
        self.target_ids: List[int] = []
        
    def start_simulation(self) -> bool:
        """
        Start PyBullet simulation with configured parameters.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to PyBullet
            if self.config.gui_enabled:
                self.physics_client = p.connect(p.GUI)
                # Configure GUI settings
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
                p.resetDebugVisualizerCamera(
                    cameraDistance=6, 
                    cameraYaw=0, 
                    cameraPitch=-89.99, 
                    cameraTargetPosition=[0, 0, 0]
                )
                p.configureDebugVisualizer(rgbBackground=self.config.background_color)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            # Set gravity
            p.setGravity(*self.config.gravity)
            
            # Create ground plane
            plane_id = WorldBuilder.create_plane(self.config.plane_size)
            self.world_objects.append(plane_id)
            
            # Create grid if enabled
            if self.config.grid_enabled:
                WorldBuilder.create_grid(self.config.grid_size, self.config.grid_step)
            
            print("Simulation environment started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting simulation: {e}")
            return False
    
    def load_robot(self, robot_config: RobotConfig) -> bool:
        """
        Load robot into simulation.
        
        Args:
            robot_config: RobotConfig object
            
        Returns:
            True if successful, False otherwise
        """
        self.robot = Robot(robot_config)
        return self.robot.load()
    
    def create_walls(self, wall_configs: List[Dict[str, Any]]) -> List[int]:
        """
        Create walls in the environment.
        
        Args:
            wall_configs: List of wall configuration dictionaries
            
        Returns:
            List of wall body IDs
        """
        wall_ids = []
        for config in wall_configs:
            wall_id = WorldBuilder.create_wall(
                center=config['center'],
                half_extents=config['half_extents'],
                color=config.get('color')
            )
            wall_ids.append(wall_id)
            self.world_objects.append(wall_id)
        return wall_ids
    
    def create_target_zones(self, center: List[float], 
                          zones: List[Tuple[float, List[float]]]) -> List[int]:
        """
        Create target zones in the environment.
        
        Args:
            center: [x, y, z] center position
            zones: List of (radius, color) tuples
            
        Returns:
            List of target zone body IDs
        """
        zone_ids = WorldBuilder.create_target_zones(center, zones)
        self.target_ids = zone_ids
        self.world_objects.extend(zone_ids)
        return zone_ids


class TurtleBot3Simulation:
    """TurtleBot3 simulation environment with LIDAR and obstacle avoidance."""
    
    
    def __init__(self, urdf_path: str, gui_enabled: bool = True):
        """Initialize the TurtleBot3 simulation environment."""
        print("=== Starting TurtleBot3 Robot Simulation ===")
        
        self.urdf_path = urdf_path
        self.env = None
        self.robot = None
        
        self._setup_environment(gui_enabled)
        self._load_robot()
        self._setup_lidar()
        self._create_world()
        
        print("Environment initialization complete!")
    
    def _setup_environment(self, gui_enabled: bool):
        """Configure and start the simulation environment."""
        config = EnvironmentConfig(
            gui_enabled=gui_enabled,
            grid_enabled=True,
            background_color=[0.7, 0.7, 0.7]
        )
        
        self.env = SimulationEnvironment(config)
        if not self.env.start_simulation():
            raise RuntimeError("Failed to start simulation")
        print("✓ Simulation environment created")
    
    def _load_robot(self):
        """Load the TurtleBot3 robot model."""
        robot_config = RobotConfig(
            urdf_path=self.urdf_path,
            base_position=[0, 0, 0.01],
            left_wheel_joint=1,
            right_wheel_joint=2,
            lidar_link_idx=5
        )
        
        if not self.env.load_robot(robot_config):
            raise RuntimeError("Failed to load robot")
        print("✓ TurtleBot3 robot loaded")
    
    def _setup_lidar(self):
        """Configure and attach LIDAR sensor."""
        lidar_config = LidarConfig(
            angle_resolution_deg=1.8,
            ray_direction_range_deg=(-90, 90),
            ray_length=3.5,
            draw_debug_lines=True
        )
        
        self.env.robot.attach_lidar(lidar_config)
        print("✓ LIDAR sensor attached")

    def _create_world(self):
        """Create obstacles and target zones in the environment."""
        print("Creating world elements...")
        
        # External walls (boundary)
        wall_configs = [
            {'center': [0, 5, 0.25], 'half_extents': [5, 0.05, 0.25]},   # Front wall
            {'center': [0, -5, 0.25], 'half_extents': [5, 0.05, 0.25]},  # Back wall
            {'center': [-5, 0, 0.25], 'half_extents': [0.05, 5, 0.25]},  # Left wall
            {'center': [5, 0, 0.25], 'half_extents': [0.05, 5, 0.25]},   # Right wall
        ]
        
        # Internal walls (obstacles)
        internal_walls = [
            {'center': [0, 2.5, 0.25], 'half_extents': [1.25, 0.05, 0.25]},
            {'center': [0, -2.5, 0.25], 'half_extents': [1.25, 0.05, 0.25]},
            {'center': [-2.5, 0, 0.25], 'half_extents': [0.05, 1.25, 0.25]},
            {'center': [2.5, 0, 0.25], 'half_extents': [0.05, 1.25, 0.25]},
        ]
        
        wall_configs.extend(internal_walls)
        self.env.create_walls(wall_configs)
        
        # Create target zones with different colors and sizes
        self.target_config = [
            (1.8, [0.39, 0.43, 1.0, 0.6]),    # Outer (blue)
            (1.2, [1.0, 0.47, 0.41, 0.8]),    # Middle (green)
            (0.6, [1.0, 0.94, 0.39, 0.9])     # Inner (yellow)
        ]

    def get_lidar_observation_vector(self) -> Optional[np.ndarray]:
        """
        Process LIDAR scan into 10 aggregated observations for Actor-Critic input.

        Returns:
            A NumPy array of 10 minimal distances (1 per batch), or None if scan fails.
        """
        distances = self.env.robot.scan_lidar()
        
        if distances is None or len(distances) < 30:
            print("LIDAR scan has insufficient data (need at least 30 points).")
            return None
        
        # Use only the first 30 points 
        selected = distances[:30]
        batches = selected.reshape((10, 3))
        min_values = np.min(batches, axis=1)

        return min_values
