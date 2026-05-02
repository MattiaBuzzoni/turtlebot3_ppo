"""Equipment class for LIDAR sensor."""

import math
import numpy as np
from typing import List, Tuple


def parse_lds(marks, mark, equip):
    for lds in marks.MARK_PARAMS[mark]["hardware"]["lidar"]["lds"]:
        if "lds" in equip:
            equip["lds"].append(Lidar(
                                    lds["name"],
                                    lds["link"],
                                    lds["angle_resolution_deg"],
                                    lds["ray_direction_range_deg"],
                                    lds["ray_length"],
                                    lds["offset"])
                                )
        else:
            equip["lds"] = [Lidar(
                                lds["name"],
                                lds["link"],
                                lds["angle_resolution_deg"],
                                lds["ray_direction_range_deg"],
                                lds["ray_length"],
                                lds["offset"])]
    equip["default_lds"] = marks.MARK_PARAMS[mark]["hardware"]["lidar"]["default"]
    return equip


class Lidar:

    def __init__(
        self,
        name: str,
        link_id,
        angle_resolution_deg,
        ray_direction_range_deg,
        ray_length,
        offset,
    ):
        
        self._name = name
        self._link_id = link_id
        self._angle_resolution_deg = angle_resolution_deg
        self._ray_direction_range_deg = ray_direction_range_deg
        self._ray_length = ray_length
        self._offset = offset
        self._draw_debug_lines = True
        self.debug_line_ids: List[int] = []
        self._angle_resolution_rad = math.radians(self._angle_resolution_deg)
        self._ray_range_rad = [math.radians(self._ray_direction_range_deg[0]),
                               math.radians(self._ray_direction_range_deg[1])]

    def get_lidar_pose(self, body_id, pb_client):
        """Get LIDAR sensor pose."""
        state = pb_client.getLinkState(body_id, self._link_id)
        pos = np.array(state[0])
        orn = pb_client.getEulerFromQuaternion(state[1])

        return pos, orn

    @staticmethod
    def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute 3D rotation matrix from Euler angles."""
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

    def scan(self, position: np.ndarray, orientation: Tuple[float, float, float], pb_client) -> List[Tuple]:
        """Perform LIDAR scan from given position and orientation."""
        R = self._rotation_matrix(*orientation)
        num_rays = int((self._ray_range_rad[1] - self._ray_range_rad[0]) 
               / self._angle_resolution_rad) + 1
        
        ray_froms = []
        ray_tos = []
        
        for i in range(num_rays):
            angle = self._ray_range_rad[0] + i * self._angle_resolution_rad
            direction_local = np.array([-math.cos(angle), -math.sin(angle), 0])
            direction_world = R @ direction_local
            
            ray_from = position 
            ray_to = ray_from + self._ray_length * direction_world
            
            ray_froms.append(ray_from)
            ray_tos.append(ray_to)
        
        hit_results = pb_client.rayTestBatch(ray_froms, ray_tos)
        
        if self._draw_debug_lines:
            self._update_debug_lines(ray_froms, ray_tos, hit_results, pb_client)
        
        return hit_results
    
    def _update_debug_lines(
            self, 
            ray_froms: List[np.ndarray], 
            ray_tos: List[np.ndarray], 
            hit_results: List[Tuple],
            pb_client) -> None:
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
                    line_id = pb_client.addUserDebugLine(start, hit_pos, color)
                else:
                    end = ray_tos[i]
                    color = [0.57, 0.57, 0.75]  # Light gray for no hits
                    line_id = pb_client.addUserDebugLine(start, end, color)
                
                self.debug_line_ids.append(line_id)
        else:
            # Update existing debug lines
            for i in range(min(num_rays, len(self.debug_line_ids))):
                hit_obj_uid = hit_results[i][0]
                start = ray_froms[i]
                
                if hit_obj_uid != -1:
                    hit_pos = hit_results[i][3]
                    color = [0.35, 0.35, 0.88]
                    pb_client.addUserDebugLine(start, hit_pos, color, 
                                     replaceItemUniqueId=self.debug_line_ids[i])
                else:
                    end = ray_tos[i]
                    color = [0.57, 0.57, 0.75]
                    pb_client.addUserDebugLine(start, end, color, 
                                     replaceItemUniqueId=self.debug_line_ids[i])
                    
        
    
    def get_distances(self, hit_results: List[Tuple]) -> np.ndarray:
        """Extract distances from LIDAR hit results."""
        distances = []
        for hit_result in hit_results:
            if hit_result[0] != -1:  # Hit detected
                distances.append(hit_result[2] * self._ray_length)
            else:
                distances.append(float('Inf'))

        return np.array(distances)
    
