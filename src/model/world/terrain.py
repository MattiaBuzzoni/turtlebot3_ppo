"""Build the environment elements for the simulation."""\

import pybullet_data as pd
import pybullet as p
from typing import Any
# import random

ROBOT_INIT_POSITION_OFFSET = {
    "plane": 0,
}


class Terrain:

    def __init__(
        self, 
        terrain_type: str, 
        terrain_id: int, 
        columns: int = 256, 
        rows: int = 256
    ) -> None:
        self.terrain_type = terrain_type
        self.terrain_id = terrain_id
        self.columns = columns
        self.rows = rows
        self.terrain_shape = None
        self.id: int | None = None

    def generate_terrain(
        self, 
        pybullet_client: Any, 
        height_perturbation_range: float = 0.06
    ) -> None:
        """Generate the ground of the environment base on the terrain type selected."""
        pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        pybullet_client.configureDebugVisualizer(pybullet_client.COV_ENABLE_RENDERING, 0)

        if self.terrain_type == 'plane':
            self.id = pybullet_client.loadURDF("plane.urdf")
            pybullet_client.changeVisualShape(self.id, -1, rgbaColor=[.5, .5, .5, 1])
    
    @staticmethod
    def setup_ui_params(pybullet_client: Any) -> dict[str, Any]:
        plane = pybullet_client.addUserDebugParameter("Plane", 0, -1, 0)
        ui = {
            "plane": plane,
        }

        return ui
    
    def parse_ui_input(
        self, 
        ui: dict[str, Any], 
        pybullet_client: Any
    ) -> tuple[bool, int | None, str]:
        reset_sim = False
        terrain_type = None
        terrain_id = None

        if pybullet_client.readUserDebugParameter(ui["plane"]):
            terrain_type = "plane"
            terrain_id = None
            reset_sim = True
        else:
            terrain_type = self.terrain_type
            terrain_id = self.terrain_id
            reset_sim = False

        return reset_sim, terrain_id, terrain_type
    
    def get_terrain_z_offset(self):
        if self.terrain_type in ROBOT_INIT_POSITION_OFFSET.keys():
            return ROBOT_INIT_POSITION_OFFSET[self.terrain_type]
        
        return ROBOT_INIT_POSITION_OFFSET[f"{self.terrain_type}_{self.terrain_id}"]
    