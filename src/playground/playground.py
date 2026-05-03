"""Class to create a playground for testing the TurtleBot3."""

from __future__ import annotations

import time 
import os
import pybullet as p

from src.core.simulation import Simulation
from src.io.gamepad import xbox_one_pad
from src.model.robots.turtlebot import turtlebot

class Playground:

    def __init__(
        self,
        robot_model,
        mark,
        record_video,
        gamepad=False,
        pybullet_client=None,
    ):
        self._robot_model = robot_model
        self._mark = mark
        self._record_video = record_video
        self._gamepad = gamepad

        if self._gamepad:
            gamepad = xbox_one_pad.XboxGamepad()
            self._command_function = gamepad.get_command

        # Create the simulation
        self._create_simulation(False, pybullet_client, robot_model, mark, "plane", None)

    def _create_simulation(
        self,
        record_video,
        pybullet_client,
        robot_model,
        mark,
        terrain_type,
        terrain_id
    ):
        # Start simulation
        self._sim = Simulation(
            robot_model=robot_model,
            debug=True,
            terrain_id=terrain_id,
            terrain_type=terrain_type,
            record_video=record_video,
            mark=mark,
            render=True,
            pybullet_client=pybullet_client
        )

        # Setup UI
        self._ui = self._setup_ui_parameters()

    def _setup_ui_parameters(self):
        vx_id = self._sim.pybullet_client.addUserDebugParameter("Vx", -2., 2., 0.)
        wz_id = self._sim.pybullet_client.addUserDebugParameter("Wz", -2., 2., 0.)
        ui = {
            "terrain": self._sim.terrain.setup_ui_params(self._sim.pybullet_client),
            "equip": self.setup_equipment_ui_params(),
            "vx": vx_id,
            "wz": wz_id,
        }
        return ui
    
    def _update_world(self):
        refresh_terrain, terrain_id, terrain_type = self._sim.terrain.parse_ui_input(self._ui["terrain"],
                                                                                     self._sim.pybullet_client)
        args = {}
        if refresh_terrain:
            args = {
                "terrain_id": terrain_id,
                "terrain_type": terrain_type
            }
        return refresh_terrain, args

    def _parse_ctrl_input(self):
        if self._gamepad:
            # read gamepad input
            vx, _, wz = self._command_function()
        else:
            # read ui input
            vx = self._sim.pybullet_client.readUserDebugParameter(self._ui["vx"])
            wz = self._sim.pybullet_client.readUserDebugParameter(self._ui["wz"])
        return [vx, wz]

    def run(self):
        start_time = self._sim._get_time_since_reset()
        current_time = start_time
        try:
            while True:
                if not self._sim.pybullet_client.isConnected():
                    break
                # time.sleep(0.0008) # on some fast computer, works better with sleep on real robot?
                start_time_robot = current_time
                start_time_wall = time.time()
                # update the sim
                restart, args = self._update_world()
                if restart:
                    self._reset(args)
                    continue
                if self.parse_equipment_ui_params(self._ui["equip"]):
                    # show cam view
                    self._sim.robot.update_camera()
                # get controller generated action
                action = self._parse_ctrl_input()
                # apply action to robot
                self._sim.apply_step_action(action)
                self._sim.robot.update_lidar()
                current_time = self._sim._get_time_since_reset()
                expected_duration = current_time - start_time_robot
                actual_duration = time.time() - start_time_wall
                if actual_duration < expected_duration:
                    time.sleep(expected_duration - actual_duration)
                # print("actual_duration=", actual_duration)
        except self._sim.pybullet_client.error as e:
            os._exit(0)


    def _reset(self, args):
        if args is None:
            args = {
                "terrain_id": self._sim.terrain.terrain_id,
                "terrain_type": self._sim.terrain.terrain_type
            }
        self._sim.pybullet_client.resetSimulation()
        self._sim.pybullet_client.removeAllUserParameters()
        self._create_simulation(False, self._sim.pybullet_client, self._robot_model, self._mark, **args)

    def setup_equipment_ui_params(self):
        cam_ui = self._sim.pybullet_client.addUserDebugParameter("Show robot cam", 0, -1, 0)
        return cam_ui

    def parse_equipment_ui_params(self, ui):
        cam_flag = self._sim.pybullet_client.readUserDebugParameter(ui)
        if cam_flag % 2 != 0:
            return True
        return False


if __name__ == "__main__":
    playground = Playground(turtlebot.TurtleBot3, "1", record_video=False, gamepad=True)
    playground.run()
