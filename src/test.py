import pybullet as p
import pybullet_data as pd
import numpy as np
import time

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF('plane.urdf')
robot_id = p.loadURDF('C:/Users/Mattia/Desktop/projects/Robotics/TurtleBot3 - PPO/src/utils/pybullet_data/robots/rex_description/urdf/rex.urdf', basePosition=[0,0,.2])

for i in range(p.getNumJoints(robot_id)):
    print(p.getJointInfo(robot_id, i)[12].decode("utf-8"))
    
try:
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()