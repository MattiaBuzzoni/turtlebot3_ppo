# test_go_env.py
import time 
import os
import pybullet as p

from src.core.simulation import Simulation
from src.io.gamepad import xbox_one_pad
from src.model.robots.turtlebot import turtlebot
from src.gym.env.go_to.go_env import GoEnv

env = GoEnv(
    robot_model=turtlebot.TurtleBot3,  
    mark="1",
    target_position=None,
    obstacles_list=None,
    render=True,               
    debug=True,
)

# Test reset
obs = env.reset()
print("Observation shape:", len(obs))
print("Observation:", obs)
# Test step loop
for i in range(10000):

    action = env.action_space.sample()  
    print(action) # azione random
    if action[0] <= 0:
        print("ATTENTION: It's Backward.")
    time.sleep(.5)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {i} | reward: {reward:.3f} | done: {done} | info: {info}")
    if done:
        print("Episode finished!")
        break

env.close()
