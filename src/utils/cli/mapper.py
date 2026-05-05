from src.agents.ppo.ppo import PPO
from src.model.robots.turtlebot import turtlebot


ROBOTS = {
    'turtlebot3': turtlebot.TurtleBot3
}

AGENTS = {
    'ppo': PPO
}
