import sys
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.gym
from src.utils.cli import flags


@click.group()
def cli():
    pass


@cli.command()
@click.option('--env', '-e', required=True, help="The Environment name.",
              type=click.Choice([e for e in flags.ENV_ID_TO_ENV.keys()], case_sensitive=True))
@click.option('--param', '-p', type=(str, float), help="The Environment's param(s).", multiple=True)
@click.option('--flag', '-f', type=(str, bool), help="The Environment's flag(s).", multiple=True)
@click.option('--log-dir', '-log', '-l', required=True, help="The path where the log directory will be created.")
@click.option('--debug', '-d', type=bool, default=False, help="Demo: 1 agent, render enabled.")
@click.option('--robot', '-r', default="ghost", help="The robot you want to use.",
              type=click.Choice(flags.SUPPORTED_ROBOTS))
@click.option('--terrain', '-t', default="plane", help="Set the simulation terrain.",
              type=click.Choice(flags.TERRAIN_TYPE))
@click.option('--mark', '-m', default="1", help="Set the robot mark.")
@click.option('--agent', '-a', default="ppo", help="The agent you want to use.",
              type=click.Choice(flags.SUPPORTED_AGENTS))
def policy(env, param, flag, log_dir, debug, robot, terrain, mark, agent):
    # import locally the PolicyPlayer to avoid the pyBullet loading at every cli command
    from src.core.policy_player import PolicyPlayer
    # parse input args
    args, robot_obj, agent_obj = _parse_input(param, flag, terrain, mark, robot, agent)
    # run the Policy Player
    PolicyPlayer(env, robot_obj, debug, args, log_dir, agent_obj).play()


@cli.command()
@click.option('--env', '-e', required=True, help="The Environment name.",
              type=click.Choice([e for e in flags.ENV_ID_TO_ENV.keys()], case_sensitive=True))
@click.option('--param', '-p', type=(str, float), help="The Environment's param(s).", multiple=True)
@click.option('--flag', '-f', type=(str, bool), help="The Environment's flag(s).", multiple=True)
@click.option('--log-dir', '-log', '-l', required=True, help="The path where the log directory will be created.")
@click.option('--debug', '-d', type=bool, default=False, help="Demo: 1 agent, render enabled.")
@click.option('--robot', '-r', default="turtlebot3", help="The robot you want to use.",
              type=click.Choice(flags.SUPPORTED_ROBOTS))
@click.option('--terrain', '-t', default="plane", help="Set the simulation terrain.",
              type=click.Choice([t for t in flags.TERRAIN_TYPE.keys()]))
@click.option('--mark', '-m', default="1", help="Set the robot mark (version).")
@click.option('--agent', '-a', default="ppo", help="The agent you want to use.",
              type=click.Choice(flags.SUPPORTED_AGENTS))
def train(env, param, flag, log_dir, debug, robot, terrain, mark, agent):
    # import locally the Trainer to avoid the pyBullet loading at every cli command
    from src.core.trainer import Trainer
    # parse input args
    args, robot_obj, agent_obj = _parse_input(param, flag, terrain, mark, robot, agent)
    # run the Trainer
    Trainer(env, args, robot_obj, debug, log_dir, agent_obj).train()


@cli.command()
@click.option('--robot', '-r', default="ghost", help="The robot you want to use.",
              type=click.Choice(flags.SUPPORTED_ROBOTS))
@click.option('--mark', '-m', default="1", help="Set the robot mark (version).")
@click.option('--record-video', '-rec', type=bool, default=False, help="Record a video clip (mp4).")
@click.option('--controller', '-g', type=str, default='keyboard', help="Control the robot using a gamepad or the keyboard.")
def playground(robot, mark, record_video, controller):
    from src.playground.playground import Playground
    from src.utils.cli import mapper
    args = _parse_mark(mark)
    args["robot_model"] = mapper.ROBOTS[robot]
    args["record_video"] = record_video
    args["controller"] = controller
    Playground(**args).run()


def _parse_input(param, flag, terrain, mark, robot, agent):
    terrain_args = _parse_terrain(terrain)
    args = _parse_args(param + flag)
    args.update(terrain_args)
    args.update(_parse_mark(mark))
    robot_obj  = _parse_robot(robot)
    agent_obj = _parse_agent(agent)
    return args, robot_obj, agent_obj


def _parse_mark(mark_id):
    return {'mark': mark_id}


def _parse_terrain(terrain_id):
    arg = {
        "terrain_type": flags.TERRAIN_TYPE[terrain_id],
        "terrain_id": terrain_id
    }
    return arg


def _parse_args(params):
    args = {}
    for k, v in params:
        args[k] = v
    return args


def _parse_robot(robot):
    from src.utils.cli import mapper
    return mapper.ROBOTS[robot]


def _parse_agent(agent):
    from src.utils.cli import mapper
    return mapper.AGENTS[agent]

if __name__ == "__main__":
    cli()
