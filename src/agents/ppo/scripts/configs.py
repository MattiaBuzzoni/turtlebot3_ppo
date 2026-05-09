# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example configurations using the PPO algorithm."""


def default():
    """Default configuration for PPO."""
    # General
    total_timesteps = 2_500_000
    timesteps_per_batch = 500
    max_timesteps_per_episode = 500
    n_updates_per_iteration = 50

    # Network
    lr = 2e-4

    gamma = 0.99 
    clip = 0.2 
    save_freq = 2
    use_gpu = True 

    return locals()


def go():
    """Configuration for the "go to" task."""
    locals().update(default())
    # Environment
    env = 'GoEnv'
    return locals()


# def gallop_ik():
#     """Configuration for Spotmicro gallop task based on inverse kinematics controller."""
#     locals().update(default())
#     # Environment
#     env = 'RexGalloping-v0'
#     max_length = 2000
#     steps = 1e6  # 1M
#     return locals()
#
#
# def gallop_ol():
#     """Configuration for Spotmicro gallop task based on open loop controller."""
#     locals().update(default())
#     # Environment
#     env = 'RexGalloping-v0'
#     max_length = 2000
#     steps = 2e6  # 2M
#     return locals()
#
#
# def turn_ol():
#     """Configuration for Spotmicro turn task."""
#     locals().update(default())
#     # Environment
#     env = 'RexTurn-v0'
#     max_length = 1000
#     steps = 1e6  # 1M
#     return locals()
#
#
# def turn_ik():
#     """Configuration for Spotmicro turn task."""
#     locals().update(default())
#     # Environment
#     env = 'RexTurn-v0'
#     max_length = 1000
#     steps = 1e6  # 1M
#     return locals()
#
#
# def standup_ol():
#     """Configuration for Spotmicro stand up task."""
#     locals().update(default())
#     # Environment
#     env = 'RexStandup-v0'
#     max_length = 500
#     steps = 1e6  # 1M
#     return locals()
#
#
# def poses_ik():
#     """Configuration for Spotmicro reach-a-pose task."""
#     locals().update(default())
#     # Environment
#     env = 'RexPoses-v0'
#     max_length = 1000
#     steps = 1e6  # 1M
#     return locals()
