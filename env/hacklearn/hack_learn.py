# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 14:14:46
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 15:45:40
import gymnasium as gym
import minihack
from minihack import LevelGenerator

maze_map = """
--------------------
|.......|.|........|
|.-----.|.|.-----|.|
|.|...|.|.|......|.|
|.|.|.|.|.|-----.|.|
|.|.|...|....|.|.|.|
|.|.--------.|.|.|.|
|.|..........|...|.|
|.|--------------|.|
|..................|
--------------------
"""

lvl_gen = LevelGenerator(map = maze_map)
lvl_gen.set_start_pos((9,1))
lvl_gen.add_goal_pos((19,9))
lvl_gen.add_monster(name="minotaur",
    place=(19, 9))
# Add wand of death
lvl_gen.add_object("death", "/")

from hack_utils import get_action_compass
compass_actions = tuple(get_action_compass())
operate_actions = ()
action_space = compass_actions + operate_actions


env = gym.make(
    "MiniHack-Skill-Custom-v0",
    observation_keys = {"pixel"},
    des_file = lvl_gen.get_des(),
    actions = action_space,
    max_episode_steps = 1000
)

from hack_utils import run_policy, DummyPolicy

model = DummyPolicy(env)

run_policy(env, model, render = True, delta = 0.1)
