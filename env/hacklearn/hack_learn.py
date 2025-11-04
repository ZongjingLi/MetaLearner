# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 14:14:46
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 15:45:40
import torch
import minihack
from nle import nethack
import gymnasium as gym
from minihack import LevelGenerator
from hack_utils import run_policy, DummyPolicy
from helchriss.algs.reinforce import Reinforce
import matplotlib.pyplot as plt

maze_map = """
--------------------
|.......|..........|
|.......|.|.-----|.|
|.......|.|......|.|
|-|.....|.|-----.|.|
|.|...........|.|.|.|
|.|---------.|.|.|.|
|.|..........|...|.|
|.|--------------|.|
|..................|
--------------------
"""

lvl_gen = LevelGenerator(map = maze_map)
lvl_gen.set_start_pos((1,1))
lvl_gen.add_goal_pos((7,5))
lvl_gen.add_gold(1,(4,4))
lvl_gen.add_gold(1,(3,5))
lvl_gen.add_object("dagger", ")", (3,3))
lvl_gen.add_object("dagger", ")", (4,4))
lvl_gen.add_object("light", "/", (6,5))
lvl_gen.add_object("apple", "%", (7,3))
lvl_gen.add_sink((2,2))
#slvl_gen.add_sink((2,2))

#lvl_gen.add_object("apple", "%" )
#lvl_gen.add_monster(name="minotaur",place=(1, 9))
#lvl_gen.add_object("death", "/", (5,5))

rwd_manager = minihack.RewardManager()
rwd_manager.add_eat_event("apple")
rwd_manager.add_wield_event("dagger")
rwd_manager.add_wield_event("light")
rwd_manager.add_location_event("sink", reward= -1.)
rwd_manager.add_location_event("goal_pos", reward=3.)
#rwd_manager.add_coordinate_event((3,3), 10., terminal_required=False, terminal_sufficient=True)
#rwd_manager.add_location_event("gold", reward = 5.0)

from hack_utils import get_action_compass
compass_actions = tuple(get_action_compass())
operate_actions = tuple([nethack.Command.PICKUP])

action_space = compass_actions + operate_actions



env = gym.make(
    "MiniHack-Skill-Custom-v0",
    observation_keys = {"pixel", "glyphs","chars", "message"},
    des_file = lvl_gen.get_des(),
    actions = action_space,
    max_episode_steps = 100,
    reward_manager=rwd_manager
)


model = DummyPolicy(env)


model.load_state_dict(torch.load("test_reinforce_maze.pth"))
reward = run_policy(env, model, render = True, delta = 0.5)
plt.close()
#env = gym.make("CartPole-v1")

#policy = SimplePolicy(env)
learner = Reinforce(model=model, env=env, run_policy= run_policy, gamma=0.99, extra = 1)

rewards = learner.train(3000)
torch.save(learner.model.state_dict(), "test_reinforce_maze.pth")
plt.plot(rewards)
plt.show()


reward = run_policy(env, model, render = True, delta = 0.1)
