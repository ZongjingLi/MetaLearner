# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 15:14:02
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 21:16:52
import gymnasium as gym
import minihack
from hack_utils import render_pixels, init_render
from hack_utils import DummyPolicy, run_policy

des_file = """
MAZE: "mylevel",' '
FLAGS: premapped
GEOMETRY:center, center
MAP
|-----     ------
|.....-- --.....|
|.T.T...-....K..|
|.......+.......|
|.T.T...-.......|
|.....-----.....|
|-----     ------
ENDMAP
BRANCH: (3,3,3,3),(4,4,4,4)
"""

rooms_des = """
LEVEL: "mylevel"
FLAGS: premapped

ROOM : "ordinary", lit, (1,1), (left, top), (10,5) {
    MONSTER: ('F', "red mold"), (0,0)
    TRAP:"hole",(4,4)
}
ROOM : "ordinary", lit, (5,3), (right, center), (15,15) {
    TRAP: random, random
    TRAP: random, random
    OBJECT:('%', "food ration"), random
    OBJECT:'*', (10,10)
    OBJECT :('"', "amulet of life saving"), random
    OBJECT:('%', "c orpse"), random
    OBJECT:('`', "statue"), (0,0), montype:"forest centaur", 1
    OBJECT:('(', "crystal ball"), (17,08), blessed, 5,name:"The Orb of Fate"
    OBJECT:('%',"egg"), (05,04), montype:"yellow dragon"
    SUBROOM : "ordinary", lit, (0,0), (3,3) {
    }
    MONSTER: ('F', "lichen"), (5,7)
    SINK: random
    FOUNTAIN: random
    ALTAR: random, random, random
    STAIR: random, down
}
ROOM : "ordinary", lit, (1,5), (left, bottom), (5,5) {
    MONSTER: ('F', "lichen"), (1,1)
    TRAP:"fire", random
    
}

RANDOM_CORRIDORS
"""
#des_file = rooms_des


from nle import nethack
compass_actions = tuple(nethack.CompassDirection)
operate_actions = (
    nethack.Command.OPEN,
    nethack.Command.KICK,
    nethack.Command.CLOSE,
)

action_space = compass_actions + operate_actions



def make_env():
    env = gym.make(
    "MiniHack-Navigation-Custom-v0",
    des_file = des_file,
    observation_keys = {"pixel","glyphs", "colors", "message"},
    actions = action_space, 
    max_episode_steps = 1000
    )
    return env

env = make_env()

policy = DummyPolicy(env)
results = run_policy(env, policy, render = 1)

print(results["episode_reward"])

