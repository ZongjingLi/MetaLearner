import gymnasium as gym
import minihack

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
    OBJECT:('%', "corpse"), random
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
des_file = rooms_des


def init_render(name = 'Naxx'):
    fig = plt.figure(name, figsize=(10, 3))
    fig.canvas.manager.window.wm_geometry("+300+200")
    return fig

def render_pixels( pixels, delta = 0.01):
    plt.cla()
    plt.imshow(pixels)
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.pause(delta)

from nle import nethack
compass_actions = tuple(nethack.CompassDirection)
operate_actions = (
    nethack.Command.OPEN,
    nethack.Command.KICK,
    nethack.Command.CLOSE,
)

action_space = compass_actions + operate_actions

env = gym.make(
    "MiniHack-Navigation-Custom-v0",
    des_file = des_file,
    observation_keys = {"pixel","glyphs", "colors"},
    actions = action_space, 
    max_episode_steps = 1000
    )

def single_run_env(env, policy, render : str = False):
    return

done = False
env.reset()
init_render()

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action = action)
    done = terminated or truncated
    pixels = obs["pixel"]

    if not done:
        env.render()
        render_pixels(pixels, delta = 0.1)

plt.show()