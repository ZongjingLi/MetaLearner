import gym
import minihack

des_file = """
LEVEL: "mylevel"
FLAGS: premapped
REGION: (0,0,20,80), lit, "ordinary"

ROOM: "ordinary" , lit, random, random, random {
    MONSTER: random, random
}
ROOM: "ordinary" , lit, random, random, random {
    MONSTER: ('F', "lichen"), random
    TRAP:"hole",(0,0)
}
ROOM: "ordinary" , lit, random, random, random {
    MONSTER: ('F', "red mold"), (0,0)
    TRAP: random, random
} 


RANDOM_CORRIDORS
"""
env = gym.make(
    "MiniHack-Navigation-Custom-v0",
    des_file=des_file,
    max_episode_steps=50,
)

from minihack.envs import register


env.reset() # each reset generates a new environment instance
import numpy as np
import time

done = False
reward = [0.0]
while not done:
    action = env.action_space.sample()
    obs,r,done,rs = env.step(action)  # move agent '@' north
    reward.append(r + reward[-1])
    #env.render()
    #print(obs["glyphs"].shape)
print(reward)
import matplotlib.pyplot as plt
print(obs)
plt.imshow(obs["glyphs_crop"])
plt.show()