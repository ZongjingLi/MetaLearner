from domains.blockworld.blockworld_env import StackBlockWorldEnv

env = StackBlockWorldEnv(4, True)

env.reset()

env.render(mode = "rgb_array")

env.step([0,1])