import gymnasium as gym
import minihack
env = gym.make("MiniHack-River-v0")
print(env.observation_space)
env.reset() # each reset generates a new environment instance
print(env.step(0))  # move agent '@' north
env.render()