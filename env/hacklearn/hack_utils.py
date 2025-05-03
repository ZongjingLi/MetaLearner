# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 15:14:00
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 15:40:19
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import minihack

def init_render(name = 'Naxx'):
    fig = plt.figure(name, figsize=(10, 3))
    fig.canvas.manager.window.wm_geometry("+300+200")
    return fig

def render_pixels(pixels, delta = 0.01):
    plt.cla()
    plt.imshow(pixels)
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.pause(delta)

def get_action_compass():
	from nle import nethack
	return nethack.CompassDirection

def run_policy(env, policy, render : str = False, delta = 0.1):
    done = False
    episode_reward = 0.0
    obs, info = env.reset()
    init_render()
    while not done:
        action = policy.sample(obs)
        obs, reward, terminated, truncated, info = env.step(action = action)
        done = terminated or truncated
        episode_reward += reward
        if not done and render:
            pixels = obs["pixel"]
            render_pixels(pixels, delta = delta)
            env.render()
    return episode_reward

class DummyPolicy:

	def __init__(self, env):
		self.env = env

	def sample(self, obs):
		return self.env.action_space.sample()