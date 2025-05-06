
# The env_factory provides the entry point to BulletArm
from bulletarm import env_factory


def runDemo():
  env_config = {'render': True}
  # The env_factory creates the desired number of PyBullet simulations to run in
  # parallel. The task that is created depends on the environment name and the
  # task config passed as input.
  env = env_factory.createEnvs(1, 'block_stacking', env_config)

  # Start the task by resetting the simulation environment.
  obs = env.reset()
  done = False
  while not done:
    # We get the next action using the planner associated with the block stacking
    # task and execute it.
    action = env.getNextAction()
    obs, reward, done = env.step(action)
  env.close()

import open3d as o3d
import pybullet as p
from helchriss.envs.base_env import BaseEnv
from helchriss.envs.gripper_env import GripperSimulator

class TAMPEnv(GripperSimulator):
  def __init__(self, render = True):
    self.render = render

tamp_env = TAMPEnv(render = True)
t