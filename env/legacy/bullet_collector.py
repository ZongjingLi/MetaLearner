from bulletarm import env_factory
import pybullet as p
import pybullet_data
import numpy as np
import trimesh
import time
import os
from datetime import datetime
from bullet_utils import gather_object_info

def runDemo():
    env_config = {'render': True}

    #name = "close_loop_drawer_opening"
    #name = "block_stacking"
    name = "house_building_4"
    env = env_factory.createEnvs(0, f'{name}', env_config)

    obs = env.reset()

    done = False
    while not done:
        action = env.getNextAction()
        obs, reward, done = env.step(action)

        
        # Get object information including point clouds
        objs = gather_object_info()

runDemo()