import pybullet as p
import open3d as o3d
from helchriss.envs.recorder import SceneRecorder
from helchriss.envs.base_env import BaseEnv
from helchriss.envs.contact import ContactModel
from helchriss.utils.data import ListDataset
from env.mechanism.pick_mechanism import PickBlockEnv
import torch
import torch.nn as nn
import random
import math
import sys


from datasets.ccsp_dataset import collate_graph_batch
from core.spatial.energy_graph import PointEnergyMLP
from core.spatial.diffusion import ScheduleLogLinear, training_loop

constraints = {
        "valid-grasp-block": 1,
    }

model    = PointEnergyMLP(constraints, dim = 3)
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)

class SamplerLearner:
    """take a dataset and a environment, and train the Sampler(nn.Module)"""
    def __init__(self, sampler, env, iterations = 32, epochs = 100, lr = 1e-3):
        super().__init__()    
        self.env = env
        assert isinstance(self.env, BaseEnv), "given environment is not a BaseEnv class"
        
        self.sampler = sampler # a torch sampler module to train
        assert isinstance(self.sampler, nn.Module), "sampler is not a nn.Module"
        
        # setup the env iteration for the given sampler learner.
        self.iterations = iterations # the number of iterations for the replay of the data to check if the continous seach possible
        self.dataset = ListDataset([])

        # setup the training data for the given construction
        self.epochs = epochs
        self.lr = lr
    
    def add_data(self,train_data): self.dataset.add(train_data)

    def replay(self):
        assert isinstance(self.env, PickBlockEnv), "given environment is not a BaseEnv class"
        for itrs in range(self.iterations):
            self.env.reset()
            self.env.reset_arm()
            obj_idx = env.block_id

            obj_set = self.env.get_object_attributes()
            #print(obj_set)
            attributes = self.env.get_object_attributes()
            #print(attributes)
            #print(attributes[obj_idx])
 
            # TODO: how to apply an actual mechanism
            self.env.pick_object(env.block_id)

            edge = {"edge" : [obj_idx, "valid-grasp-block"]}
            goal_success = self.env.check_goal()
            if goal_success:
                print("goal success")
                #self.add_data([variables, 1])
            else:
                print("pick up failed")
            self.env.remove_last_object()
        return
    
    def fit_dataset(self):
        assert isinstance(self.sampler, nn.Module), "sampler is not a nn.Module"
        loader = collate_graph_batch(self.dataset)
        trainer  = training_loop(loader, self.sampler, self.schedule, epochs=self.epochs)
        losses   = [ns.loss.item() for ns in trainer]
        return self.sampler


if __name__ == "__main__":

    save_dir = "outputs/mechansim"
    num_views = 8
    recorder = SceneRecorder(
        num_views = num_views,
        camera_distance = 1.32,
        camera_height = 1.35,
        target_point = [0.0, 0.0, 0.5]
    )

    

    env = PickBlockEnv(gui = False)
    sampler = PointEnergyMLP(constraints = constraints, dim = 3)

    sampler_learner = SamplerLearner(sampler = sampler, env = env, iterations=100)

    sampler_learner.replay()
