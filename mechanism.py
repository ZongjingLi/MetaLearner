import pybullet as p
import open3d as o3d
from rinarak.envs.recorder import SceneRecorder
from rinarak.envs.base_env import BaseEnv
from env.mechanism.edge_mechanism import GripEdge
import math
import sys
import torch
import torch.nn as nn

from datasets.ccsp_dataset import collate_graph_batch
from core.spatial.energy_graph import PointEnergyMLP
from core.spatial.diffusion import ScheduleLogLinear, training_loop



constraints = {
        "valid-grasp": 1,
    }

model    = PointEnergyMLP(constraints, dim = 3, attr_dim = 6)
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)

class SamplerLearner:
    """take a dataset and a environment, and train the Sampler(nn.Module)"""
    def __init__(self, sampler, env, iterations = 32, epochs = 100, lr = 1e-3):
        super().__init__()    
        self.env = env
        assert isinstance(self.env, BaseEnv), "given environment is not a BaseEnv class"
        
        self.sampler = sampler # a torch sampler module to train
        assert isinstance(self.sampler, nn.Module()), "sampler is not a nn.Module"
        
        # setup the env iteration for the given sampler learner.
        self.iterations = iterations # the number of iterations for the replay of the data to check if the continous seach possible
        self.dataset = []

        # setup the training data for the given construction
        self.epochs = epochs
        self.lr = lr
    
    def add_data(self,train_data):
        self.dataset.append(train_data)

    def replay(self):
        for itrs in self.iterations:
            self.env.reset()
        return
    
    def fit_dataset(self):
        assert isinstance(self.sampler, nn.Module), "sampler is not a nn.Module"
        loader = None
        trainer  = training_loop(loader, self.sampler, self.schedule, epochs=self.epochs)
        losses   = [ns.loss.item() for ns in trainer]



if __name__ == "__main__":

    save_dir = "outputs/mechansim"
    num_views = 8
    recorder = SceneRecorder(
        num_views = num_views,
        camera_distance = 1.32,
        camera_height = 1.35,
        target_point = [0.0, 0.0, 0.5]
    )

for scene in range(10):
    env = GripEdge(gui = 0)#GripEdge(gui = True)

    gripper_orientation = p.getQuaternionFromEuler([0, math.pi, 0])

    n = 0
    sim_steps = 50
    
    import random
    for i in range(random.randint(1, 4)):
        pos_x = random.randint(45,45)/100.#(random.random()-0.5) * 0.8
        pos_y = random.randint(-45,45)/100.#(random.random()-0.5) * 0.8
        env.add_box([pos_x, pos_y, 0.7], size = [0.02,0.02,0.02])

    #cabinet_path = "assets/single_door_cabinet.urdf"  # Make sure this matches your saved file
    #initial_position = [0, -0.8, 0.65]
    #initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
    #cabinetId = p.loadURDF(cabinet_path, initial_position, initial_orientation, useFixedBase=1)
    #env.objects.append(cabinetId)

    recorder.record_scene_with_segmentation(output_dir="data/mechanism_data", frame_idx=scene, save=True)

    p.disconnect()
    #recorder.record_scene_with_segmentation(output_dir="data/mechanism_data", frame_idx=scene + 1, save=True)
    """
    env.control_gripper(1)
    recorder.record_scene_with_segmentation(output_dir="data/mechanism_data", frame_idx=scene, save=True)
    env.move_arm([pos_x, pos_y, 1.0], gripper_orientation)
    env.move_arm([pos_x, pos_y, 0.65], gripper_orientation)
    env.control_gripper(0)
    env.move_arm([pos_x, pos_y, 0.9], gripper_orientation)
    env.move_arm([pos_x, -pos_y, 0.9], gripper_orientation)
    env.move_arm([pos_x, -pos_y, 0.65], gripper_orientation)
    env.control_gripper(1)
    env.move_arm([pos_x, -pos_y, 0.9], gripper_orientation)
    
    #recorder.record_scene_with_segmentation(output_dir="data/mechanism_data", frame_idx=scene, save=True)
    """

    clouds = [o3d.io.read_point_cloud(f"/Users/sunyiqi/Documents/GitHub/Aluneth/data/mechanism_data/scene_frame_{scene}/point_clouds/merged_point_cloud.ply")]
    #clouds = [o3d.io.read_point_cloud(f"/Users/sunyiqi/Documents/GitHub/Aluneth/data/mechanism_data/scene_frame_0/point_clouds/view_{i}_points.ply") for i in range(num_views)]
    clouds = [o3d.io.read_point_cloud(f"/Users/sunyiqi/Documents/GitHub/Aluneth/data/mechanism_data/scene_frame_{scene}/point_clouds/segmented/merged_object_{i}.ply") for i in range(1,5)]
    #o3d.visualization.draw_geometries(clouds)    # Visualize point cloud    