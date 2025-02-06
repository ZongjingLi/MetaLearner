# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-14 09:26:27
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-04 05:13:59
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from datasets.ccsp_dataset import collate_graph_batch, DataLoader
from domains.direction.direction_data import DirectionDataset  # Changed to DirectionDataset
from core.spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP
from core.spatial.diffusion import training_loop, samples, ScheduleLogLinear, ScheduleSigmoid

num_pts = 1000
pose_dim = 2  # Changed to 2 since direction domain uses 2D points

def generate_grid_edges(n, m):
    edges = []

    for i in range(n):
        for j in range(m):
            current = i * m + j

            # North edge
            if i > 0:
                north = (current, (i - 1) * m + j, "north")
                edges.append(north)

            # South edge
            if i < n - 1:
                south = (current, (i + 1) * m + j, "south")
                edges.append(south)

            # East edge
            if j < m - 1:
                east = (current, i * m + (j + 1), "east")
                edges.append(east)

            # West edge
            if j > 0:
                west = (current, i * m + (j - 1), "west")
                edges.append(west)

    return {"edges": edges}



def random_pose(space_scale = 10.0):
    # Simplified for 2D points
    return torch.cat([
        (torch.rand([2]) - 0.5) * (2 * space_scale),
    ])[:pose_dim]

def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Updated constraints for directional relations
constraints = {
    "north": 2,
    "south": 2,
    "east": 2,
    "west": 2,
    "northeast": 2,
    "northwest": 2,
    "southeast": 2,
    "southwest": 2
    }
# Initialize direction dataset instead of RCC8Dataset
plt.ion()
dataset = DirectionDataset(num_samples=100)
loader = DataLoader(dataset, batch_size=1024, collate_fn=collate_graph_batch)
model = PointEnergyMLP(constraints, dim=2)  # Changed dimension to 2
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
#trainer = training_loop(loader, model, schedule, epochs=4)
#losses = [ns.loss.item() for ns in trainer]
#plt.plot(losses)
#torch.save(model.state_dict(),"checkpoints/direction_state.pth")


model.load_state_dict(torch.load("checkpoints/direction_state.pth", map_location="mps", weights_only = False))

# Example directional constraintsb
cond = {"edges": [(0,1,"south"), (1,2,"west"), (2,3,"west"), (1,4,"south")]}
#cond = {"edges": [(0,1,"north"), (0,2,"south"), (0,3,"east"), (0,4,"west")]}
cond = generate_grid_edges(4,4)
batchsize = 3 * 7

# grid structure
xt = torch.randn([1, batchsize, 2]) 
*xt, x0 = samples(model, schedule.sample_sigmas(320), gam=2, cond=cond, batchsize=batchsize,xt = xt)


# Import direction executor instead of RCC8
from domains.direction.direction_domain import direction_executor
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

# Create a directory to store individual images

# Initialize a list to store the file paths of saved images
image_files = []

# Enable interactive mode if process is set
process = 1

if process:
    plt.ioff()  # Turn off interactive mode to capture frames properly
    from tqdm import tqdm
    for idx, x in tqdm(enumerate(xt)):
        state_dict = {0: {"state": x[0].detach()}, 1: {"state": x[0].detach()}}
        mat = direction_executor.evaluate("(north $0 $1)", state_dict)["end"]
        
        # Create the figure

        fig, ax = direction_executor.visualize(state_dict)
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        # Save the current figure as an image
        image_path = f'outputs/frame_{idx:03d}.png'
        plt.savefig(image_path)
        image_files.append(image_path)

        plt.close(fig)  # Close the figure after saving the image

# Add the final state
state_dict = {0: {"state": x0[0].detach()}, 1: {"state": x0[0].detach()}}
mat = direction_executor.evaluate("(north $0 $1)", state_dict)["end"]


fig, ax = direction_executor.visualize(state_dict)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

# Save the final figure
final_image_path = 'outputs/final_frame.png'
plt.savefig(final_image_path)
image_files.append(final_image_path)


# Load saved images and compile them into a GIF
frames = [imageio.imread(img) for img in image_files]
imageio.mimsave('outputs/visualization.gif', frames, duration=0.5)



