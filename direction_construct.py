# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-02 12:40:01
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-02 16:52:10
# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-14 09:26:27
# @Last Modified by:   Melkor
# @Last Modified time: 2024-10-16 16:41:40
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
    "north_of": 2,
    "south_of": 2,
    "east_of": 2,
    "west_of": 2,
    "northeast_of": 2,
    "northwest_of": 2,
    "southeast_of": 2,
    "southwest_of": 2
}

# Initialize direction dataset instead of RCC8Dataset
dataset = DirectionDataset(num_samples=100)
loader = DataLoader(dataset, batch_size=2048, collate_fn=collate_graph_batch)
model = PointEnergyMLP(constraints, dim=2)  # Changed dimension to 2
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
trainer = training_loop(loader, model, schedule, epochs=100)
losses = [ns.loss.item() for ns in trainer]
torch.save(model.state_dict(),"checkpoints/direction_state.pth")

batchsize = 3
#model.load_state_dict(torch.load("checkpoints/direction_state.pth", map_location="cpu"))

# Example directional constraintsb
cond = {"edges": [(0,1,"north_of"), (1,2,"east_of")]}
xt = torch.randn([1, batchsize, 2])
*xt, x0 = samples(model, schedule.sample_sigmas(20), gam=2, cond=cond, batchsize=batchsize,xt = xt)

print(x0.shape)

# Import direction executor instead of RCC8
from domains.direction.direction_domain import direction_executor

plt.ion()
for x in xt:
	state_dict = {0:{"state": x[0].detach()}, 1:{"state": x[0].detach()}}

	mat = direction_executor.evaluate( "(north $0 $1)", state_dict,)["end"]
	direction_executor.visualize(state_dict, mat)
	plt.pause(0.01)
	plt.close()
plt.ioff()
state_dict = {0:{"state": x0[0].detach()}, 1:{"state": x0[0].detach()}}
mat = direction_executor.evaluate( "(north $0 $1)", state_dict,)["end"]
direction_executor.visualize(state_dict,mat)
plt.show()