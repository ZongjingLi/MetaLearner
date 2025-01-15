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
from domains.rcc8.rcc8_data import RCC8Dataset
from spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP
from spatial.diffusion import training_loop, samples,  ScheduleLogLinear, ScheduleSigmoid


num_pts = 1000
pose_dim = 3

def random_pose(space_scale = 10.0, scale_range = (0.5, 1.0)):
    smin, smax = scale_range[0], scale_range[1]
    return torch.cat([
        (torch.rand([2]) - 0.5) * (2 * space_scale),
        torch.zeros([1]),
        torch.randn(3),
        torch.rand([1]) * (smax - smin) + smin
        ])[:pose_dim]


def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

constraints = {
        "externally_connected": 2,
        "partial_overlap": 2,
        "equal": 2,
        "disconnected": 2,
        "tangential_proper_part": 2,
        "non_tangential_proper_part": 2,
        "tangential_proper_part_inverse": 2,
        "non_tangential_proper_part_inverse": 2
    }


dataset  = RCC8Dataset(num_samples = 100)
loader   = DataLoader(dataset, batch_size=2048, collate_fn=collate_graph_batch)
#model    = TimeInputEnergyMLP(dim = 3,hidden_dims=(16,128,128,128,128,16))
model    = PointEnergyMLP(constraints, dim = 3)
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
trainer  = training_loop(loader, model, schedule, epochs=0000)
losses   = [ns.loss.item() for ns in trainer]
torch.save(model.state_dict(),"checkpoints/circular_state.pth")

batchsize = 3
#model.load_state_dict(torch.load("checkpoints/state.pth"))

cond = {"edges":[(0,1,"disconnected"), (1,0,"partial_overlap")]}
*xt, x0  = samples(model, schedule.sample_sigmas(20), gam=2, cond = cond, batchsize = batchsize)

print(x0.shape)

from domains.rcc8.rcc8_domain import rcc8_executor

plt.ion()
for x in xt:
    rcc8_executor.visualize({0:{"state" : x.detach()}})
    plt.pause(0.01)
    plt.close()

rcc8_executor.visualize({0:{"state" : x0.detach()}})
plt.show()