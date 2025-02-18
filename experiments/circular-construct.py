# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-14 09:26:27
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-15 11:11:12
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from datasets.ccsp_dataset import collate_graph_batch, DataLoader
from domains.rcc8.rcc8_data import RCC8Dataset
from core.spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP
from core.spatial.diffusion import training_loop, samples,  ScheduleLogLinear, ScheduleSigmoid


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


dataset  = RCC8Dataset(num_samples = 300)
loader   = DataLoader(dataset, batch_size=2048, collate_fn=collate_graph_batch)

model    = PointEnergyMLP(constraints, dim = 3)
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
#trainer  = training_loop(loader, model, schedule, epochs=1000)
#losses   = [ns.loss.item() for ns in trainer]
#torch.save(model.state_dict(),"checkpoints/circular_state.pth")4
batchsize = 4
checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "circular_state.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only = False))

#cond = {"edges":[(0,1,"partial_overlap"), (1,2,"partial_overlap"), (0, 3, "equal")]}
#ond = {"edges":[(0,1,"disconnected")]}
#cond = {"edges":[(0,1,"partial_overlap")]}
#cond = {"edges":[(0,1,"externally_connected")]}
#cond = {"edges":[(0,1,"partial_overlap"), (1,2,"disconnected"),(2,3,"non_tangential_proper_part")]}


cond = {"edges":[
(0,1,"non_tangential_proper_part"),
(2,0,"externally_connected"),
]} # Venn Diagram

cond = {"edges":[
(0,1,"non_tangential_proper_part"),
(0,2,"non_tangential_proper_part"),
(1,2, "partial_overlap"),
(1,3, "non_tangential_proper_part"),
(2,3, "non_tangential_proper_part"),
]} # Venn Diagram
#cond = {"edges":[(0,1,"externally_connected"), (1,2,"disconnected"),(2,3,"non_tangential_proper_part")]}


xt = torch.randn([1,batchsize,3]) 
*xt, x0  = samples(model, schedule.sample_sigmas(320), gam=2, cond = cond, batchsize = batchsize, xt = xt)

print("Solution:")
print(x0[0].cpu().detach().numpy())

from domains.rcc8.rcc8_domain import rcc8_executor

#plt.ion()
#for x in xt:

    #rcc8_executor.visualize({0:{"state": x[0].cpu().detach()}})
    #plt.pause(0.01)
    #plt.close()
#plt.ioff()
context = {
    0 : {"state" : x0[0].cpu().detach()},
    1 : {"state" : x0[0].cpu().detach()}
}

res = rcc8_executor.evaluate("(partial_overlap $0 $1)", context)["end"].detach()

#rcc8_executor.visualize(context, res)
rcc8_executor.visualize({0 : {"state" : x0[0].cpu().detach()},})
plt.show()