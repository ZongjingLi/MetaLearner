# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-03 03:49:31
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-03 15:47:16


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from datasets.ccsp_dataset import *
from core.spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP
from core.spatial.diffusion import training_loop, samples,  ScheduleLogLinear, ScheduleSigmoid

#count  = 0
#for i in range(2, 6):
#    count += i*2
#count /= 5
#print(count) = 5.6


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

constraints = {"online" : 1}


dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=32, collate_fn=collate_graph_batch)
model    = TimeInputEnergyMLP(hidden_dims=(16,128,128,128,128,16))
model    = PointEnergyMLP(constraints)
#model.load_state_dict(torch.load("checkpoints/state.pth"))
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
#trainer  = training_loop(loader, model, schedule, epochs=5)
#losses   = [ns.loss.item() for ns in trainer]
#torch.save(model.state_dict(),"checkpoints/state.pth")

batchsize = 200
model.load_state_dict(torch.load("checkpoints/state.pth", map_location = "mps"))



cond = {"edges":[(i,"online") for i in range(batchsize // 1)]}
xt = torch.randn([1, batchsize, 2])
*xt, x0  = samples(model, schedule.sample_sigmas(100), gam=2, cond = cond, batchsize = batchsize, xt = xt)

plt.ion()
for x in xt:    
    plot_batch(x[0].cpu().detach())
    plt.pause(0.1)
    plt.cla()
plt.ioff()
plot_batch(x0[0].cpu().detach())
plt.show()

