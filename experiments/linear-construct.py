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

from datasets.ccsp_dataset import *
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

constraints = {"online" : 1}


dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=2048, collate_fn=collate_graph_batch)
model    = TimeInputEnergyMLP(hidden_dims=(16,128,128,128,128,16))
model    = PointEnergyMLP(constraints)
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
trainer  = training_loop(loader, model, schedule, epochs=10000)
losses   = [ns.loss.item() for ns in trainer]
torch.save(model.state_dict(),"checkpoints/state.pth")

batchsize = 300
model.load_state_dict(torch.load("checkpoints/state.pth"))

cond = {"edges":[(i,"online") for i in range(batchsize)]}
*xt, x0  = samples(model, schedule.sample_sigmas(20), gam=2, cond = cond, batchsize = batchsize)



plot_batch(x0)
plt.show()
"""


#plt.plot(dataset.vals[:,0], dataset.vals[:,1])
#plt.scatter(x0[:,0], x0[:,1])
plt.show()



dataset = Swissroll(np.pi/2, 5 * np.pi, 100)
schedule = ScheduleLogLinear(N=1000, sigma_min=0.01, sigma_max=10)
model =  TimeInputEnergyMLP(hidden_dims=(16,128,128,128,128,16))
model.load_state_dict(torch.load("states.pth"))

def build_grid(resolution):
    w, h = resolution
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    x, y = torch.meshgrid([x, y])
    return torch.cat([x[..., None], y[..., None]], dim = -1)
W, H = 64, 64
xt = build_grid([W,H]).reshape([W*H,2])
sigmas = schedule.sample_sigmas(20)
#xt = model.rand_input(100)
for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        model.eval()
        sigma_embeds = get_sigma_embeds(xt.shape[0], sig.squeeze()) # shape: b x 2
        nn_input = torch.cat([xt, sigma_embeds], dim=1)               # 
        energy = model.net(nn_input)
        plt.figure("vis", figsize=(5,5))
        plt.scatter(xt[:,0],xt[:,1],c = energy.detach())
        plt.plot(dataset.vals[:,0], dataset.vals[:,1])
        plt.pause(0.01)
        plt.cla()

plt.plot(dataset.vals[:,0], dataset.vals[:,1])

plt.show()
"""

"""create some default objects to visualize """
"""test the swissroll experiment

disk_xs, disk_ys, disk_zs = sample_sphere_surface(num_pts)
unit_disk = torch.stack([disk_xs, disk_ys, disk_zs])

square_xs, square_ys, square_zs = sample_square_region(num_pts)
unit_square = torch.stack([square_xs, square_ys, square_zs])


obj0 = GeometricObject(unit_disk, random_pose())
obj1 = GeometricObject(unit_disk, random_pose())
obj2 = GeometricObject(unit_square, random_pose())
obj3 = GeometricObject(unit_square, random_pose())

geom_graph = GeometricGraph([obj0, obj1, obj2, obj3], [(0, 1, "near"), (1, 2, "far")])

constraints = {
	"near" : 2,
	"far" : 2,
}

dataset = [geom_graph]

geom_graph.setup(False)
plot_geom_graph(geom_graph)
solution = constraint_ensemble.solve(geom_graph)
geom_graph.update_control_variable(solution.detach())
plot_geom_graph(geom_graph)
"""
