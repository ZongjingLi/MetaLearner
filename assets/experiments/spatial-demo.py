# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-15 07:35:30
# @Last Modified by:   Melkor
# @Last Modified time: 2024-10-16 16:44:31
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from datasets.ccsp_dataset import *
from spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP, pairwise, get_sigma_embeds
from spatial.diffusion import training_loop, samples,  ScheduleLogLinear

dataset = Swissroll(np.pi/2, 5 * np.pi, 100)
schedule = ScheduleLogLinear(N=500, sigma_min=0.01, sigma_max=10)

constraints = {
	"online" : 1
}

model = PointEnergyMLP(constraints)
model.load_state_dict(torch.load("checkpoints/spiral_state.pth"))

def build_grid(resolution):
    w, h = resolution
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    x, y = torch.meshgrid([x, y])
    return torch.cat([x[..., None], y[..., None]], dim = -1)

W, H = 64, 64
xt = build_grid([W,H]).reshape([W*H,2])

batchsize = 300
steps = 30
cond = {"edges":[(i,"online") for i in range(batchsize)]}

*xs_t, x0 = samples(model, schedule.sample_sigmas(steps), gam=2, cond=cond, batchsize=batchsize, xt=None)
xs_t.append(x0)
sigmas = schedule.sample_sigmas(steps)

from IPython import display

# Visualization loop
for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        model.eval()
        energy = model.energies["online"](xt, sig)["energy"]

        fig = plt.figure("vis", figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.scatter(xt[:,0].detach(), xt[:,1].detach(), c=energy.detach(), cmap='viridis', alpha=0.5)
        ax.scatter(xs_t[i][:,0].detach(), xs_t[i][:,1].detach(), color='red', alpha=0.6)
        ax.plot(dataset.vals[:,0], dataset.vals[:,1], 'k-', linewidth=2)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_title(f'Step {i+1}, σ = {sig.item():.3f}')
        
        plt.pause(0.01)
        plt.cla()
        """
        display.display(plt.gcf())
        if i < len(sigmas) - 2:  # Don't clear the last frame
            display.clear_output(wait=True)
        plt.close()
        """

# Final visualization (won't be cleared)
fig = plt.figure("final", figsize=(5,5))
ax = fig.add_subplot(111)
energy = model.energies["online"](xt, sigmas[-1])["energy"]
ax.scatter(xt[:,0].detach(), xt[:,1].detach(), c=energy.detach(), cmap='viridis', alpha=0.5)
ax.scatter(x0[:,0].detach(), x0[:,1].detach(), color='red', alpha=0.6, label='Samples')
ax.plot(dataset.vals[:,0], dataset.vals[:,1], 'k-', linewidth=2, label='Target')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_title('Final Result')
ax.legend()
plt.show()