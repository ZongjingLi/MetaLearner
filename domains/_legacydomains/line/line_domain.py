'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-07-21 17:44:44
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-07-21 17:47:44
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch
import torch.nn as nn
from rinarak.dklearn.nn.mlp import FCBlock
from rinarak.utils.tensor import logit
from domains.utils import build_domain_executor

gc = 0.0
tc = 0.15

domain_name = "line_domain.txt"
domain_file = os.path.dirname(__file__) + "/" + domain_name
line_executor = build_domain_executor(domain_file)

class VectorModule(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.linear0 = FCBlock(128,2,state_dim, output_dim)

    def forward(self, x): return self.linear0(x)

line_executor.redefine_predicate(
        "pos", 
        lambda x: {**x,
                   "from": "pos", 
                   "set":x["end"], 
                   "end": x["pos"] if "pos" in x else x["state"]}
)
def greater_logits(x, y):
    # Reshape x to [n, 1] and y to [1, m] so we can broadcast
    x = x["state"].view(-1, 1)  # [n, 1]
    y = y["state"].view(1, -1)  # [1, m]
    
    diff = (x - y - gc) / tc  # This will automatically broadcast to [n, m]
    
    return diff

def lesser_logits(x, y):
    x = x["state"].view(-1, 1)  # [n, 1]
    y = y["state"].view(1, -1)  # [1, m]
    
    diff = (y - x - gc) / tc  # This will automatically broadcast to [n, m]
    return diff

line_executor.redefine_predicate(
        "left",
        lambda x: lambda y:{**x,
                   "from": "right", 
                   "set":x["end"], 
                   "end": greater_logits(x,y) }
)

line_executor.redefine_predicate(
        "right",
        lambda x: lambda y:{**x,
                   "from": "left", 
                   "set":x["end"], 
                   "end": lesser_logits(x,y) }
)
import matplotlib.pyplot as plt
def visualize(context, save_name = "line_repr"):
    plt.figure("line representation", figsize = (6, 6))
    plt.cla()
    plt.axis("off")
    plt.xlim([-1. ,1.])
    plt.ylim([-1. ,1.])
    plt.scatter(0,0, c = "blue")
    for i in context:
        cont = {0: context[i]}
        pos = line_executor.evaluate("(pos $0)", context = cont)["end"].detach()
        plt.scatter(pos[0], 0, c = "red", marker = "*")
        plt.text(pos[0], 0.0, f"[{i}]:{float(pos)}")

    plt.savefig(f"outputs/{save_name}.png")

line_executor.visualize = visualize