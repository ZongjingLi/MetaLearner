'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-07-21 17:44:44
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-07-21 17:47:44
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn
from rinarak.dklearn.nn.mlp import FCBlock
from rinarak.utils.tensor import logit
from domains.utils import build_domain_executor

gc = 0.0
tc = 0.25

geometry_executor = build_domain_executor("domains/plane_domain.txt")

geometry_executor.redefine_predicate(
        "pos",
        lambda x: {**x,
                   "from": "pos", 
                   "set":x["end"], 
                   "end": x["pos"] if "pos" in x else x["state"]}
)

def greater_logits(x, y):
    assert x["end"].reshape([1,-1]).shape[1] == 1, x["end"].shape
    assert y["end"].reshape([1,-1]).shape[1] == 1, y["end"].shape
    return (x["end"].reshape([-1]) - y["end"].reshape([-1]) - 0 - gc) / tc 

def lesser_logits(x, y):
    assert x["end"].reshape([1,-1]).shape[1] == 1, x["end"].shape
    assert y["end"].reshape([1,-1]).shape[1] == 1, y["end"].shape
    return (y["end"].reshape([-1]) - x["end"].reshape([-1]) - 0 - gc) / tc 


def norm(x):
    assert x["end"].reshape([1,-1]).shape[1] == 2, x["end"].shape
    return torch.norm(x["end"], dim = 1)

geometry_executor.redefine_predicate(
        "length-f",
        lambda x: {**x,
                   "from": "left", 
                   "set":x["end"], 
                   "end":  norm(x)}
)

import matplotlib.pyplot as plt
def visualize(context, save_name = "geometry_repr"):
    plt.cla()
    plt.figure("geometry representation", figsize = (10,10))
    plt.axis("off")
    plt.scatter(0,0, c = "blue")
    for i in context:
        cont = {0: context[i]}
        pos = geometry_executor.evaluate("(pos $0)", context = cont)["end"].detach()
        length = geometry_executor.evaluate("(length $0)", context = cont)["end"].detach()
        plt.scatter(pos[0,0], pos[0,1], c = "red", marker = ".")
        plt.text(pos[0,0], pos[0,1], f"[{i}]")
        plt.text(pos[0,0] + 0.05, pos[0,1] - .1, f"length:{float(length)}")

    plt.savefig(f"outputs/{save_name}.png")

geometry_executor.visualize = visualize