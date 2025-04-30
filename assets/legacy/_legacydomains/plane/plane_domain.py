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

gc = 1.0
tc = 0.1

domain_name = "plane_domain.txt"
domain_file = os.path.dirname(__file__) + "/" + domain_name
plane_executor = build_domain_executor(domain_file)

plane_executor.redefine_predicate(
        "pos",
        lambda x: {**x,
                   "from": "pos", 
                   "set":x["end"], 
                   "end": x["pos"] if "pos" in x else x["state"]}
)
def norm(x):return torch.norm(x["end"], dim = 1)

def diff(x, y):
    x = x["state"].view([-1,2])
    n, _ = x.shape
    y = y["state"].view([-1,2])
    m, _ = y.shape
    return x.unsqueeze(1).repeat([1,m,1]) - y.unsqueeze(0).repeat([n,1,1])

plane_executor.redefine_predicate(
        "length-f",
        lambda x: {**x,
                   "from": "length", 
                   "set":x["end"], 
                   "end":  norm(x)}
)

sigma = 1.0
plane_executor.redefine_predicate(
    "near",
    lambda x: lambda y: {
        **x,
        "from" : "near",
        "set" : x["end"],
        "end":  logit(torch.exp(torch.norm(diff(x, y), dim = -1)/(2 * sigma)))
    }
)

plane_executor.redefine_predicate(
    "far",
    lambda x: lambda y: {
        **x,
        "from" : "far",
        "set" : x["end"],
        "end": logit(torch.exp(-torch.norm(diff(x, y), dim = -1)/(2 * sigma)))
    }
)

def angle(x, y, eps = 1e-6):
    # Reshape x to [n, 1] and y to [1, m] for broadcasting
    x = x.view(-1, 2)  # [n, 1]
    y = y.view(2, -1)  # [1, m]

    # Compute the dot product between x and y
    dot_product = torch.matmul(x , y)  # Element-wise multiplication, will broadcast to [n, m]

    # Compute the L2 norms of x and y
    x_norm = torch.norm(x, dim=1)  # Scalar L2 norm of x
    y_norm = torch.norm(y, dim=0)  # Scalar L2 norm of y


    # Compute the cosine similarity
    cosine_sim = dot_product / (x_norm * y_norm + 1e-8)  # Add epsilon for numerical stability
    
    #return cosine_sim
    
    # Compute the angle between the two vectors using the dot product formula
    angle = torch.acos( cosine_sim )
    angle_degrees = angle * 180 / torch.pi
    return angle_degrees

up = torch.tensor([0.0, 1.0])
down = torch.tensor([0.0, -1.])
right = torch.tensor([-1.0, 0.0])
left = torch.tensor([1.0, 0.0])
theshold = 45.

plane_executor.redefine_predicate(
    "right-of",
    lambda x: lambda y: {
        **x,
        "from" : "right",
        "set" : x["end"],
        "end": logit( angle(right, x["state"] - y["state"]) < theshold )
    }
)

plane_executor.redefine_predicate(
    "left-of",
    lambda x: lambda y: {
        **x,
        "from" : "left",
        "set" : x["end"],
        "end": logit( angle(right, x["state"] - y["state"]) < theshold )
    }
)

plane_executor.redefine_predicate(
    "above",
    lambda x: lambda y: {
        **x,"from" : "above","set" : x["end"],
        "end": logit( angle(up, x["state"] - y["state"]) < theshold )
    })

plane_executor.redefine_predicate(
    "below",
    lambda x: lambda y: {
        **x,"from" : "below","set" : x["end"],
        "end": logit( angle(down, x["state"] - y["state"]) < theshold )
    })


import matplotlib.pyplot as plt
def visualize(context, save_name = "plane_repr"):
    plt.figure("plane representation", figsize = (6,6))
    plt.cla()
    plt.axis("off")
    plt.xlim([-1. ,1.])
    plt.ylim([-1. ,1.])
    plt.scatter(0,0, c = "blue")
    for i in context:
        cont = {0: context[i]}
        pos = plane_executor.evaluate("(pos $0)", context = cont)["end"].detach()
        length = plane_executor.evaluate("(length $0)", context = cont)["end"].detach()
        plt.scatter(pos[0,0], pos[0,1], c = "red", marker = ".")
        plt.text(pos[0,0], pos[0,1], f"[{i}]")
        plt.text(pos[0,0] + 0.05, pos[0,1] - .1, f"length:{float(length)}")

    plt.savefig(f"outputs/{save_name}.png")

plane_executor.visualize = visualize