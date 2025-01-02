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
from rinarak.knowledge.measure import Measure
from rinarak.utils.tensor import logit
from domains.utils import build_domain_executor


domain_name = "box_domain.txt"
domain_file = os.path.dirname(__file__) + "/" + domain_name
box_executor = build_domain_executor(domain_file)
box_measure = Measure(2, temperature = 0.02)

box_executor.redefine_predicate(
        "box",
        lambda x: {**x,
                   "from": "box", 
                   "set":x["end"], 
                   "end": x["box"] if "box" in x else x["state"]}
)
def contain_logits(x, y):
    x_boxes = x["state"].reshape([-1,4])
    n, _ = x_boxes.shape
    y_boxes = y["state"].reshape([-1,4])
    m, _ = y_boxes.shape

    x_boxes = x_boxes.unsqueeze(1).repeat(1,m,1)
    y_boxes = y_boxes.unsqueeze(0).repeat(n,1,1)

    return box_measure.entailment(x_boxes, y_boxes)

box_executor.redefine_predicate(
        "contain",
        lambda x: lambda y:{**x,
                   "from": "contain", 
                   "set":x["end"], 
                   "end": contain_logits(x,y) }
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def visualize(context, save_name = "box_repr"):
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)
    plt.cla()
    plt.axis("off")
    plt.xlim([-1. ,1.])
    plt.ylim([-1. ,1.])

    for i in context:
        cont = {0: context[i]}
        box = box_executor.evaluate("(box $0)", context = cont)["end"].reshape([4]).detach()
        # Create a Rectangle patch
        center_x, center_y = box[:2]
        offset = box[2:] / 2.
        rect = patches.Rectangle(
            [center_x - offset[0], center_y - offset[1]],
            offset[0] * 2,offset[1] * 2, linewidth=1, edgecolor='r', facecolor='none')
        plt.text(center_x, center_y, f"${i}")

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig(f"outputs/{save_name}.png")

box_executor.visualize = visualize