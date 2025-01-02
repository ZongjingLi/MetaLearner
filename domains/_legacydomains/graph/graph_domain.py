'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-09-08 12:15:33
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-09-08 12:23:34
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

domain_name = "graph_domain.txt"
domain_file = os.path.dirname(__file__) + "/" + domain_name
graph_executor = build_domain_executor(domain_file)

print(graph_executor.domain)


graph_executor.redefine_predicate(
        "state",
        lambda x: {**x,
                   "from": "state", 
                   "set":x["end"], 
                   "end": x["state"] if "state" in x else x["state"]}
)


graph_executor.redefine_predicate(
        "left",
        lambda x: lambda y:{**x,
                   "from": "right", 
                   "set":x["end"], 
                   "end": True}
)


import matplotlib.pyplot as plt