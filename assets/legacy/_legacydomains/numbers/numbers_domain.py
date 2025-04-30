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
tc = 0.25

domain_name = "numbers_domain.txt"
domain_file = os.path.dirname(__file__) + "/" + domain_name
numbers_executor = build_domain_executor(domain_file)

class VectorModule(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.linear0 = FCBlock(128,2,state_dim, output_dim)

    def forward(self, x): return self.linear0(x)

numbers_executor.redefine_predicate(
        "value",
        lambda x: {**x,
                   "from": "value", 
                   "set":x["end"], 
                   "end": x["value"] if "value" in x else x["state"]}
)

numbers_executor.redefine_predicate(
        "right",
        lambda x: lambda y:{**x,
                   "from": "right", 
                   "set":x["end"], 
                   "end": (x["end"] - y["end"] - 1 - gc) / tc}
)

numbers_executor.redefine_predicate(
        "left",
        lambda x: lambda y:{**x,
                   "from": "left", 
                   "set":x["end"], 
                   "end": (y["end"] - x["end"] - 1 - gc) / tc}
)