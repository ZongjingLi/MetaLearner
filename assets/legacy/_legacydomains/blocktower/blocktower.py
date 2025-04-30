'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-08-20 08:58:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-08-20 08:59:17
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch
import torch.nn as nn
from rinarak.dklearn.nn.mlp import FCBlock
from rinarak.utils.tensor import logit
from domains.utils import build_domain_executor