'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-08-03 16:28:53
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-08-03 16:28:55
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class GradientModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x