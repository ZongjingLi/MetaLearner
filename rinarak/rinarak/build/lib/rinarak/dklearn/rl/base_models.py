import torch
import torch.nn as nn
import numpy as np

class RandomDiscreteModel(nn.Module):
    def __init__(self, action_num):
        super().__init__()
        self.pseudo_parameter = torch.nn.Parameter(torch.randn(1))
        self.action_num = action_num
    
    def forward(self, x):
        action_id = np.random.randint(0,self.action_num)
        pseudo_loss = self.pseudo_parameter.norm()
        return action_id, pseudo_loss

class RandomContinuousModel(nn.Module):
    def __init__(self, action_num):
        super().__init__()
    
    def forward(self, x):
        return x