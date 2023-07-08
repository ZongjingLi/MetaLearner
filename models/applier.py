import torch
import torch.nn as nn

class Applier(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):return x

    def evaluate(self, state, goal):
        return