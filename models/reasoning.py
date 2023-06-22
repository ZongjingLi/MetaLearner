import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, quries):
        return quries

class MetaReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, quries):
        return quries