'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-03-02 13:54:56
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-03-02 13:55:00
 # @ Description: This file is distributed under the MIT license.
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import F
from torchvision import transforms

import kornia
from einops import rearrange

class Patchify(nn.Module):
    def __init__(self, patch_size = (16,16), temporal_dim = 1, squeeze_channel_dim = True):
        super().__init__()

class LinearPatchEmbed(Patchify):
    """Embed patches of shape (pt, ph, pw) using linear model"""
    def __init__(self, input_dim = 3,
                output_dim = None,
                patch_size = (1, 8, 8),
                temporal_dim = 1):
        super().__init__(
            patch_size = patch_size,
            temporal_dim = temporal_dim,
            squeeze_channel_dim = True,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim or (np.prod(patch_size) * self.input_dim)
        self.patch_size = patch_size
        self.embed = nn.Linear(self.input_dim * np.prod(patch_size), output_dim)
    
    def forward(self, x, split_time = False, **kwargs):
        x = super().forward(x, **kwargs)
        x = self.embed(x)
        if split_time: x = x.split_by_time(x)
        return x