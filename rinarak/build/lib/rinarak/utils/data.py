'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-24 18:42:10
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-24 18:42:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_img(img):
    if len(img.shape) == 4:
        if not img.shape[1] in [1,3,4]: return img.permute(0,3,1,2)
    if len(img.shape) == 3:
        if not img.shape[0] in [1,3,4]: return img.permute(2,0,1)

