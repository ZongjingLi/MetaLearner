import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ManariKnowledge(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def __len__(self):return 0

    def __getitem__(self, idx):return idx

class ManariSearch(Dataset):
    def __init__(self, config):
        super().__init__(self, config)
        self.config = config 

    def __len__(self):return 0

    def __getitem__(self, idx): return idx