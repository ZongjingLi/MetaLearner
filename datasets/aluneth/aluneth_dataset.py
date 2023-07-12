import torch
from torch.utils.data import Dataset, DataLoader

class AlunethKnowledge(Dataset):
    def __init__(self, config):
        super().__init__()

    def __len__(self):return 0

    def __getitem__(self, idx): return idx