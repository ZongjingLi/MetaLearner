
import torch
from torch.utils.data import Dataset, DataLoader

class GroundDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return []

    def __getitem__(self, idx):
        """ each item will be an a diction that contains, input, modal, ground truth, effective mask
        Returns:
            input: anything that an perception module can take as inoput
            modal: a object centric detection module that take
        """
        return idx