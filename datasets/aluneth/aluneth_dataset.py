import torch
from torch.utils.data import Dataset, DataLoader
import json

root_dir = "/Users/melkor/Documents/datasets"

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
        return data

class AlunethKnowledge(Dataset):
    def __init__(self, config):
        super().__init__()
        self.statements = load_json(root_dir + "/Aluneth/Knowledge/u1_knowledge.json")

    def __len__(self):return len(self.statements)

    def __getitem__(self, idx): return self.statements[idx]
    
class AlunethSearch(Dataset):
    def __init__(self, config):
        super().__init__()

    def __len__(self):return 0

    def __getitem__(self, idx):return 