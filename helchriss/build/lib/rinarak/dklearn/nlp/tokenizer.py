import torch
import torch.nn as nn

class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()

class Embedding(nn.Module):
    def __init__(self, max_embeddings, feature_dim, keys = None):
        self.keys = None
        self.embeddings = nn.Embedding(max_embeddings, feature_dim)
    
    def forward(self, idx, get_keys = False):
        keys = None
        if get_keys:
            assert self.keys is not None,print("This Embedding Module Do Not Have Keys")
            keys = [self.keys[idx_] for idx_ in idx]
        features = self.embeddings[idx]
        return {"keys":keys,"features":features}