import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import *
from .executor import *

class VanillaReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, quries):
        return quries

class MetaReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)

        # [Neuro Symbolic Executor]
        self.executor = SceneProgramExecutor(config)
        self.rep = config.concept_type
    def forward(self, quries):
        return quries