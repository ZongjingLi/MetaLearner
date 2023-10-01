import torch
import torch.nn as nn

from queue import PriorityQueue

class FactorState:
    def __init__(self, entities):
        self.entities = entities

class NeuroPredicate(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = []
        self.system = []
        self.pre_condition = None
        self.effect = None
    
    def applicable(self, args):return self.pre_condition(args)

    def get_effect(self, args):return self.pre_condition(args)

class NeuroReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self,x):
        """
        inputs: init_state(abs), goal_eval()
        outputs: several possible plans
        """
        return x