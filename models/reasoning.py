from queue import PriorityQueue
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import *
from .executor import *
import networkx as nx

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
        self.entailment = build_entailment(config)

        # [Neuro Symbolic Executor]
        self.executor = SceneProgramExecutor(config)
        self.rep = config.concept_type
        self.applier = None
        self.p = .97

    def forward(self, quries):
        return quries
    
    def search(self, init_state, init_latent, goal):
        """
        args:
            init_state: 
            init_latent:
            goal: the goal statement of search. The search ends when the goal statement is evaluated as true.
        outputs:
            the search path and the corresponding confidence
        """
        open_states = PriorityQueue()
        close_states = set()

        for possible_state in self.applier.expand(init_state):
            if self.applier.evaluate(possible_state, goal) >= self.p:
                print("solved")
            pass

        return 0