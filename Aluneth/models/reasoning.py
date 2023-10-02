import torch
import torch.nn as nn

class Entity:
    def __init__(self):
        self.latex_name = "A"

class FactorState:
    def __init__(self, entities):
        self.entities = entities

class NeuroAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = []
        self.system = []
        self.pre_condition = None
        self.effect = None
        
        self.sampler = None
    
    def applicable(self, args):
        return self.pre_condition(args)

    def get_effect(self, args):
        return self.effect(args)

class NeuroReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self,x):
        """
        inputs: init_state(abs), goal_eval()
        outputs: several possible plans
        """
        return x
    
    def reason(self,x, max_steps = 1132):
        init_state = FactorState([])
        reason_stop = 1
        itrs = 0

        visited_states = []
        curr_state = init_state
        while not reason_stop and itrs <= max_steps:
            itrs += 1
            for next_states in self.get_next_state(curr_state):
                pass
        return x
    
    def get_next_state(self, state):
        return []

if __name__ == "__main__":

    
    nr = NeuroReasoner(None)