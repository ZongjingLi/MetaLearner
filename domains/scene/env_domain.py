
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

objects_domain_str = """
(domain Env)
(:type
    position - vector[float, 3]
    orientation - vector[float, 3]
    pose - vector[float, 6]
    trajectory - vector[10, 6]
    observation - vector[float,128,128]
)
(:predicate
    pickup ?x-observation ?y-pose ?z-pose -> observation 
)
"""

objects_domain = load_domain_string(objects_domain_str)

objects_domain.print_summary()

class ObjectsExecutor(CentralExecutor):
    
    def scene(self):
        features = self._grounding["objects"] # [nxd] features
        scores = self._grounding["scores"] # [nx1] scores as logits
        return torch.cat([features, scores], dim = -1)

    def unique(self, objset):
        features = objset[:,:-2] # [nxd] features
        scores = torch.logit(torch.softmax(objset[:,-1:])) # [nx1] scores normalized
        return torch.cat([features, scores], dim = -1)

objects_executor = ObjectsExecutor(objects_domain)