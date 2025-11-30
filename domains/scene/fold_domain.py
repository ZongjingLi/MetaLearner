
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

objects_domain_str = """
(domain Objects)
(:type
    Obj - vector[float, 128] ;; a normalized distribution over a list of objects
    ObjSet - List[vector[float, 128]] ;; unnormalized distribution over a list of objects
)
(:predicate
    scene -> ObjSet
    ;;do not use it now to_object ?x-ObjSet ?y-List[boolean] -> ?x-Object
)
"""

objects_domain = load_domain_string(objects_domain_str)

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