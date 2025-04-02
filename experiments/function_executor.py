import torch
import torch.nn as nn
import torch.nn.functional as F
from rinarak.knowledge.executor import CentralExecutor
from rinarak.utils.data import ListDataset
from rinarak.utils import stprint
from dataclasses import dataclass
from typing import Tuple, List

from rinarak.domain import load_domain_string

domain_str = """
(domain World)
(:type
    ObjSet - vector[float, 3]
    position - vector[float, 3]
)
(:predicate
    scene -> boolean
    red ?x-ObjSet -> boolean
)
"""

domain = load_domain_string(domain_str)

@dataclass
class Item:
    color : str
    size : Tuple[float]
    position : Tuple[float]

@dataclass
class Scene:
    objects : List[Item]

class CustomExecutor(CentralExecutor):
    def scene(self):
        return self.grounding.objects
    
    def red(self, scene):
        red_items = []
        for item in scene:
            if item.color == "red":
                red_items.append(item)
        
        return red_items

executor = CustomExecutor(domain)

expr = "red(scene())"
grounding = Scene([
    Item("red", (1., 1., 1.), [0.1, 0.0, 0.3]),
    Item("blue",(2., 1., 2.), [-0.1, 0.2, 0.0])
    ])

res = executor.evaluate(expr, grounding)

stprint(res.value)

