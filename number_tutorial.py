# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

dim = 128

numbers_domain_str = """
(domain Numbers)
(:type
    set - vector[float, 128] ;; necessary encoding for a set
    number -vector[float, 3] ;; a number can be encoded by 8 dims (overwhelm)
)
(:predicate
    one -> number
    two -> number
    three -> number
    R -> set
    Z -> Set
    plus ?x-number ?y-number -> number
)

"""

numbers_domain = load_domain_string(numbers_domain_str)

class NumbersExecutor(CentralExecutor):
    one_embed = nn.Parameter(torch.randn([3]))
    two_embed = nn.Parameter(torch.randn([3]))
    three_embed = nn.Parameter(torch.randn([3]))

    R_embed = nn.Parameter(torch.randn([128]))
    Z_embed = nn.Parameter(torch.randn([128]))

    def one(self): return self.one_embed
    
    def two(self): return self.two_embed

    def three(self): return self.three_embed
    
    def R(self): return self.R_embed

    def Z(self): return self.Z_embed

    def plus(self, x, y): return x + y

numbers_executor = NumbersExecutor(numbers_domain, concept_dim = dim)


objects_domain_str = """
(domain Objects)
(:type
    ObjSet - vector[float, 128] ;; necessary encoding for a set
    Obj - vector[float, 128]
)
(:predicate
    scene -> ObjSet
)
"""

objects_domain = load_domain_string(objects_domain_str)

class ObjectsExecutor(CentralExecutor):
    def scene(self):
        return self._grounding["objects"]

objects_executor = ObjectsExecutor(objects_domain, concept_dim = dim)



domains = [
    numbers_executor, objects_executor
]


from helchriss.knowledge.symbolic import Expression
from core.metaphors.diagram_executor import MetaphorExecutor


executor = MetaphorExecutor(domains, concept_dim = 128)

"""create a demo scene for the executor to execute on."""
num_objs = 4
grounding = {"objects": torch.randn([num_objs, 128]), "ref" : torch.randn([num_objs,1])}

expr = "scene:Objects()"
result = executor.evaluate(expr, grounding)


expr = "plus:Numbers(one:Numbers(), two:Numbers())"
result = executor.evaluate(expr, grounding)

print(result)

for tp in executor.types: print(tp["name"], tp["type_space"])

for f in executor.functions:print(f)