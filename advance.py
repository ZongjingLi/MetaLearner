# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
import torch.nn as nn
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string

function_domain_str = """
(domain Function)
(:type
    function - vector[float, 128] ;; abstract repr of function
    set - vector[float, 128] ;; necessary encoding for a state
    point -vector[float, 64] ;; a point in a set
)
(:predicate
    func_domain ?x-function -> set
    func_codomain ?x-function -> set
    range ?x-function -> set
    map ?x-function ?y-point -> point ;; how to represent that, why that repr is good
)

"""


function_domain = load_domain_string(function_domain_str)

class FunctionalExecutor(CentralExecutor):
    def func_domain(self, function):
        return
    
    def func_codomain(self, function):
        return 
    
    def range(self):
        return self.grounding

function_executor = FunctionalExecutor(function_domain, concept_dim = 128)


objects_domain_str = """
(domain Objects)
(:type
    objects - vector[float, 256] ;; abstract repr of function
)
(:predicate
    feature ?x-objects -> object
)

"""
objects_domain = load_domain_string(objects_domain_str)
objects_executor = CentralExecutor(objects_domain, concept_dim = 128)

path_domain_str = """
(domain Path)
(:type
    path - vector[float, 128] ;; abstract repr of path
    point -vector[float, 64]
)
(:predicate
    start ?x-path -> point
    end ?x-path -> point
    go ?x-point ?y-path -> point
)

"""
path_domain = load_domain_string(path_domain_str)
path_executor = CentralExecutor(path_domain, concept_dim = 128)

set_domain_str = """
(domain Set)
(:type
    set - vector[float, 128]
    point - vector[float,64]
)
(:predicate 
    union ?x-set ?y-set -> set
    intersect ?x-set ?y-set -> set
    compl ?x-set -> set
    subset ?x-set ?y-set -> boolean
    in ?x-point ?y-set -> bool
)
"""
set_domain = load_domain_string(set_domain_str)
set_executor = CentralExecutor(set_domain, concept_dim = 128)

group_domain_str = """
(domain Group)
(:type
    group - vector[float, 72]
    set - vector[float, 128]
    point - vector[float,64]
)
(:predicate 
    mul ?x-point ?y-point -> point
    inv ?x-point ?y-point -> point
    id -> point
)
"""
group_domain = load_domain_string(group_domain_str)
group_executor = CentralExecutor(group_domain, concept_dim = 128)

complex_domain_str = """
(domain Complex)
(:type
    set - vector[float, 128]
    complex - vector[float, 5]
    real - vector[float, 1]
)
(:predicate
    real ?x-complex -> num
    im   ?x-complex -> num
    mul ?x-complex ?y-complex -> complex
    add ?x-complex ?y-complex -> complex
)
"""


domains = [
    function_executor, objects_executor, path_executor
]

from rinarak import stprint
from core.metaphors.diagram_executor import MetaphorExecutor


executor = MetaphorExecutor(domains, concept_dim = 128)


executor.add_caster("point:Path", "set:Function") # a point as a function


point_type = torch.randn([4,64])

val, p = executor.cast_type(point_type, "point:Path", "set:Function")

grounding = {"Grounding": "HaHa"}


from rinarak.knowledge.symbolic import Expression
expr = Expression.parse_program_string("go:Path(1)")

print(expr)

executor.chainer.add_edge("go:Path", "map:Function")
executor.evaluate(expr, grounding)


"""
import matplotlib.pyplot as plt
result = executor.chainer.find_most_far_reaching_nodes("go:Path")
executor.chainer.visualize_graph("go:Path", result)
plt.show()"
"""