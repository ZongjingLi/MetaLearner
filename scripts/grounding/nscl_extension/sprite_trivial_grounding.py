# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
import torch.nn as nn

"""create some test domains """
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string, Domain
from domains.utils import domain_parser

function_domain_str = """
(domain Function)
(:type
    function - vector[float, 128] ;; abstract repr of function
    set - vector[float, 128] ;; necessary encoding for a state
    point -vector[float, 64] ;; a point in a set
)
(:predicate
    domain ?x-function -> set
    codomain ?x-function -> set
    range ?x-function -> set
    map ?x-function ?y-point -> point ;; how to represent that, why that repr is good
)

"""
function_domain = load_domain_string(function_domain_str, domain_parser)
function_executor = CentralExecutor(function_domain, "cone", 256)


objects_domain_str = """
(domain Objects)
(:type
    objects - vector[float, 256] ;; abstract repr of function
)
(:predicate
    feature ?x-objects -> object
)

"""
objects_domain = load_domain_string(objects_domain_str, domain_parser)
objects_executor = CentralExecutor(objects_domain, "cone", 256)

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
path_domain = load_domain_string(path_domain_str, domain_parser)
path_executor = CentralExecutor(path_domain, "cone", 256)

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
set_domain = load_domain_string(set_domain_str, domain_parser)
set_executor = CentralExecutor(set_domain, "cone", 256)

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
group_domain = load_domain_string(group_domain_str, domain_parser)
group_executor = CentralExecutor(group_domain, "cone", 256)

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

from domains.rcc8.rcc8_domain import rcc8_executor
from domains.logic.logic_domain import build_logic_executor
logic_executor = build_logic_executor()

from core.metaphors.base import *
from core.metaphors.diagram_legacy import *
from core.metaphors.diagram import *

plt.show()
diagram = ConceptDiagram()

"""1. Create the test Concept Diagram"""

nodes = {
	"Objects" : objects_executor,
	"Function" : function_executor,
	"RCC8" : rcc8_executor,
    "Set" : set_executor,
    "Logic" : logic_executor
}
edges = [
	("Objects", "Function"),
	("Objects", "Set"),
    ("Objects", "Set"),
    ("Objects", "RCC8"),
	("Function", "Set"),
    ("Set", "RCC8"),
    ("RCC8", "Set"),
    ("Objects", "Logic")
]


for domain in nodes: diagram.add_domain(domain, nodes[domain])
for morph in edges: diagram.add_morphism(morph[0], morph[1], MetaphorMorphism(nodes[morph[0]], nodes[morph[1]]))

diagram.visualize("outputs/a-diagram")

device = "mps"
diagram.to(device)


"""2. Evaluate an input for the Concept Diagram (Expectation)"""

scores = torch.tensor([1.0, 0.1, 1.0, ], device = device)

args = [
    {"end" : torch.randn([3,256], device = device), "type":"objects", "domain" : "Objects","score" : scores},
    {"end" : torch.randn([3,256], device = device), "type":"objects", "domain" : "Objects","score" : torch.ones([3], device = device)}
]

sample_path, sample_prob, results = diagram.sample_state_path(
args, "Objects", "Set", "set")


path_dist, masks, states = diagram.gather(results)


paths = results["paths"]
weights = results["path_scores"]


masks = results["path_masks"][0]


dot_code = diagram.visualize_paths(paths, weights, masks)
from graphviz import Source
graph = Source(dot_code)
graph.render('outputs/paths_visualization', format='png', cleanup=True)


#tree, edges, node_map = diagram.tree_expansion("Objects")
#diagram.visualize_tree("Objects")


results = diagram.evaluate_predicate("subset", args)

