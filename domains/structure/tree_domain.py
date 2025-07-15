import numpy as np
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

tree_domain_str = """
(domain Tree)
(:type
    boolean - vector[float, 1]
    tree - vector[float, 128]
    node - vector[float, 32] ;; necessary encoding for a set
    edge - vector[float, 32] ;; a number can be encoded by 8
    node_set - vector[float, 64]
    edge_set - vector[float, 64]
)
(:predicate
    nodes ?x-tree -> node_set
    edges ?y-tree -> edge_set

    edge_between ?x-node ?y-node -> boolean
    exist_path ?x-node ?y-node -> boolean
    get_child ?x-node -> node_set
    get_father ?x-node -> node
)

"""



tree_domain = load_domain_string(tree_domain_str)

class TreeExecutor(CentralExecutor):

    def nodes(self):
        return 

tree_executor = TreeExecutor(tree_domain)
