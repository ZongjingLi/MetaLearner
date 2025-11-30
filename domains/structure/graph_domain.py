import numpy as np
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

graph_domain_str = """
(domain Graph)
(:type
    boolean - vector[float, 1]
    graph - vector[float, 128]
    node - vector[float, 32] ;; necessary encoding for a set
    edge - vector[float, 32] ;; a number can be encoded by 8
    node_set - vector[float, 64]
    edge_set - vector[float, 64]
)
(:predicate
    nodes ?x-graph -> node_set
    edges ?y-graph -> edge_set
    connected ?x-graph -> boolean
    edge_between ?x-node ?y-node -> boolean
    exist_path ?x-node ?y-node -> boolean
)

"""

graph_domain = load_domain_string(graph_domain_str)

class GraphExecutor(CentralExecutor):

    def nodes(self):
        return 

graph_executor = GraphExecutor(graph_domain)
