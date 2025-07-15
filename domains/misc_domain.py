import numpy as np
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

graph_domain_str = """
(domain Misc)
(:type
    boolean - vector[float, 1]
)
(:predicate
    Id ?x-boolean -> boolean
)

"""

graph_domain = load_domain_string(graph_domain_str)

class MiscExecutor(CentralExecutor):
    def Id(self, x): return x

misc_executor = MiscExecutor(graph_domain)
