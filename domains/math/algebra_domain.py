import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

integer_domain_str = """
(domain Integers)
(:type
    set - vector[float, 32] ;; necessary encoding for a set
    int -vector[float, 1] ;; a number can be encoded by 8 dims (overwhelm)
)
(:predicate

)
"""

integers_domain = load_domain_string(integer_domain_str)

class AlgebraExecutor(CentralExecutor):

    N_embed = nn.Parameter(torch.randn([32]))
    Z_embed = nn.Parameter(torch.randn([32]))
    temperature = 0.132


integers_executor = AlgebraExecutor(integers_domain)
