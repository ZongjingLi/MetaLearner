import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

integer_domain_str = """
(domain :: Integer)
(def type
    set - Embedding[set, 32] ;; necessary encoding for a set
    num - float ;; a number can be encoded by 8 dims (overwhelm)
)
(def function
    one : num := by pass
    two : num := by pass
    three : num := by pass

    plus (x y : num) : num := by pass
    minus (x y : num) : num := by pass
    mul (x y : num) : num := by pass

    avg (x : List[num]) : num := by pass

    smaller  (x y : num) : boolean := by pass
    bigger (x y : num) : boolean := by pass

    N : set := by pass
    Z : set := by pass

    max (x : List[num]) : num := by pass

    subset (x y : set) : boolean := by pass
)

"""

integer_domain = load_domain_string(integer_domain_str)

class NumbersExecutor(CentralExecutor):

    N_embed = nn.Parameter(torch.randn([32]))
    Z_embed = nn.Parameter(torch.randn([32]))
    temperature = 0.132

    def max(self, x): return torch.max(x)

    def one(self): return torch.tensor(1.0)

    def two(self): return torch.tensor(2.0)

    def three(self): return torch.tensor(3.0)
    
    def N(self): return self.N_embed

    def Z(self): return self.Z_embed

    def plus(self, x, y): return x + y

    def mul(self, x, y): return x * y


    def minus(self, x, y): return x - y

    def smaller(self, x, y):
        return (y - x) / self.temperature
    
    def bigger(self, x, y):
        return (x - y) / self.temperature
    
    def subset(self, set1, set2): return torch.norm( set1 - set2)

    def avg(self, x): 
        average = 0.0
        for v in x: average += v
        return average / len(x)

integer_executor = NumbersExecutor(integer_domain)
