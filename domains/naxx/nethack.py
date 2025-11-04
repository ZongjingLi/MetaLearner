import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

nethack_domain_str = """
(domain :: Nethack)
(def type
    obj - Embedding[hack_obj, 32] ;; encoding of an nethack object
    region - Embedding[region_mask, 64] ;; encoding of a region mask
    coord - Embedding[coord2d, 2] ;; the 2-dim coord
)
(def function
    player : obj := by pass
    location (x : obj): coord := by pass
    mobs : (r : region) : List[obj] := by pass
)

"""

nethack_domain = load_domain_string(nethack_domain_str)

class NethackExecutor(CentralExecutor):

    N_embed = nn.Parameter(torch.randn([32]))
    Z_embed = nn.Parameter(torch.randn([32]))
    temperature = 0.132


    def one(self): return torch.tensor(1.0)

    def two(self): return torch.tensor(2.0)

    def three(self): return torch.tensor(3.0)
    
    def N(self): return self.N_embed

    def Z(self): return self.Z_embed

    def plus(self, x, y): return x + y


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

nethack_executor = NethackExecutor(nethack_domain)
