import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

# Set Domain Definition
equalizer_domain_str = """
(domain Equalizer)
(def type
    object - Embedding[equalizer, 64]
)
(:predicate
    make_set<obj> : ?x-List[obj] ?y- comparator<obj> ->  List[obj]
)
"""

equalizer_domain = load_domain_string(equalizer_domain_str)

class EqualizerExecutor(CentralExecutor):
    empty_embed = nn.Parameter(torch.randn([32]))
    universal_embed = nn.Parameter(torch.randn([32]))
    temperature = 0.1
    
    def empty(self):
        return self.empty_embed
    
    def universal(self):
        return self.universal_embed
    
    def contains(self, s, e):
        # Compute similarity between set and element
        similarity = torch.sum(s[:16] * e) / self.temperature
        return torch.sigmoid(similarity)
    
    def subset(self, s1, s2):
        # Set s1 is subset of s2 if their embeddings align in a certain way
        diff = s2 - s1
        score = torch.mean(torch.relu(diff)) / self.temperature
        return torch.sigmoid(score)
    
    def union(self, s1, s2):
        # Element-wise maximum as a simple approximation for union
        return torch.max(s1, s2)
    
    def intersection(self, s1, s2):
        # Element-wise minimum as a simple approximation for intersection
        return torch.min(s1, s2)
    
    def difference(self, s1, s2):
        # Approximate set difference
        return s1 * (1 - torch.sigmoid((s2 - s1) / self.temperature))
    
    def cardinality(self, s):
        # Approximate cardinality from set embedding
        energy = torch.sum(torch.abs(s))
        return torch.tensor([torch.log(1 + energy)])

equalizer_executor = EqualizerExecutor(equalizer_domain)
