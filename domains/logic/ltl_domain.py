import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

ltl_domain_str = """
(domain LTL)
(:type
    boolean - vector[float, 1]
    integer
    timepoint
)
(:predicate
    next ?x-boolean -> boolean
    until ?x-boolean ?y-boolean -> boolean
    eventually ?x-boolean -> boolean
    always ?x-boolean -> boolean
    release ?x-boolean ?y-boolean -> boolean
    weak_until ?x-boolean ?y-boolean -> boolean
    negate ?x-boolean -> boolean
    and ?x-boolean ?y-boolean -> boolean
    or ?x-boolean ?y-boolean -> boolean
    imply ?x-boolean ?y-boolean -> boolean
)
"""

ltl_domain = load_domain_string(ltl_domain_str)

class LTLExecutor(CentralExecutor):
    def next(self, logits):
        # Shift logits to represent the next state
        # Assuming logits has a time dimension
        return torch.roll(logits, shifts=-1, dims=0)
    
    def until(self, logits1, logits2):
        # φ until ψ: ψ holds at the current or future position, and φ holds until that position
        result = torch.zeros_like(logits1)
        T = logits1.size(0)
        
        for i in range(T):
            # Check if ψ holds at current position
            result[i] = logits2[i]
            
            # For each future position j, check if φ holds until j and ψ holds at j
            for j in range(i+1, T):
                temp = logits2[j]
                for k in range(i, j):
                    temp = torch.min(temp, logits1[k])
                result[i] = torch.max(result[i], temp)
        
        return result
    
    def eventually(self, logits):
        # Eventually φ: φ holds at the current or some future position
        T = logits.size(0)
        result = torch.zeros_like(logits)
        
        for i in range(T):
            result[i] = torch.max(logits[i:])
        
        return result
    
    def always(self, logits):
        # Always φ: φ holds at the current and all future positions
        T = logits.size(0)
        result = torch.zeros_like(logits)
        
        for i in range(T):
            result[i] = torch.min(logits[i:])
        
        return result
    
    def release(self, logits1, logits2):
        # φ releases ψ: ψ holds until and including the point where φ holds (or forever if φ never holds)
        T = logits1.size(0)
        result = torch.zeros_like(logits1)
        
        for i in range(T):
            hold_forever = True
            for j in range(i, T):
                if j == T-1 and hold_forever:
                    result[i] = torch.min(result[i], logits2[j])
                else:
                    release_at_j = torch.min(logits1[j], logits2[j])
                    temp = release_at_j
                    for k in range(i, j):
                        temp = torch.max(temp, logits2[k])
                    result[i] = torch.max(result[i], temp)
        
        return result
    
    def weak_until(self, logits1, logits2):
        # Weak until: either "until" holds, or "always φ" holds
        until_result = self.until(logits1, logits2)
        always_result = self.always(logits1)
        return torch.max(until_result, always_result)
    
    def negate(self, logit):
        return -logit
    
    def andl(self, logit1, logit2):
        return torch.min(logit1, logit2)
    
    def orl(self, logit1, logit2):
        return torch.max(logit1, logit2)
    
    def imply(self, logit1, logit2):
        return torch.max(-logit1, logit2)

ltl_executor = LTLExecutor(ltl_domain)