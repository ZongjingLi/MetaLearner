import math
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lexicon import CCGSyntacticType, SemProgram, LexiconEntry

class CCGRule:
    """Abstract base class for CCG combinatory rules"""
    @staticmethod
    def can_apply(left_type, right_type):
        """Check if the rule can apply to the given types"""
        return False
    
    @staticmethod
    def apply(left_entry, right_entry):
        """Apply the rule to combine two lexicon entries"""
        return None

class ForwardApplication(CCGRule):
    """Forward application rule: X/Y Y => X"""
    @staticmethod
    def can_apply(left_type, right_type):
        return (not left_type.is_primitive and 
                left_type.direction == '/' and 
                left_type.arg_type == right_type)
    
    @staticmethod
    def can_forced_apply(left_type, right_type):
        return (not left_type.is_primitive and
                left_type.direction == "/" and
                1 # this means the type does not need to match
                )
    
    @staticmethod
    def apply(left_entry, right_entry):
        # Create a new lexicon entry with the result type and computed program
        result_type = left_entry.syn_type.result_type
        
        # For lambda functions, apply the argument to the function
        if left_entry.sem_program.lambda_vars:
            # This is a simplified application - in reality would be more complex
            new_args = left_entry.sem_program.args.copy()
            new_args.append(right_entry.sem_program)
            new_program = SemProgram(
                left_entry.sem_program.func_name,
                new_args,
                left_entry.sem_program.lambda_vars[1:] if len(left_entry.sem_program.lambda_vars) > 1 else []
            )
        else:
            # Simple function application
            new_args = left_entry.sem_program.args.copy()
            new_args.append(right_entry.sem_program)
            new_program = SemProgram(left_entry.sem_program.func_name, new_args)
        
        # Create a new lexicon entry with combined weight
        # Use direct addition to maintain gradient graph
        combined_weight = left_entry.weight + right_entry.weight
        
        return LexiconEntry(f"{left_entry.word} {right_entry.word}", result_type, new_program, combined_weight)

class BackwardApplication(CCGRule):
    """Backward application rule: Y X\Y => X"""
    @staticmethod
    def can_apply(left_type, right_type):
        return (not right_type.is_primitive and 
                right_type.direction == '\\' and 
                right_type.arg_type == left_type)

    @staticmethod
    def can_forced_apply(left_type, right_type):
        return (not right_type.is_primitive and 
                right_type.direction == '\\')

    @staticmethod
    def apply(left_entry, right_entry):
        # Similar to forward application but with different direction
        result_type = right_entry.syn_type.result_type
        
        if right_entry.sem_program.lambda_vars:
            new_args = right_entry.sem_program.args.copy()
            new_args.append(left_entry.sem_program)
            new_program = SemProgram(
                right_entry.sem_program.func_name,
                new_args,
                right_entry.sem_program.lambda_vars[1:] if len(right_entry.sem_program.lambda_vars) > 1 else []
            )
        else:
            new_args = right_entry.sem_program.args.copy()
            new_args.append(left_entry.sem_program)
            new_program = SemProgram(right_entry.sem_program.func_name, new_args)
        
        # Use PyTorch addition to maintain gradient graph
        combined_weight = left_entry.weight + right_entry.weight
        return LexiconEntry("", result_type, new_program, combined_weight)

class ChartParser(nn.Module):
    """
    Implementation of G2L2 parser with CKY-E2 algorithm
    Modified to support PyTorch gradient computation
    """
    def __init__(self, lexicon: nn.ModuleDict, rules: List[CCGRule] = None):
        super().__init__()
        
        module_dict = {}
        for word_key, entries in lexicon.items():
            assert all(isinstance(e, nn.Module) for e in entries)
            module_dict[word_key] = nn.ModuleList(entries)  # Register entries properly
        self.lexicon = nn.ModuleDict(module_dict)  # Register dict properly

        self.rules = [ForwardApplication, BackwardApplication] if rules is None else rules
    
    def get_likely_entries(self, word: str, K: int = 3) -> List[LexiconEntry]:
        """given a word we find the top K"""
        if word not in self.lexicon: return []
        entries = sorted(self.lexicon[word], key=lambda e: e.weight, reverse=True)[:K]
        return entries

    def parse(self, sentence: str, topK = None, forced = False):
        words = sentence.split()
        n = len(words)
        chart = {}
        for i in range(n):
            word = words[i]
            if word in self.lexicon:
                chart[(i, i+1)] = self.lexicon[word] if topK is None else self.get_likely_entries(word, topK)
            else:
                chart[(i, i+1)] = []
                print(f"Warning: Word '{word}' not in lexicon")
    
        # Fill chart with dynamic programming
        for length in range(2, n+1):
            for start in range(n - length + 1):
                end = start + length
                chart[(start, end)] = []
                
                for split in range(start+1, end):
                    left_entries = chart[(start, split)]
                    right_entries = chart[(split, end)]
                    
                    for left_entry in left_entries:
                        for right_entry in right_entries:
                            for rule in self.rules:
                                if forced:
                                    applicable = rule.can_forced_apply(left_entry.syn_type, right_entry.syn_type)
                                else:
                                    applicable = rule.can_apply(left_entry.syn_type, right_entry.syn_type)
                                if applicable:
                                    result = rule.apply(left_entry, right_entry)
                                    if result:
                                        # Register the new entry to track parameters
                                        chart[(start, end)].append(result)
        
        return chart[(0, n)]
    
    def get_parse_probability(self, parses):
        """Calculate probability for each parse"""
        if not parses: return torch.tensor([], requires_grad=True)
        
        # Get weights and apply softmax
        #print([parse.weight.requires_grad for parse in parses])
        weights = torch.stack([parse.weight for parse in parses])

        log_probs = F.log_softmax(weights, dim=0)

        optim = torch.optim.Adam(self.parameters(), lr = 1e-2)
        

        loss = torch.sum(log_probs)    
        optim.zero_grad()
        loss.backward()
        
        params_with_grad = [
             (name, param) for name, param in self.named_parameters()
                if param.grad is not None and param.grad.abs().sum() > 0
            ]

        for name, param in params_with_grad:
            print(name)

        optim.step()
        

        return log_probs