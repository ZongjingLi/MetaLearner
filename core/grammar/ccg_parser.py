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
        # The weight is the sum of the two entries' weights (maintaining the gradient graph)
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
        
        # The weight is the sum of the two entries' weights (maintaining the gradient graph)
        combined_weight = left_entry.weight + right_entry.weight
        
        return LexiconEntry(f"{left_entry.word} {right_entry.word}", result_type, new_program, combined_weight)
    
class ChartParser(nn.Module):
    """
    Implementation of G2L2 parser with CKY-E2 algorithm
    Modified to support PyTorch gradient computation with centralized weight management
    """
    def __init__(self, lexicon: Dict[str, List], rules: List = None):
        super().__init__()
        
        # Create ModuleDict for lexicon structure (without weights)
        module_dict = {}
        # Create ParameterDict for lexicon weights
        weight_dict = {}
        
        for word_key, entries in lexicon.items():
            # Store entries without their weights
            module_list = []
            for idx, entry in enumerate(entries):
                # Create unique parameter key for each word-type combination
                weight_key = f"{word_key}_{idx}"
                # Store the initial weight value
                if isinstance(entry.weight, nn.Parameter):
                    weight_dict[weight_key] = entry.weight
                else:
                    # Convert to parameter if it's not already
                    weight_dict[weight_key] = nn.Parameter(entry.weight)
                
                # Add entry to module list (it will be used as a template)
                module_list.append(entry)
            
            module_dict[word_key] = (module_list)
        
        self.lexicon = module_dict#nn.ModuleDict(module_dict)
        self.lexicon_weight = nn.ParameterDict(weight_dict)
        self.rules = [ForwardApplication, BackwardApplication] if rules is None else rules

    def save_weights(self, save_path):
        """
        Save the lexicon weights to a file
        
        Args:
            save_path: Path to save the weights
        """
        weight_dict = {}
        for key, param in self.lexicon_weight.items():
            weight_dict[key] = param.data.clone()
        
        torch.save(weight_dict, save_path)
        
    def load_weights(self, load_path):
        """
        Load lexicon weights from a file
        
        Args:
            load_path: Path to load the weights from
        """
        weight_dict = torch.load(load_path)
        
        # Create new ParameterDict
        new_weight_dict = nn.ParameterDict()
        
        for key, value in weight_dict.items():
            new_weight_dict[key] = nn.Parameter(value)
        
        self.lexicon_weight = new_weight_dict
        
        # Update the lexicon entries with the loaded weights
        for word, entries in self.lexicon.items():
            for idx, entry in enumerate(entries):
                weight_key = f"{word}_{idx}"
                if weight_key in self.lexicon_weight:
                    entry.weight = self.lexicon_weight[weight_key]

    def get_entry_weight(self, word: str, idx: int) -> torch.Tensor:
        """Get weight for a lexicon entry from the centralized parameter dictionary"""
        weight_key = f"{word}_{idx}"
        return self.lexicon_weight[weight_key]
    
    def get_likely_entries(self, word: str, K: int = 3) -> List:
        """Get top K lexicon entries for a word based on their weights"""
        if word not in self.lexicon: 
            return []
        
        # Get entries and their weights
        entries = self.lexicon[word]
        weights = [self.get_entry_weight(word, idx) for idx in range(len(entries))]
        
        # Sort by weights (descending)
        entry_weights = list(zip(entries, weights))
        sorted_entries = sorted(entry_weights, key=lambda x: x[1], reverse=True)[:K]
        
        # Create new LexiconEntry objects with proper weights
        result = []
        for entry, weight in sorted_entries:
            # Create a new entry with the weight from our parameter dict
            new_entry = LexiconEntry(
                entry.word,
                entry.syn_type,
                entry.sem_program,
                weight  # This is an nn.Parameter
            )
            result.append(new_entry)
        
        return result

    def parse(self, sentence: str, topK = None, forced = False):
        words = sentence.split()
        n = len(words)
        chart = {}
        
        # Fill in the chart with lexicon entries for each word
        for i in range(n):
            word = words[i]
            if word in self.lexicon:
                if topK is None:
                    # Get all entries with their weights from parameter dict
                    entries = []
                    for idx, entry in enumerate(self.lexicon[word]):
                        weight = self.get_entry_weight(word, idx)
                        new_entry = LexiconEntry(
                            entry.word,
                            entry.syn_type,
                            entry.sem_program,
                            weight  # This is an nn.Parameter
                        )
                        entries.append(new_entry)
                    chart[(i, i+1)] = entries
                else:
                    chart[(i, i+1)] = self.get_likely_entries(word, topK)
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
                                        chart[(start, end)].append(result)
        
        return chart[(0, n)]
    
    def get_parse_probability(self, parses):
        """Calculate probability for each parse"""
        if not parses: 
            return torch.tensor([], requires_grad=True)
        #print(list(self.parameters()))
        # Get weights and apply softmax
        weights = torch.stack([parse.weight for parse in parses])
        
        # Use softmax to get log probabilities
        log_probs = F.log_softmax(weights, dim=0)

        
        return log_probs
    
    def compute_loss(self, parses, target_idx=0):
        """Compute negative log likelihood loss for training"""
        if not parses:
            return torch.tensor(0.0, requires_grad=True)
            
        log_probs = self.get_parse_probability(parses)
        
        # Use negative log likelihood of the target parse as loss
        if target_idx < len(log_probs):
            loss = -log_probs[target_idx]
        else:
            # If target_idx is out of bounds, use the most likely parse
            loss = -log_probs[0]
            
        return loss