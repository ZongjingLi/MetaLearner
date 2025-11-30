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
        # Use PyTorch addition to maintain gradient graph
        combined_weight = left_entry.weight + right_entry.weight
        return LexiconEntry("", result_type, new_program, combined_weight)

class BackwardApplication(CCGRule):
    """Backward application rule: Y X\Y => X"""
    @staticmethod
    def can_apply(left_type, right_type):
        return (not right_type.is_primitive and 
                right_type.direction == '\\' and 
                right_type.arg_type == left_type)
    
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



class G2L2Parser(nn.Module):
    """
    Implementation of G2L2 parser with CKY-E2 algorithm
    Modified to support PyTorch gradient computation
    """
    def __init__(self, lexicon: Dict[str, List[LexiconEntry]], rules: List[CCGRule]):
        super(G2L2Parser, self).__init__()
        self.lexicon = lexicon
        self.rules = rules
        
        # Initialize trainable parameters
        self.word_weights = nn.ParameterDict()
        for word, entries in lexicon.items():
            for i, entry in enumerate(entries):
                param_name = f"{word}_{i}"
                # Create parameter and replace the entry's weight
                param = nn.Parameter(entry.weight.clone())
                self.word_weights[param_name] = param
                entry._weight = param

    def parse(self, sentence: str):
        """Parse a sentence using the CKY-E2 algorithm"""
        words = sentence.split()
        n = len(words)
        
        # Initialize the parse chart
        chart = {}
        for i in range(n):
            word = words[i]
            if word in self.lexicon:
                chart[(i, i+1)] = self.lexicon[word]
            else:
                chart[(i, i+1)] = []
                print(f"Warning: Word '{word}' not in lexicon")
    
        # Build the parse chart using CKY
        for length in range(2, n+1):
            for start in range(n - length + 1):
                end = start + length
                chart[(start, end)] = []
                
                # Try all split points
                for split in range(start+1, end):
                    left_entries = chart[(start, split)]
                    right_entries = chart[(split, end)]
                    
                    # Try to combine using each rule
                    for left_entry in left_entries:
                        for right_entry in right_entries:
                            for rule in self.rules:
                                if rule.can_apply(left_entry.syn_type, right_entry.syn_type):
                                    result = rule.apply(left_entry, right_entry)
                                    if result:
                                        chart[(start, end)].append(result)
                
                # Apply expected execution to compress similar derivations
                #self._expected_execution(chart[(start, end)])
        
        return chart[(0, n)]
    
    def _expected_execution(self, entries: List[LexiconEntry]):
        """
        Implement the Expected Execution from the paper
        Compress derivations with identical structure but different subtrees
        """
        if not entries:
            return
            
        # Group entries by their syntactic type 
        by_type = {}
        for entry in entries:
            key = str(entry.syn_type)
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(entry)
        
        # For each group, merge entries with similar structure
        for type_key, type_entries in by_type.items():
            # Find entries with the same program structure
            by_program_structure = {}
            for entry in type_entries:
                # In real implementation, we would check program structure
                # Here, we just use function name as a proxy for structure
                key = entry.sem_program.func_name
                if key not in by_program_structure:
                    by_program_structure[key] = []
                by_program_structure[key].append(entry)
            
            # Merge entries with the same structure
            for prog_key, prog_entries in by_program_structure.items():
                if len(prog_entries) > 1:
                    # Compute the expected program by combining weights using PyTorch operations
                    # Convert weights to a tensor while maintaining gradient tracking
                    weights_tensor = torch.stack([entry.weight for entry in prog_entries])
                    log_softmax_weights = F.log_softmax(weights_tensor, dim=0)
                    
                    # Use LogSumExp trick for numerical stability while maintaining gradients
                    log_total_weight = torch.logsumexp(weights_tensor, dim=0)
                    
                    # Use the first entry as a template
                    merged_entry = prog_entries[0]
                    merged_entry.weight = log_total_weight
                    
                    # Store softmax weights for potential future reference
                    merged_entry.structure_weights = torch.exp(log_softmax_weights)
                    
                    # Remove the merged entries from the original list
                    for entry in prog_entries[1:]:
                        if entry in entries:
                            entries.remove(entry)
    
    def get_parse_probability(self, parse_result):
        """
        Get probabilities of parse results that can be used for gradient-based optimization
        Returns a tensor that can be used for loss computation
        """
        if not parse_result:
            return torch.tensor(0.0, requires_grad=True)
        
        # For multiple parses, combine probabilities
        log_weights = torch.stack([entry.weight for entry in parse_result])
        log_probs = F.log_softmax(log_weights, dim=0)
        
        # Return the log probability of the most likely parse
        # or you could return the logsumexp for the probability of all valid parses
        return log_probs  # Return highest probability parse
    
    def forward(self, sentence: str, target_program: str = None):
        """
        Forward pass for the parser, returns loss that can be used for optimization
        If target_program is specified, computes loss against that specific program
        """
        parses = self.parse(sentence)
        
        # If no parses found, return a high loss
        if not parses:
            return torch.tensor(float('inf'), requires_grad=True)
        
        # If target program specified, find it in the parses
        if target_program:
            for i, parse in enumerate(parses):
                if str(parse.sem_program) == target_program:
                    # Return negative log probability as loss
                    log_weights = torch.stack([entry.weight for entry in parses])
                    log_probs = F.log_softmax(log_weights, dim=0)
                    return -log_probs[i]
            
            # Target program not found - return high loss
            return torch.tensor(float('inf'), requires_grad=True)
            
        # Otherwise, return negative log probability of the best parse
        return -self.get_parse_probability(parses)
