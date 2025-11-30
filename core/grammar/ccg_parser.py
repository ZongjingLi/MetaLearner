import copy
import math
import torch
import tabulate
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .lexicon import CCGSyntacticType, SemProgram, LexiconEntry
from typing import Dict, List, Tuple, Set, Optional, Any, Union

class CCGRule:
    """Abstract base class for CCG combinatory rules"""
    mismatch_logit = 10. ### magic number for the match-regularity

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
    def can_apply(left_type : CCGSyntacticType, right_type : CCGSyntacticType):
        if (not left_type.is_primitive):
            pass
            #print(left_type.arg_type, right_type, )
        return (not left_type.is_primitive and 
                left_type.direction == '/' and 
                str(left_type.arg_type) == str(right_type))
    
    @staticmethod
    def can_forced_apply(left_type : CCGSyntacticType, right_type : CCGSyntacticType):
        return (not left_type.is_primitive and
                left_type.direction == "/")

    @staticmethod
    def forced_logit(left_type : CCGSyntacticType , right_type : CCGSyntacticType):
        return ((left_type.arg_type == right_type) - 0.5) * CCGRule.mismatch_logit

    @staticmethod
    def apply(left_entry : LexiconEntry, right_entry : LexiconEntry, offset : Union[float, torch.Tensor]):
        result_type = left_entry.syn_type.result_type
        if left_entry.sem_program.lambda_vars:
            new_args = left_entry.sem_program.args.copy()
            new_args.append(right_entry.sem_program)
            new_program = SemProgram(
                left_entry.sem_program.func_name,
                new_args,
                left_entry.sem_program.lambda_vars[1:] if len(left_entry.sem_program.lambda_vars) > 1 else []
            )
        else:
            new_args = left_entry.sem_program.args.copy()
            new_args.append(right_entry.sem_program)
            new_program = SemProgram(left_entry.sem_program.func_name, new_args)
        combined_weight = left_entry.weight + right_entry.weight + offset
        return LexiconEntry(f"{left_entry.word} {right_entry.word}", result_type, new_program, combined_weight)

class BackwardApplication(CCGRule):
    """Backward application rule: Y X\Y => X"""
    @staticmethod
    def can_apply(left_type : CCGSyntacticType, right_type : CCGSyntacticType):

        return (not right_type.is_primitive and 
                right_type.direction == '\\' and 
                right_type.arg_type == left_type)

    @staticmethod
    def can_forced_apply(left_type : CCGSyntacticType, right_type : CCGSyntacticType):
        return (not right_type.is_primitive and 
                right_type.direction == '\\')

    @staticmethod
    def forced_logit(left_type : CCGSyntacticType , right_type : CCGSyntacticType):
        return ((right_type.arg_type == left_type) - 0.5) * CCGRule.mismatch_logit

    @staticmethod
    def apply(left_entry : LexiconEntry, right_entry : LexiconEntry, offset : Union[float, torch.Tensor]):
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

        combined_weight = left_entry.weight + right_entry.weight + offset
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
            module_list = []
            for idx, entry in enumerate(entries):
                weight_key = f"{word_key}_{idx}"
                if isinstance(entry.weight, nn.Parameter):
                    weight_dict[weight_key] = entry.weight
                else:
                    # Convert to parameter if it's not already
                    weight_dict[weight_key] = nn.Parameter(entry.weight)
                
                # Add entry to module list (it will be used as a template)
                module_list.append(entry)
            
            module_dict[word_key] = (module_list)
        
        self.lexicon : Dict = module_dict
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
        weight_dict = torch.load(load_path, weights_only=True)
        
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

    def string_to_program(self, program : str):
        from helchriss.knowledge.symbolic import Expression, FunctionApplicationExpression, ConstantExpression, VariableExpression
        expr = Expression.parse_program_string(program)
        def convert(expr: Expression) -> SemProgram:
            if isinstance(expr, FunctionApplicationExpression):
                func_name = expr.func.name
                args = [convert(arg) for arg in expr.args]
                return SemProgram(func_name, args)

            elif isinstance(expr, ConstantExpression): return SemProgram(str(expr.const), [])
            elif isinstance(expr, VariableExpression): return SemProgram(str(expr.name), [])
            else:
                raise TypeError(f"Unsupported expression type: {type(expr)}")

        return convert(expr)
    
    def generate_sentences_for_program(self, target_program: str, max_depth: int = 10) -> List[str]:
        """
        Given a target SemProgram, generate possible surface-level sentences 
        that parse to it using the lexicon and CCG rules.

        Args:
            target_program (SemProgram): The semantic program tree to match.
            max_depth (int): Maximum depth of recursive search to avoid infinite loops.

        Returns:
            List[str]: List of possible surface sentences.
        """
        memo = {}
        target_program : SemProgram = self.string_to_program(target_program)

        def match_program(prog: SemProgram, depth: int) -> List[List[LexiconEntry]]:
            if depth > max_depth: return []

            key = (prog.func_name, tuple(str(arg) for arg in prog.args), tuple(prog.lambda_vars))
            if key in memo: return memo[key]

            matches = []

            # Try direct matches from lexicon entries
            for word, entries in self.lexicon.items():
                for idx, entry in enumerate(entries):
                    lex_prog = entry.sem_program
                    if lex_prog == prog:
                        matches.append([self.gather_word_entries(word)[idx]])

            # Try to decompose into argument + function applications
            for rule in self.rules:
                if not prog.args:
                    continue  # Can't apply rules to a nullary program
            
                # Try matching function and arguments recursively
                func_prog = SemProgram(prog.func_name, prog.args[:-1], prog.lambda_vars[1:] if prog.lambda_vars else [])
                arg_prog = prog.args[-1]

                left_options = match_program(func_prog, depth + 1)
                right_options = match_program(arg_prog, depth + 1)

                for left in left_options:
                    for right in right_options:
                        left_entry = left[-1]
                        right_entry = right[-1]
                        if rule.can_apply(left_entry.syn_type, right_entry.syn_type):
                            combined = rule.apply(left_entry, right_entry)
                            if combined and combined.sem_program == prog:
                                matches.append(left + right)

            memo[key] = matches
            return matches

        candidate_derivations = match_program(target_program, 0)

        # Convert LexiconEntry sequences into surface sentences
        sentences = [" ".join(entry.word for entry in derivation) for derivation in candidate_derivations]
    
        return sorted(set(sentences))

    def purge_entry(self, word: str, p: float, abs: bool = False):
        """only keep the word entries with weight greater than or equal to threshold p
        Args:
            word: The word to purge entries from
            p: The threshold value for purging
            abs: If True, use absolute threshold; if False, normalize weights with softmax first
        """
        if word not in self.lexicon: return
    
        # gather all weights for query word
        weights = []
        for idx in range(len(self.lexicon[word])):
            weight_key = f"{word}_{idx}"
            weights.append(self.lexicon_weight[weight_key].detach().clone())
    

        weights_tensor = torch.stack(weights)
        if not abs: normalized_weights = torch.nn.functional.softmax(weights_tensor, dim=0)
        else: normalized_weights = torch.sigmoid(weights_tensor)
    

        keep_indices = []
        for idx, norm_weight in enumerate(normalized_weights):
            if norm_weight.item() >= p:
                keep_indices.append(idx)
    
        # If no entries are kept, remove the word entirely
        if not keep_indices:
            del self.lexicon[word]
            # Remove all weights for this word
            keys_to_remove = [k for k in self.lexicon_weight.keys() if k.startswith(f"{word}_")]
            for key in keys_to_remove:
                del self.lexicon_weight[key]
            return
    
        # Create new entries list with only the kept entries
        new_entries = [self.lexicon[word][i] for i in keep_indices]
    
        # Store the old weights we want to keep
        old_weights = {}
        for idx in keep_indices:
            old_key = f"{word}_{idx}"
            old_weights[old_key] = self.lexicon_weight[old_key]
    
        # Update lexicon with filtered entries
        self.lexicon[word] = new_entries
    
        # Remove all old weights for this word
        keys_to_remove = [k for k in self.lexicon_weight.keys() if k.startswith(f"{word}_")]
        for key in keys_to_remove: del self.lexicon_weight[key]
    
        # Add back the weights we want to keep with new indices
        for new_idx, old_idx in enumerate(keep_indices):
            old_key = f"{word}_{old_idx}"
            new_key = f"{word}_{new_idx}"
            self.lexicon_weight[new_key] = old_weights[old_key]
    
    def add_word_entries(self, word : str, entries : List[LexiconEntry]):
        if word in self.lexicon: existing_entries = list(self.lexicon[word])
        else: existing_entries = []
        start_idx = len(existing_entries)
    
        for i, entry in enumerate(entries):
            idx = start_idx + i
            weight_key = f"{word}_{idx}"
        
        if hasattr(entry, 'weight'):
            if isinstance(entry.weight, nn.Parameter): self.lexicon_weight[weight_key] = entry.weight
            else: self.lexicon_weight[weight_key] = nn.Parameter(entry.weight)
        else: self.lexicon_weight[weight_key] = nn.Parameter(torch.tensor(0.0))
        self.lexicon[word] = existing_entries + entries
    
    def clear_word_entry(self, word): self.lexicon[word] = []
    
    def gather_word_entries(self, word : str) -> List[LexiconEntry]:
        if word not in self.lexicon: return []
        entries = []
        for idx, entry in enumerate(self.lexicon[word]):
            weight_key = f"{word}_{idx}"
            weight = self.lexicon_weight[weight_key]
        
            # Create a copy of the entry with the current weight
            entry_copy = copy.deepcopy(entry)
            entry_copy.weight = weight
            entries.append(entry_copy)
        return entries

    def display_word_entries(self, word : str, verbose = True):
        entries = self.gather_word_entries(word)
        headers = ["word", "type", "program", "weight"]
        data = sorted(
            [(word, str(entry.syn_type), entry.sem_program, entry.weight) for entry in entries],
            key=lambda x: x[3],  # weight is the 4th element (index 3)
            reverse=True         # descending order
            )
        table = tabulate.tabulate(data, headers = headers, tablefmt = "grid")
        if verbose: print(table)
        return table

    def get_entry_weight(self, word: str, idx: int) -> torch.Tensor:
        """Get weight for a lexicon entry from the centralized parameter dictionary"""
        weight_key = f"{word}_{idx}"
        return self.lexicon_weight[weight_key]
    
    def get_likely_entries(self, word: str, K: int = 3) -> List:
        """Get top K lexicon entries for a word based on their weights"""
        if word not in self.lexicon:  return []
        
        # get entries and their weights
        entries = self.lexicon[word]
        weights = [self.get_entry_weight(word, idx) for idx in range(len(entries))]
        
        # sort by weights (descending)
        entry_weights = list(zip(entries, weights))
        sorted_entries = sorted(entry_weights, key=lambda x: x[1], reverse=True)[:K]
        

        result = []
        for entry , weight in sorted_entries:
            assert isinstance(entry, LexiconEntry)
            new_entry = LexiconEntry(
                entry.word,
                entry.syn_type,
                entry.sem_program,
                weight  # This is an nn.Parameter
            )
            result.append(new_entry)
        
        return result
    
    def type_downcast(self, tp1 : str, tp2 : str) -> bool:
        return tp1 == tp2 or tp1 == 'AnyType' or tp2 == 'AnyType'

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
                                    offset = rule.forced_logit(left_entry.syn_type, right_entry.syn_type) if forced else 0.
                                    #print(left_entry.syn_type, right_entry.syn_type, offset)
                                    result = rule.apply(left_entry, right_entry, offset)
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
