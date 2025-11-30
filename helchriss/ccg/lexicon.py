# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:25:05
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-20 08:59:38
import math
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CCGSyntacticType:
    """
    Represents syntactic types in Combinatory Categorial Grammar (CCG)
    Can be primitive (e.g., objset) or complex (e.g., objset/objset)
    """
    def __init__(self, name: str, arg_type=None, result_type=None, direction=None):
        self.name = name
        self.arg_type = arg_type  # For complex types
        self.result_type = result_type  # For complex types
        self.direction = direction  # '/' for forward, '\' for backward
    
    @property
    def is_primitive(self): return self.arg_type is None and self.result_type is None
    
    def __str__(self):
        if self.is_primitive:
            return self.name
        else:
            return f"{self.result_type}{self.direction}{self.arg_type}"
    
    def __eq__(self, other):
        if self.is_primitive and other.is_primitive:
            return self.name == other.name
        else:
            return (self.direction == other.direction and 
                    self.result_type == other.result_type and 
                    self.arg_type == other.arg_type)
    
    def __hash__(self):
        if self.is_primitive:
            return hash(self.name)
        else:
            return hash((self.direction, str(self.result_type), str(self.arg_type)))


class SemProgram:
    """Represents a semantic program or lambda function"""
    def __init__(self, func_name: str, args=None, lambda_vars=None):
        self.func_name = func_name
        self.args = args if args else []
        self.lambda_vars = lambda_vars if lambda_vars else []  # For lambda functions
    
    def execute(self, context=None):
        """Execute the program in a given context"""
        # This would execute on input data like images in the full implementation
        # Here we'll just return a dummy result for demonstration
        return f"Executed {self}"
    
    def __str__(self):
        if self.lambda_vars:
            lambda_str = "λ" + ".".join(self.lambda_vars) + "."
            return f"{lambda_str}{self.func_name}({', '.join(str(arg) for arg in self.args)})"
        else:
            return f"{self.func_name}({', '.join(str(arg) for arg in self.args)})"
    
    def __eq__(self, other):
        return (self.func_name == other.func_name and 
                self.args == other.args and 
                self.lambda_vars == other.lambda_vars)


class LexiconEntry:
    """
    A lexicon entry for a word, containing syntactic type and semantic program
    """
    def __init__(self, word: str, syn_type: CCGSyntacticType, sem_program: SemProgram, weight: Union[float, torch.Tensor] = 0.0):
        self.word = word
        self.syn_type = syn_type
        self.sem_program = sem_program
        
        # Convert weight to PyTorch tensor if it's not already
        if isinstance(weight, float):
            self._weight = torch.tensor(weight, requires_grad=True)
        else:
            self._weight = weight
    
    @property
    def weight(self): return torch.log( torch.sigmoid( self._weight))

    def __str__(self):
        weight_value = self.weight.item() if isinstance(self.weight, torch.Tensor) else math.exp(self.weight)
        return f"{self.word} : {self.syn_type} : {self.sem_program} : {weight_value:.3f}"
