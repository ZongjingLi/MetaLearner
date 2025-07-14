# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:25:05
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-04-29 06:25:38
import re
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import torch
import torch.nn as nn
from helchriss.dsl.dsl_types import TypeBase

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
        #try:
        #    pass
        #    print(self.arg_type)
        #    print(self.result_type)
        #except:
        #    print(self.arg_type.__class__)
        #    print(self.result_type.__class__)
    @property
    def is_primitive(self): return  self.result_type is None
    
    def __str__(self):
        if self.is_primitive:
            return str(self.name)
        else:
            return f"{self.result_type}{self.direction}{self.arg_type}"
    
    def __eq__(self, other):
        if self.is_primitive and other.is_primitive:
            if isinstance(self.name, TypeBase) and isinstance(other, TypeBase):
                return self.name == other.name
            else: return str(self.name) == str(other.name)
        else:
            return (self.direction == other.direction and 
                    self.result_type == other.result_type and 
                    self.arg_type == other.arg_type)
    
    def __hash__(self):
        if self.is_primitive:
            return hash(self.name)
        else:
            return hash((self.direction, str(self.result_type), str(self.arg_type)))

    @staticmethod
    def syntatic_type_from_string(type_str: str):
        def tok(s):
            out, buf = [], ''
            for c in s:
                if c in '()/\\': out += [buf] * bool(buf) + [c]; buf = ''
                elif c.isspace(): out += [buf] * bool(buf); buf = ''
                else: buf += c
            return out + [buf] * bool(buf)

        def parse(toks):
            def expr(i):
                if toks[i] == '(':
                    res, j = recur(i + 1)
                    assert toks[j] == ')'
                    return res, j + 1
                return CCGSyntacticType(toks[i]), i + 1

            def recur(i):
                left, i = expr(i)
                while i < len(toks) and toks[i] in '/\\':
                    d, right, i = toks[i], *expr(i + 1)
                    left = CCGSyntacticType(f"{left}{d}{right}", left, right, d)
                return left, i

            res, idx = recur(0)
            if idx != len(toks):
                raise ValueError("Trailing tokens: " + str(toks[idx:]))
            return res

        return parse(tok(type_str))


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
    
    def __hash__(self): 
        return hash(self.__str__())
    
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
    def __init__(self, word: str, syn_type: CCGSyntacticType, sem_program: SemProgram, weight: Union[float, torch.Tensor] = 0.0):
        super().__init__()
        self.word = word
        self.syn_type = syn_type
        self.sem_program = sem_program
        
        # Convert weight to parameter
        if isinstance(weight, float) or isinstance(weight, int):
            self.weight = torch.tensor(float(weight), requires_grad=True)
        elif isinstance(weight, torch.Tensor) and not isinstance(weight, nn.Parameter):
            self.weight = weight
        else:
            self.weight = weight  # Already a parameter
    
    
    def __str__(self):
        weight_value = self.weight.item()
        return f"{self.word} : {self.syn_type} : {self.sem_program} : {weight_value:.3f}"
    
