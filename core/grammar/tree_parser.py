'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-12-18 16:07:53
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-12-18 16:08:01
'''

from typing import List, Tuple
from itertools import product

from helchriss.dsl.dsl_types import TypeBase
from helchriss.knowledge.symbolic import Expression, FunctionApplicationExpression, VariableExpression, ConstantExpression

class FunctionEntry:
    def __init__(self , word, fn : str, types : List[TypeBase],weight = 0.0, compat = True):
        self.word   = word
        self.fn     = fn
        self.types  = types
        self.weight = weight
        self.compat = compat
    
    def __repr__(self)-> str:return f"Entry[{self.word}->{self.fn},{self.types},{self.weight},{self.compat}]"

    def __eq__(self, other)-> bool: return (self.word == other.word) and (self.fn == other.fn) and (self.types == other.types)

class TreeParser:
    def __init__(self, entries: List[FunctionEntry] = []):
        self._entries = entries
        self.supress = 1 # supress the mismatch

    def load_entries(self, functions : List[Tuple[str, List[TypeBase], TypeBase]]):
        for function in functions:
            fn, arg_types, out_type = function
            if ":" in fn: word, _ = fn.split(":")
            else: word = fn
            #print(fn, word)
            entry = FunctionEntry(word, fn, arg_types + [out_type], 1.0)
            if entry not in self._entries:

                self._entries.append(entry)

    def get_function_entry(self, word: str) -> List[FunctionEntry]:
        if ":" in word: 
            #print(word,"->",[entry for entry in self._entries if entry.word == word])
            return [entry for entry in self._entries if entry.fn == word]
        else:
            return [entry for entry in self._entries if entry.word == word]

    def parse_program_string(self, program: str):
        return Expression.parse_program_string(program)
    
    def parse(self, program : str):
        program = program.replace(" ", "")
        tree = Expression.parse_program_string(program)
        supress = self.supress # supree the mismatch


        def dp(x : Expression) -> List[FunctionEntry]:
            """return the list"""
            programs = []
            
            if isinstance(x, FunctionApplicationExpression):

                fn : str = x.func
                assert isinstance(fn, VariableExpression)
                fn = fn.name

                args : List[Expression] = x.args
                entries : List[FunctionEntry] = self.get_function_entry(fn)
                
                arg_entries : List[List[FunctionEntry]] = [dp(arg) for arg in args]
                arg_entries = [list(comb) for comb in product(*arg_entries)]
                #print("entries:", len(entries))
                #print("arg entries:",len(arg_entries))
                
                for entry in entries:
                    for arg_pair in arg_entries:

                        not_compat = [arg.types[-1] != entry.types[i] for i,arg in enumerate(arg_pair)]
                        
                        vars = ",".join([arg.fn for arg in arg_pair])
                        words = ",".join([arg.word for arg in arg_pair])

                        if not supress and sum(not_compat) != 0:
                            continue
                        
                        composite_weight = 0.
                        for i,ncomp in enumerate(not_compat):
                            if not ncomp: composite_weight += arg_pair[i].weight
                            else : composite_weight -= arg_pair[i].weight                        

                        composite_entry = FunctionEntry(
                                f"{entry.word}({words})",
                                f"{entry.fn}({vars})",
                                [arg.types[-1] for arg in arg_pair] + [entry.types[-1]],
                                composite_weight + entry.weight,
                                compat = sum(not_compat) == 0
                            )
                        programs.append(composite_entry)
    
                return programs
            if isinstance(x, VariableExpression):
                return self.get_function_entry(x.name)
            raise Exception(f"unknown node type {x}")
        
        programs : List[FunctionEntry] = dp(tree)
        return programs