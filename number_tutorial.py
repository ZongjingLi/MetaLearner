# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
import torch.nn as nn
from typing import List, Union
from helchriss.domain import load_domain_string
from helchriss.knowledge.symbolic import Expression
from helchriss.knowledge.executor import CentralExecutor
from core.metaphors.diagram_executor import MetaphorExecutor
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import CCGSyntacticType, LexiconEntry, SemProgram
from helchriss.knowledge.symbolic import Expression

from domains.numbers.integers_domain import integers_executor
from domains.scene.objects_domain import objects_executor


domains = [
    integers_executor, objects_executor
]

"""create a demo scene for the executor to execute on."""
num_objs = 4
grounding = {"objects": torch.randn([num_objs, 128]), "ref" : torch.randn([num_objs,1])}


def create_sample_lexicon():
    """Create a sample lexicon for testing"""
    lexicon = {}
    
    # Define primitive types
    OBJ = CCGSyntacticType("objset")
    INT = CCGSyntacticType("int")
    
    # Define complex types
    OBJ_OBJ = CCGSyntacticType("objset", OBJ, OBJ, "/")  # objset/objset
    OBJ_OBJ_BACK = CCGSyntacticType("objset", OBJ, OBJ, "\\")  # objset\objset
    INT_OBJ = CCGSyntacticType("int", OBJ, INT, "/")  # int/objset
    
    # Create lexicon entries with PyTorch tensor weights
    
    # Nouns
    lexicon["cube"] = [
        LexiconEntry("cube", OBJ, SemProgram("filter", ["cube"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    lexicon["sphere"] = [
        LexiconEntry("sphere", OBJ, SemProgram("filter", ["sphere"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Adjectives
    lexicon["red"] = [
        LexiconEntry("red", OBJ_OBJ, SemProgram("filter", ["red"], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    lexicon["blue"] = [
        LexiconEntry("blue", OBJ_OBJ, SemProgram("filter", ["blue"], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    lexicon["shiny"] = [
        LexiconEntry("shiny", OBJ_OBJ, SemProgram("filter", ["shiny"], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Count
    lexicon["count"] = [
        LexiconEntry("count", INT_OBJ, SemProgram("count", [], ["x"]), torch.tensor(0.0, requires_grad=True)),
        LexiconEntry("count", INT_OBJ, SemProgram("id-count", [], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Prepositions
    lexicon["of"] = [
        LexiconEntry("of", OBJ_OBJ_BACK, SemProgram("id", [], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Determiners
    lexicon["the"] = [
        LexiconEntry("the", OBJ_OBJ, SemProgram("id", [], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    return lexicon


class Aluneth(nn.Module):
    def __init__(self, domains : List[Union[CentralExecutor]]):
        super().__init__()
        self.parser = ChartParser(create_sample_lexicon())
        self.executor : CentralExecutor = MetaphorExecutor(domains)
    
    def forward(self, sentence, groundings = None):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        
        results = []
        for i,parse in enumerate(parses):
            parse_prob = distrs[i].exp()
            program = str(parse.sem_program)

            
            print(program, parse_prob)
            expr = Expression.parse_program_string(program)

            result = self.executor.evaluate(expr, grounding)
        return results

alunet = Aluneth(domains)

alunet("count blue cube", grounding)