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
from core.grammar.learn import enumerate_search
from helchriss.knowledge.symbolic import Expression

from domains.numbers.integers_domain import integers_executor
from domains.scene.objects_domain import objects_executor

domains = [
    integers_executor, #objects_executor
]


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
    def __init__(self, domains : List[Union[CentralExecutor]], vocab = None):
        super().__init__()
        self._domain :List[Union[CentralExecutor]]  = domains
        self.executor : CentralExecutor = MetaphorExecutor(domains)
        
        self.vocab = vocab
        self.parser = None
        self.gather_format = self.executor.gather_format

        self.entries_setup()
    
    def freeze_word(self,word):
        for entry in self.parser.word_weights:
            if word in entry:
                self.parser.word_weights[entry]._requires_grad = False

    @property
    def domains(self):
        gather_domains = []
        for domain in self._domain:
            if isinstance(domain, CentralExecutor):
                gather_domains.append(domain.domain)
            else: gather_domains.append(domain)
        return gather_domains

    def collect_funcs(self,func_bind, domain):
        bind = {
            "name" : func_bind["name"],
            "parameters" : [self.gather_format(param.split("-")[-1], domain)  for param in func_bind["parameters"]],
            "type" : self.gather_format(func_bind["type"], domain)
            }
        return bind
    
    @property
    def types(self):
        domain_types = {}
        for domain in self.domains:
            for tp in domain.types:
                domain_types[self.gather_format(tp, domain.domain_name)] =  domain.types[tp].replace("'","")
        return domain_types

    @property
    def functions(self):
        domain_functions = {}
        for domain in self.domains:
            domain_name = domain.domain_name
            for func in domain.functions:
                domain_functions[self.gather_format(func, domain_name)] =  self.collect_funcs(domain.functions[func], domain_name)
        return domain_functions

    def entries_setup(self, depth = 1):
        self.entries = enumerate_search(self.types, self.functions, max_depth = depth)
        lexicon_entries = {} 
        for word in self.vocab:
            lexicon_entries[word] = []
            for syn_type, program in self.entries:

                lexicon_entries[word].append(LexiconEntry(
                    word, syn_type, program, weight = torch.tensor(-10.0, requires_grad=True)
                ))

        self.lexicon_entries = lexicon_entries
        #for entry in self.entries:print(entry[0], entry[1])
        self.parser = ChartParser(lexicon_entries)

    def forward(self, sentence, grounding = None):
        parses = self.parser.parse(sentence)
        log_distrs = self.parser.get_parse_probability(parses)
        
        results = []
        probs = []
        programs = []
        for i,parse in enumerate(parses):
            parse_prob = log_distrs[i]
            program = str(parse.sem_program)

            #if len(parse.sem_program.lambda_vars) == 0:            

            try:
                    expr = Expression.parse_program_string(program)
                    result = self.executor.evaluate(expr, grounding)
                    results.append(result)
                    probs.append(parse_prob)
                    programs.append(program)
            except:pass
        return results,  probs, programs


"""create a demo scene for the executor to execute on."""
num_objs = 4
grounding = {"objects": torch.randn([num_objs, 128]), "scores" : torch.randn([num_objs,1])}

vocab = ["one", "plus", "two", "three"]

alunet = Aluneth(domains, vocab)


optim = torch.optim.Adam(alunet.parameters(), lr = 1e-1)
print("start training of the parsing")



def train(model, epochs, test_sentences, test_answers):
    for itrs in range(epochs):
        loss = 0.0
        for i,sent in enumerate(test_sentences):
            results, probs, programs = model(sent, grounding)
            for j,result in enumerate(results):
                try:
                    if torch.abs(result - test_answers[i]) < 0.1:
                        loss -= probs[j]
                        #print(programs[j], probs[j])
                except:pass
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(loss)
    return model

test_sentences = ["one", "two","three"]
test_answers = [ 1.0, 2.0, 3.0]

alunet = train(alunet, 1000, test_sentences, test_answers)


for entry in alunet.lexicon_entries["one"]:
    print(entry)

test_sentences = ["two plus two", "two", "one", "one", "two plus one"]
test_answers = [4.0, 2.0, 1.0, 1.0, 3.0]

alunet = train(alunet, 1000, test_sentences, test_answers)


parses = alunet.parser.parse("two plus two")
distrs = alunet.parser.get_parse_probability(parses)

for i, parse in enumerate(parses[:]):print(parse.sem_program, float(distrs[i].exp()))

for entry in alunet.lexicon_entries["plus"]:
    print(entry)
