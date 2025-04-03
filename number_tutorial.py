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
from helchriss.dsl.dsl_values import Value

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
                    word, syn_type, program, weight = torch.tensor(-0.0, requires_grad=True)
                ))

        self.lexicon_entries = lexicon_entries
        self.parser = ChartParser(lexicon_entries)

    def forward(self, sentence, grounding = None):
        parses = self.parser.parse(sentence)
        log_distrs = self.parser.get_parse_probability(parses)
        
        results = []
        probs = []
        programs = []
        for i,parse in enumerate(parses):
            parse_prob = log_distrs[i]
            program = parse.sem_program
            output_type = self.functions[program.func_name]["type"]

            if len(program.lambda_vars) == 0:
                print(program, program)
                expr = Expression.parse_program_string(str(program))
                result = self.executor.evaluate(expr, grounding)

                results.append(Value(output_type.split(":")[0],result))
                probs.append(parse_prob)

            else:
                results.append(None)
                probs.append(parse_prob)

        return results,  probs, programs

    def parse_display(self, sentence):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip(parses, distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)
        for i, parse in enumerate(sorted_parses[:4]):
            print(f"{parse[0].sem_program}, {parse[1].exp():.2f}")
        print("")


"""create a demo scene for the executor to execute on."""
num_objs = 4
grounding = {"objects": torch.randn([num_objs, 128]), "scores" : torch.randn([num_objs,1])}

vocab = ["one", "plus", "two", "three"]

alunet = Aluneth(domains, vocab)


print("start training of the parsing")
from tqdm import tqdm

def train(model, epochs, test_sentences, test_answers):
    optim = torch.optim.Adam(model.parameters(), lr = 1e-1)
    for itrs in tqdm(range(epochs)):
        loss = 0.0
        for i,sent in enumerate(test_sentences):
            results, probs, programs = model(sent, grounding)
            for j,result in enumerate(results):
                answer = test_answers[i]
                if result is not None: # filter make sense progams
                    if answer.vtype == result.vtype:
                        loss += torch.exp(probs[j]) * torch.abs(result.value - answer.value)
                    else: loss += torch.exp(probs[j])
                else: loss += torch.exp(probs[j])
        optim.zero_grad()
        loss.backward()
        optim.step()

    return model

test_sentences = ["one", "two","three"]
test_answers = [Value("int",1.0), Value("int",2.0), Value("int",3.0)]

alunet = train(alunet, 100, test_sentences, test_answers)
alunet.parse_display("one")
alunet.parse_display("two")
alunet.parse_display("three")


test_sentences = ["two", "one", "two plus one", "two plus two"]
test_answers = [Value("int",2.0),Value("int",1.0),Value("int",3.0),Value("int",4.0)]

alunet = train(alunet, 100, test_sentences, test_answers)

alunet.parse_display("one plus two")
alunet.parse_display("one")
alunet.parse_display("two")

#for entry in alunet.lexicon_entries["plus"]:
#    print(entry)
