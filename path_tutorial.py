# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
import torch.nn as nn
from typing import List, Union, Any
from helchriss.dsl.dsl_values import Value
from helchriss.domain import load_domain_string
from helchriss.knowledge.symbolic import Expression
from helchriss.knowledge.executor import CentralExecutor

from core.metaphors.diagram_executor import MetaphorExecutor
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import CCGSyntacticType, LexiconEntry, SemProgram
from core.grammar.learn import enumerate_search

from domains.numbers.integers_domain import integers_executor
from domains.scene.objects_domain import objects_executor
from domains.logic.fol_domain import fol_domain_str

domains = [
    integers_executor, objects_executor
]

from tqdm import tqdm

from torch.utils.data import DataLoader
from helchriss.utils.data import ListDataset
from helchriss.utils.data import GroundBaseDataset

class SceneGroundingDataset(ListDataset):
    def __init__(self, queries : List[str], answers : List[Union[Value, Any]], groundings : None):
        query_size = len(queries)
        if groundings is None: groundings = [{} for _ in range(query_size)]
        data = [{"query":queries[i], "answer":answers[i], "grounding": groundings[i]} for i in range(query_size)]
        super().__init__(data)


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
                expr = Expression.parse_program_string(str(program))
                result = self.executor.evaluate(expr, grounding)

                results.append(Value(output_type.split(":")[0],result))
                probs.append(parse_prob)

            else:
                results.append(None)
                probs.append(parse_prob)

        return results,  probs, programs


    def train(self, dataset : SceneGroundingDataset,  epochs : int = 100, lr = 1e-2):
        optim = torch.optim.Adam(self.parameters(), lr = lr)

        for epoch in tqdm(range(epochs)):
            loss = 0.0     
            for idx, sample in dataset:
                query = sample["query"]
                answer = sample["answer"]
                grounding = sample["grounding"]
                results, probs, _ = self(query, grounding)
                for i,result in enumerate(results):
                    measure_conf = torch.exp(probs[i])
                    if result is not None: # filter make sense progams
                        if answer.vtype == result.vtype:
                            measure_loss =  torch.abs(result.value - answer.value)
                            loss += measure_conf * measure_loss
                        else: loss += measure_conf # suppress type-not-match outputs
                    else: loss += measure_conf # suppress the non-sense outputs
            optim.zero_grad()
            loss.backward()
            optim.step()

        return self

    def parse_display(self, sentence):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip(parses, distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)
        for i, parse in enumerate(sorted_parses[:4]):
            print(f"{parse[0].sem_program}, {parse[1].exp():.2f}")
        print("")

from config import config
from helchriss.utils.os import load_corpus
from helchriss.utils.vocab import build_vocab
corpus = load_corpus(config.corpus)
vocab = build_vocab(corpus)


test_sentences = ["two plus one", "two plus three", "one plus one"]
test_answers = [Value("int",3.0),Value("int",5.0), Value("int", 2.0)]
sum_dataset = SceneGroundingDataset(test_sentences, test_answers, groundings = None)


vocab = build_vocab(test_sentences)
model = Aluneth(domains, vocab)



model.train(sum_dataset, epochs = 500, lr = 1e-1)

model.parse_display("one")
model.parse_display("two")
model.parse_display("three")
