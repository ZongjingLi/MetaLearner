# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-04-29 03:28:03
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-04-29 23:31:07
import os
import re
import yaml
import torch
import torch.nn as nn
from typing import List, Union, Any
from helchriss.domain import load_domain_string
from helchriss.knowledge.symbolic import Expression
from helchriss.knowledge.executor import CentralExecutor
from core.metaphors.diagram_executor import ReductiveExecutor, ExecutorGroup
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import CCGSyntacticType, LexiconEntry, SemProgram
from core.grammar.learn import enumerate_search
from helchriss.knowledge.symbolic import Expression
from helchriss.dsl.dsl_values import Value

from tqdm import tqdm
from datasets.base_dataset import SceneGroundingDataset

def write_vocab(vocab, vocab_file = "core_vocab.txt"):
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')

class MetaLearner(nn.Module):
    def __init__(self, domains : List[Union[CentralExecutor]], vocab : List[str] = []):
        super().__init__()
        self.name = "prototype"
        self._domain :List[Union[CentralExecutor]]  = domains
        self.domain_infos = {}
        self.executor : CentralExecutor = ReductiveExecutor(ExecutorGroup(domains))
        
        self.vocab = vocab
        self.lexicon_entries = nn.ModuleDict({})
        self.parser = ChartParser(self.lexicon_entries)

        self.gather_format = self.executor.gather_format
        self.entries_setup() 

    def load_ckpt(self, ckpt_path):

        core_knowledge_config = f"{ckpt_path}/config.yaml"
        state_dict_dir = f"{ckpt_path}/state_dict.pth"

        with open(core_knowledge_config, 'r') as file:
            model_config = yaml.safe_load(file)

        """start to load the model config, core knowledge and lexicon associated with it"""

        # load the domain functions used for the meta-learner
        domain_executors = []
        for domain_name in model_config["domains"]:
            domain =  model_config["domains"][domain_name]
            path = domain["path"]
            name = domain["name"]
            self.domain_infos[domain_name] = {"path" : path, "name" : name}
            exec(f"from {path} import {name}")
            domain_executors.append(eval(name))
        self._domain = domain_executors
        self.executor = ReductiveExecutor(ExecutorGroup(domain_executors))
        self.executor.load_ckpt(ckpt_path)

        # load the lexicon entries and the vocab learned in the meta-learner
        vocab_path = model_config["vocab"]["path"]
        with open(f"{ckpt_path}/{vocab_path}", 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        self.vocab = vocab
        self.lexicon_entries = torch.load(f"{ckpt_path}/lexicon_entries.ckpt", weights_only=False)

        self.name = model_config["name"]
       
        return self

    def save_ckpt(self, ckpt_path):
        if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
        #torch.save(self.state_dict(), f"{ckpt_path}/state_dict.ckpt")

        """create the vocab and corresponding lexicons"""
        data = {
            "name" : self.name,
            "domains": self.domain_infos,
            "vocab": {"path":"core_vocab.txt"},
        }
        write_vocab(self.vocab, f"{ckpt_path}/core_vocab.txt")
        torch.save(self.lexicon_entries,  f"{ckpt_path}/lexicon_entries.ckpt")
        
        """save the structure of the domain knowledge and other stuff"""
        self.executor.save_ckpt(ckpt_path)

        with open(f'{ckpt_path}/config.yaml', 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        return 0

    
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


    def group_lexicon_entries(self, moduledict : List[nn.Module]):
        grouped = {}
        pattern = re.compile(r'(.+?)_(\d+)$')  # match 'word_{i}'
    
        for key, module in moduledict.items():
            match = pattern.match(key)
            if match:
                word, idx = match.groups()
                idx = int(idx)
                if word not in grouped:
                    grouped[word] = [module]
                else:
                    grouped[word].append(module)
            else:
                raise ValueError(f"Key '{key}' does not match expected pattern 'word_idx'")

        return grouped

    def entries_setup(self, depth = 1):
        entries = enumerate_search(self.types, self.functions, max_depth = depth)
    
        self.lexicon_entries = nn.ModuleDict({})
    

        for word in self.vocab:
            for idx, (syn_type, program) in enumerate(entries):
                self.lexicon_entries[f"{word}_{idx}"] = LexiconEntry(
                        word, syn_type, program, weight = torch.randn(1).item() - 0.0
                    )
    
        grouped_lexicon = self.group_lexicon_entries(self.lexicon_entries)
    
        self.parser = ChartParser(grouped_lexicon)
        self.lexicon_entries = None
        return 0

    def add_vocab(self, vocab: List[str], domains : List[str]):
        """this method add a new set of vocab and related domains that could associate it with """
        return -1

    def forward(self, sentence, grounding = None, topK = None, train = True):

        parses = self.parser.parse(sentence,  topK = topK)
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
                results.append(result)
                probs.append(parse_prob)

            else:
                results.append(None)
                probs.append(parse_prob)

        import matplotlib.pyplot as plt
        plt.cla()
        self.executor.display("assets/static/images/parse_tree")

        return results, probs, programs

    def infer_metaphor_expressions(self, meta_exprs: List[Expression]):
        for meta_expr in meta_exprs:
            infers = self.executor.infer_reductions(meta_expr)
            self.executor.add_metaphors(infers)

    def train(self, dataset : SceneGroundingDataset,  epochs : int = 100, lr = 1e-2, topK = None):
        import tqdm.gui as tqdmgui
        optim = torch.optim.Adam(self.parameters(), lr = lr)
        # epoch_bar = tqdmgui.tqdm(range(epochs), desc="Training epochs", unit="epoch")
        #print(list(self.parameters()))

        epoch_bar = tqdm(range(epochs), desc="Training epochs", unit="epoch")

        for epoch in epoch_bar:
            loss = 0.0     
            for idx, sample in dataset:
                query = sample["query"]
                answer = sample["answer"]
                grounding = sample["grounding"]
                results, probs, programs = self(query, grounding)
                if not results: print(f"no parsing found for query:{query}")
                for i,result in enumerate(results):

                    measure_conf = torch.exp(probs[i])
                    if result is not None: # filter make sense progams
                        assert isinstance(result, Value), f"{programs[i]} result is :{result} and not a Value type"

                        if answer.vtype in result.vtype.alias:

                            if answer.vtype == "boolean":
                                measure_loss =  torch.nn.functional.binary_cross_entropy_with_logits(result.value - answer.value)
                            if answer.vtype == "int" or answer.type == "float":
                                measure_loss = torch.abs(result.value - answer.value)

                            loss += measure_conf * measure_loss
                        else:

                            loss += measure_conf # suppress type-not-match outputs

                        #from torchviz import make_dot
                        #dot = make_dot(loss, params=dict(self.named_parameters()))
                        #dot.render("computation_graph", format="png")  # Saves to computation_graph.png
                        #dot.view()
                        #dot.render("graph", format="png")


                    else: loss += measure_conf # suppress the non-sense outputs
            optim.zero_grad()
            #loss.backward()
            params_with_grad = [
             (name, param) for name, param in self.named_parameters()
                if param.grad is not None and param.grad.abs().sum() > 0
            ]

            for name, param in params_with_grad:
                print(name)
            optim.step()

            avg_loss = loss / len(dataset) if len(dataset) > 0 else 0
            epoch_bar.set_postfix({"avg_loss": f"{avg_loss.item():.4f}"})

        return self

    def parse_display(self, sentence):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip(parses, distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)
        for i, parse in enumerate(sorted_parses[:4]):
            print(f"{parse[0].sem_program}, {float(parse[1].exp()):.2f}")
        print("")