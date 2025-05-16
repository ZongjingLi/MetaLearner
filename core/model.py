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
import matplotlib.pyplot as plt
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
        self.lexicon_entries = {}#nn.ModuleDict({})
        self.parser = ChartParser(self.lexicon_entries)

        self.gather_format = self.executor.gather_format
        self.entries_setup() 

    def save_ckpt(self, ckpt_path):
        """
        Save the model checkpoint including lexicon entries, weights, and domain information
        
        Args:
            ckpt_path: Path to save the checkpoint
        """
        if not os.path.exists(ckpt_path):  os.makedirs(ckpt_path)

        """ create the vocab and corresponding lexicons for the parser"""
        data = {
            "name": self.name,
            "domains": self.domain_infos,
            "vocab": {"path": "core_vocab.txt"},
        }
        write_vocab(self.vocab, f"{ckpt_path}/core_vocab.txt")
        
        # save lexicon entries structure (without weights) only function and type
        serializable_lexicon = {}
        for word, entries in self.parser.lexicon.items():
            serializable_entries = []
            for entry in entries:
                # Create serializable version (without nn.Parameter weights)
                serializable_entry = {
                    'word': entry.word,
                    'syn_type': entry.syn_type,
                    'sem_program': entry.sem_program,
                    'idx': entries.index(entry)  # Keep track of original index
                }
                serializable_entries.append(serializable_entry)
            serializable_lexicon[word] = serializable_entries
        torch.save(serializable_lexicon, f"{ckpt_path}/lexicon_entries.pth")
        
        self.parser.save_weights(f"{ckpt_path}/lexicon_weights.pth") # save lexicon weights separately
        self.executor.save_ckpt(ckpt_path) # save domain knowledge and other stuff

        with open(f'{ckpt_path}/config.yaml', 'w') as file: # save config file
            yaml.dump(data, file, default_flow_style=False)
        
        return 0

    def load_ckpt(self, ckpt_path):
        """
        Load the model checkpoint including lexicon entries, weights, and domain information
        
        Args:
            ckpt_path: Path to load the checkpoint from
        """
        core_knowledge_config = f"{ckpt_path}/config.yaml"
        state_dict_path = f"{ckpt_path}/state_dict.pth"

        with open(core_knowledge_config, 'r') as file:
            model_config = yaml.safe_load(file)

        """ load the model config, core knowledge and lexicon associated with it """
        
        # load the domain functions used for the meta-learner
        domain_executors = []
        for domain_name in model_config["domains"]:
            domain = model_config["domains"][domain_name]
            path = domain["path"]
            name = domain["name"]
            self.domain_infos[domain_name] = {"path": path, "name": name}
            exec(f"from {path} import {name}")
            domain_executors.append(eval(name))
        
        self._domain = domain_executors
        self.executor = ReductiveExecutor(ExecutorGroup(domain_executors))
        self.executor.load_ckpt(ckpt_path)

        # load the vocabulary
        vocab_path = model_config["vocab"]["path"]
        with open(f"{ckpt_path}/{vocab_path}", 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        self.vocab = vocab
        
        # load the lexicon entries (structure without weights)
        serialized_lexicon = torch.load(f"{ckpt_path}/lexicon_entries.pth", weights_only=False)
        
        # reconstruct lexicon entries
        lexicon_entries = {}
        for word, entries in serialized_lexicon.items():
            word_entries = []
            for entry_data in entries:
                # Create LexiconEntry with a placeholder weight (will be updated later)
                entry = LexiconEntry(
                    entry_data['word'],
                    entry_data['syn_type'],
                    entry_data['sem_program'],
                    torch.tensor(0.0)  # Placeholder weight, will be replaced by loading weights
                )
                word_entries.append(entry)
            lexicon_entries[word] = word_entries

        self.parser = ChartParser(lexicon_entries) # create parser with the loaded lexicon
        
        self.parser.load_weights(f"{ckpt_path}/lexicon_weights.pth") # load lexicon weights

        self.name = model_config["name"]
        return self


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
    
    def filter_types(self, domains : Union[str, List[str]]):
        if isinstance(domains, str): domains = [domains]
        domain_types = {}
        for domain in self.domains:
            domain_name = domain.domain_name
            if "Any" in domains or domain_name in domains:
                for tp in domain.types:
                    domain_types[self.gather_format(tp, domain.domain_name)] =  domain.types[tp].replace("'","")
        return domain_types

    @property
    def types(self): return self.filter_types("Any")

    def filter_functions(self, domains : Union[str, List[str]]):
        if isinstance(domains, str): domains = [domains]
        domain_functions = {}
        for domain in self.domains:
            domain_name = domain.domain_name
            if "Any" in domains or domain_name in domains:
                for func in domain.functions:
                    domain_functions[self.gather_format(func, domain_name)] =  self.collect_funcs(domain.functions[func], domain_name)
        return domain_functions

    @property
    def functions(self): return self.filter_functions("Any")

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
    
    def add_word_lexicon(self):
        return
    
    def delete_word_lexicon(self):
        return
    
    def purge_word_lexicon(self):
        return 

    def entries_setup(self, related_vocab : List[str] = None, domains : Union[str,List[str]] = "Any" ,depth = 1):
        if related_vocab is None: related_vocab = self.vocab

        related_types = self.filter_types(domains)
        related_funcs = self.filter_functions(domains)
        entries = enumerate_search(related_types, related_funcs, max_depth = depth)
    
        lexicon_entries = dict()
        for word in related_vocab:
            for idx, (syn_type, program) in enumerate(entries):
                lexicon_entries[f"{word}_{idx}"] = LexiconEntry(
                        word, syn_type, program, weight = torch.randn(1).item() - 0.0
                    )

        grouped_lexicon = self.group_lexicon_entries(lexicon_entries)    
        self.parser = ChartParser(grouped_lexicon)
        return 0

    #TODO: need the actual learned vocab by minimize the entropy or not converge
    @property
    def learned_vocab(self):return self.vocab

    def add_vocab(self, add_vocab: List[str]):
        """this method add a new set of vocab and related domains that could associate it with """
        self.vocab.extend(add_vocab)

    def forward(self, sentence : str, grounding = None, topK = None, plot : bool = False):

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
                programs.append(program)

            else:
                results.append(None)
                probs.append(parse_prob)

        return results, probs, programs

    def infer_metaphor_expressions(self, meta_exprs: Union[Expression,List[Expression]]):
        if isinstance(meta_exprs, Expression): meta_exprs = [meta_exprs]
        for meta_expr in meta_exprs:
            infers = self.executor.infer_reductions(meta_expr)
            self.executor.add_metaphors(infers)


    def train(self, dataset : SceneGroundingDataset,  epochs : int = 1000, lr = 1e-2, topK = None):
        import tqdm.gui as tqdmgui
        optim = torch.optim.Adam(self.parameters(), lr = lr)
        # epoch_bar = tqdmgui.tqdm(range(epochs), desc="Training epochs", unit="epoch")

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
                        #print(answer, answer.vtype, result, result.vtype)
                        if answer.vtype in result.vtype.alias:
                            if answer.vtype == "boolean":
                                measure_loss =  torch.nn.functional.binary_cross_entropy_with_logits(
                                    result.value , torch.tensor(answer.value))
                            if answer.vtype == "int" or answer.vtype == "float":
                                measure_loss = torch.abs(result.value - answer.value)

                            loss += measure_conf * measure_loss
                        else:
                            lex_loss = 100.
                            loss += measure_conf * lex_loss # suppress type-not-match outputs

                    else: loss += measure_conf # suppress the non-sense outputs
            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss = loss / len(dataset) if len(dataset) > 0 else 0
            epoch_bar.set_postfix({"avg_loss": f"{avg_loss.item():.4f}"})

        return {"loss" : avg_loss}
    
    def maximal_parse(self, sentence):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip(parses, distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)
        return sorted_parses

    def parse_display(self, sentence, topK = 4):
        sorted_parses = self.maximal_parse(sentence)
        for i, parse in enumerate(sorted_parses[:topK]):
            print(f"{parse[0].sem_program}, {float(parse[1].exp()):.2f}")
        print("\n")

    
    def verbose_call(self, sentence, grounding = {}, plot = True, K = 4):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip(parses, distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)
        values = []
        programs = []
        weights = []
        for i, parse in enumerate(sorted_parses[:K]):
            value = self.executor.evaluate(str(parse[0].sem_program), grounding)
            values.append(value)
            programs.append(str(parse[0].sem_program))
            weights.append(float(parse[1].exp()))

        for i, parse in enumerate(sorted_parses[:1]): value = self.executor.evaluate(str(parse[0].sem_program), grounding)
        return values, weights, programs
    
    def eval_graph(self):
        from core.metaphors.diagram_executor import convert_graph_to_visualization_data
        return convert_graph_to_visualization_data(self.executor.eval_graph)