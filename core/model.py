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
from core.metaphors.executor import RewriteExecutor, ExecutorGroup
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import  LexiconEntry
from core.grammar.learn import enumerate_search
from helchriss.knowledge.symbolic import Expression
from helchriss.knowledge.executor import CentralExecutor
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import BOOL, FLOAT, INT, AnyType
from anytree import Node, RenderTree

from tqdm import tqdm
from datasets.base_dataset import SceneGroundingDataset

__all__ = ["MetaLearner"]


def write_vocab(vocab, vocab_file = "core_vocab.txt"):
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')


def tree_display(s):
    def parse(s, i=0):
        m = re.match(r'([^:()]+)', s[i:])
        name, i = m.group(1), i + len(m.group(0))
        if i < len(s) and s[i] == ':': i = re.match(r':[^(]*', s[i:]).end() + i
        if i < len(s) and s[i] == '(':
            i, kids = i + 1, []
            while s[i] != ')':
                if s[i] == ',': i += 1
                kid, i = parse(s, i)
                kids.append(kid)
            return (name, kids), i + 1
        return name, i
    
    def build(p, parent=None):
        if isinstance(p, tuple):
            node = Node(p[0], parent=parent)
            [build(kid, node) for kid in p[1]]
            return node
        return Node(p, parent=parent)
    
    return '\n'.join(f"{pre}{node.name}" for pre, _, node in RenderTree(build(parse(s)[0])))

def check_gradient_flow(submodel):
    """检查子模型的梯度流动情况"""
    total_norm = 0
    has_gradient = False
    
    for name, param in submodel.named_parameters():
        if param.grad is not None:
            has_gradient = True
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"参数 {name} 的梯度范数: {param_norm:.6f}")

class MetaLearner(nn.Module):
    def __init__(self, domains : List[Union[CentralExecutor]], vocab : List[str] = []):
        super().__init__()
        self.name = "prototype"
        self._domain :List[Union[CentralExecutor]]  = domains
        self.domain_infos = {}
        self.executor : CentralExecutor = RewriteExecutor(ExecutorGroup(domains))
        
        self._vocab = vocab
        self.lexicon_entries = {}
        self.parser = ChartParser(self.lexicon_entries)

        self.gather_format = self.executor.gather_format
        self.entries_setup() 
        self.cheat = False # use the ground-truth to cheat

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
        self.executor = RewriteExecutor(ExecutorGroup(domain_executors))
        self.executor.load_ckpt(ckpt_path)

        # load the vocabulary
        vocab_path = model_config["vocab"]["path"]
        with open(f"{ckpt_path}/{vocab_path}", 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        self._vocab = vocab
        
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
    def vocab(self):
        words = []
        for word in self._vocab:
            if word not in ["<NULL>", "<START>", "<END>","<UNK>"]: words.append(word)
        return words

    @property
    def domains(self):
        gather_domains = []
        for domain in self._domain:
            if isinstance(domain, CentralExecutor):
                gather_domains.append(domain.domain)
            else: gather_domains.append(domain)
        return gather_domains

    def collect_funcs(self,func_bind, domain):
        fn, dep_func = func_bind
        bind = {
            "name" : fn,
            "parameters" : [arg[1] for arg in dep_func.typed_args], #[self.gather_format(param.split("-")[-1], domain)  for param in func_bind["parameters"]],
            "type" : dep_func.return_type
            }
        return bind
    
    def filter_types(self, domains : Union[str, List[str]]):
        if isinstance(domains, str): domains = [domains]
        domain_types = {}
        for domain in self.domains:
            domain_name = domain.domain_name
            if "Any" in domains or domain_name in domains:
                for tp in domain.type_aliases:
                    domain_types[self.gather_format(tp, domain.domain_name)] =  domain.type_aliases[tp][-1]#.replace("'","")
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
                    domain_functions[self.gather_format(func, domain_name)] =  self.collect_funcs([func,domain.functions[func]], domain_name)
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
    

    def entries_setup(self, related_vocab : List[str] = None, domains : Union[str,List[str]] = "Any" ,depth = 2):
        if related_vocab is None: related_vocab = self.vocab

        related_types = [str(tp) for (alia, tp) in self.filter_types(domains).items()]
        related_funcs = self.filter_functions(domains)
        entries = enumerate_search(related_types, related_funcs, max_depth = depth)

        lexicon_entries = dict()
        for word in related_vocab:
            for idx, (syn_type, program, _) in enumerate(entries):
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

    def forward(self, sentence : str, grounding = None, tp = AnyType, topK = None, execute : bool = True, forced = False):

        parses      =  self.parser.parse(sentence,  topK = topK, forced = forced)
        log_distrs  =  self.parser.get_parse_probability(parses)

        results = []
        raw_logits = []
        programs = []
        for i,parse in enumerate(parses):
            parse_prob = log_distrs[i]
            program = parse.sem_program

            output_type = self.functions[program.func_name]["type"]

            if len(program.lambda_vars) == 0 and tp == output_type:
                expr = Expression.parse_program_string(str(program))
                if execute: result = self.executor.evaluate(expr, grounding)
                else: result = 0.0
                results.append(result)
                raw_logits.append(parse_prob)
                programs.append(program)

            else:
                results.append(None)
                raw_logits.append(parse_prob)
                programs.append(program)
        
        valid_indices  = [i for i, res in enumerate(results) if res is not None]
        valid_results  = [results[i] for i in valid_indices]
        valid_programs = [programs[i] for i in valid_indices]
        valid_parses   = [parses[i] for i in valid_indices]

        if valid_indices:
            valid_logits_tensor = torch.cat([raw_logits[i].reshape([1]) for i in valid_indices], dim = 0)
            normalized_probs = torch.log_softmax(valid_logits_tensor, dim=0)
        else: normalized_probs = []

        return valid_results, normalized_probs, valid_programs, valid_parses
    

    def train(self, dataset: SceneGroundingDataset, epochs: int = 1000, lr = 1e-3, topK = None):
        import tqdm
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        
        epoch_bar = tqdm.tqdm(range(epochs), desc="Training epochs", unit="epoch")
        
        for epoch in epoch_bar:
            loss = 0.0
            correct = 0
            total_count = 0
    
            batch_loss = 0.0
            batch_size = 8
            batch_count = 0
            
            dataset.shuffle()
            #dataset.to_device(device)
            for idx, sample in dataset:
                query = sample["query"]
                answer = sample["answer"]
                grounding = sample["grounding"]
                program = sample["program"] if "program" in sample and self.cheat else None
 
                if isinstance(answer, bool): answer = Value(BOOL, answer)
                    
                if isinstance(answer, Value) and isinstance(answer.vtype, str):
                    if answer.vtype == "boolean": answer.vtype = BOOL
                    if answer.vtype == "float": answer.vtype = FLOAT
                    if answer.vtype == "int": answer.vtype = INT

                if 1:#with torch.autograd.detect_anomaly():
                    working_loss = 0.
                    if program is None:
                        results, probs, programs, _ = self(query, grounding, answer.vtype)
                    else:
                            results     =  [self.executor.evaluate(program, grounding)]
                            probs       =  [torch.tensor(1.0)]
                            programs    =  [program]

                    if not results:
                        print(f"no parsing found for query:{query}")
                
                    for i, result in enumerate(results):
                        measure_conf = torch.exp(probs[i])
                        assert isinstance(result, Value), f"{programs[i]} result is :{result} and not a Value type"
                                                                
                        if self.cheat and program:
                            if str(program) == str(programs[i]):gt_program = 1.0
                            else: gt_program = 0.0

                        if answer.vtype == BOOL:
                            measure_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                result.value.reshape([-1]).clamp(-13.,13.), 
                                torch.tensor(float(answer.value)).reshape([-1])
                            )
                            predicted = (result.value >= 0.).item()
                            actual = bool(answer.value)
                            if predicted == actual: correct += 1 * measure_conf
  
                        elif answer.vtype == FLOAT or answer.vtype == INT:
                            measure_loss = torch.abs(result.value - answer.value)
                            if measure_loss < 0.2: correct += 1 * measure_conf

                        loss += measure_conf * measure_loss
                        total_count += 1 * measure_conf
                        working_loss += measure_conf * measure_loss
                
                    batch_loss += working_loss
                    batch_count += 1

                    #for executor in self.executor.base_executor.executor_group:
                    #    check_gradient_flow(executor)
                
                    if batch_count == batch_size :#or idx == len(dataset) - 1:
                        batch_loss /= batch_count
                        try:
                        
                            optim.zero_grad()
                            batch_loss.backward()
                            optim.step()
                            batch_loss = 0.0
                            batch_count = 0
                        except: raise RuntimeError("No Valid Parse Found.")
            
            avg_acc = float(correct) / total_count
            avg_loss = loss / len(dataset) if len(dataset) > 0 else 0
            epoch_bar.set_postfix({"avg_loss": f"{avg_loss.item():.4f}", "avg_acc": f"{avg_acc:.4f}"})
        
        return {"loss": avg_loss, "acc": avg_acc}


    def infer_metaphor_expressions(self, meta_exprs: Union[str, Expression,List[Expression], List[str]]):
        if isinstance(meta_exprs, Expression) or isinstance(meta_exprs, str): meta_exprs = [meta_exprs]
        meta_exprs = [Expression.parse_program_string(str(expr)) if not isinstance(expr, Expression) else expr for expr in meta_exprs]
        metaphors = []
        for meta_expr in meta_exprs:
            infers = self.executor.infer_rewrite_expr(meta_expr)
            fixed_infers = self.executor.add_metaphors(infers)
            #for inf in fixed_infers:print(inf)
            #print("\n")
            metaphors.append(fixed_infers)

        return metaphors

    def maximal_parse(self, sentence, tp = AnyType, forced = False):
        ### return a sorted list of tuple [parsed program, logit] of the possible parses
        #parses = self.parser.parse(sentence, forced = forced)
        _, _, _, parses = self(sentence, {}, tp, execute = False, forced = forced)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip([p.sem_program for p in parses], distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)

        return sorted_parses

    def parse_display(self, sentence, tp = AnyType,topK = 4, forced = False):
        """ display the topK possible parses of the given sentence """
        sorted_parses = self.maximal_parse(sentence, tp,forced)
        for i, parse in enumerate(sorted_parses[:topK]):
            print(f"{parse[0]}, {float(parse[1].exp()):.2f}")
        print("\n")
        return sorted_parses

    def execute_display(self, sentence, grounding = {}, tp = AnyType, topK = 4, forced = False):
        """ execute the topK maximal parses """
        import tabulate
        data = []
        sorted_parses = self.maximal_parse(sentence, tp, forced)
        for i, parse in enumerate(sorted_parses[:topK]):
            expr = Expression.parse_program_string(str(parse[0]))
            answer = self.executor.evaluate(expr, grounding)
            tree = tree_display(str(parse[0]))
            data.append([tree, float(parse[1].exp()),answer])
        
        headers = ["parse-tree", "weight", "answer"]
        data = sorted(
            data,
            key=lambda x: x[1],  # weight is the 4th element (index 3)
            reverse=True         # descending order
            )
        table = tabulate.tabulate(data, headers = headers, tablefmt = "grid")
        print(table)
        return table

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
        from core.metaphors.executor import convert_graph_to_visualization_data
        return convert_graph_to_visualization_data(self.executor.eval_graph)