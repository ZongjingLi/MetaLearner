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
import numpy as np
from typing import List, Union, Any
from core.metaphors.executor import SearchExecutor, ExecutorGroup, UnificationFailure, value_types
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import  LexiconEntry
from core.grammar.learn import enumerate_search
from helchriss.knowledge.symbolic import Expression
from helchriss.knowledge.executor import CentralExecutor
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import BOOL, FLOAT, INT, AnyType
from anytree import Node, RenderTree


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
    total_norm = 0
    
    for name, param in submodel.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"Param {name} Grad Norm: {param_norm:.6f}")

def check_model_nan_params(model: nn.Module) -> None:
    has_nan = False

    for name, param in model.named_parameters():
        if param.requires_grad:  # 仅检查可训练参数（可选，注释后可检查所有参数）

            nan_mask = torch.isnan(param)
            nan_count = torch.sum(nan_mask).item()
            total_count = param.numel()  # 参数总元素个数
            has_nan_in_param = nan_count > 0

            if has_nan_in_param:
                has_nan = True
                nan_ratio = nan_count / total_count * 100
                print(f"\n参数名称: {name}")
                print(f"  参数形状: {param.shape}")
                print(f"  NaN 元素个数: {nan_count}")
                print(f"  NaN 占比: {nan_ratio:.4f}%")

    if has_nan:
        print("\n" + "=" * 80)
        print("⚠️  检测到模型中存在 NaN 参数！")
        print("=" * 80)
        return True
    else:
        return False
   
class MetaLearner(nn.Module):
    def __init__(self, domains : List[Union[CentralExecutor]], vocab : List[str] = []):
        super().__init__()
        """domain information for execution and save ckpt"""
        self.domain_infos = {}
        self._domain :List[Union[CentralExecutor]]  = domains
        #self.executor : CentralExecutor = RewriteExecutor(ExecutorGroup(domains))
        self.executor : SearchExecutor = SearchExecutor(ExecutorGroup(domains))
        self.executor.refs["executor_parent"] = self # setup the executor parent


        """misc of saveing and formatting"""
        self.name            = "prototype"
        self.gather_format   = self.executor.gather_format
        self.cheat           = False # use the ground-truth to cheat


        """setup the lexicon entries and parser for combinotorical generalization"""
        self._vocab          = vocab
        self.lexicon_entries = {}
        self.parser          = ChartParser(self.lexicon_entries)
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

        self.executor = SearchExecutor(ExecutorGroup(domain_executors))
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

    def associate_word_domains(self, word, domains : List[str]):
        #TODO: enumerate the function compositions in the domain and add them to the word entires.
        return

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
                if execute: result = self.executor.additive_evaluation(expr, grounding)
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
    

    def train(self, dataset: SceneGroundingDataset, epochs: int = 1000, lr = 1e-3, topK = None, unify = False, test_set = None):
        import tqdm
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.executor.supressed = False if unify else True
        
        epoch_bar = tqdm.tqdm(range(epochs), desc="Training epochs", unit="epoch")

        def evaluate_test_set():
            if test_set is None: return -1.0
            #self.eval()  # Set model to eval mode
            test_correct = 0.0
            test_total = 0.0
            with torch.no_grad():  # Disable gradient computation for test
                for idx, sample in tqdm.tqdm(test_set):
                    query = sample["query"]
                    answer = sample["answer"]
                    grounding = sample["grounding"]
                    images = sample["image"]
                    program = sample["program"] if "program" in sample and self.cheat else None

                    if isinstance(answer, bool): answer = Value(BOOL, answer)
                    if isinstance(answer, Value) and isinstance(answer.vtype, str):
                        if answer.vtype == "boolean": answer.vtype = BOOL
                        if answer.vtype == "float": answer.vtype = FLOAT
                        if answer.vtype == "int": answer.vtype = INT

                    try:
                        if program is None:
                            results, probs, programs, _ = self(query, grounding, answer.vtype)
                        else:
                            results = [self.executor.additive_evaluation(program, grounding)]
                            probs = [torch.tensor(1.0)]
                            programs = [program]

                        if not results: continue  # Skip if no parse

                        for i, result in enumerate(results):
                            result, _ = result
                            measure_conf = torch.exp(probs[i])
                            if not isinstance(result, Value):
                                continue

                            if answer.vtype == BOOL:
                                predicted = (result.value >= 0.).item()
                                actual = bool(answer.value)
                                if predicted == actual:
                                    test_correct += 1 * measure_conf
                            elif answer.vtype == FLOAT or answer.vtype == INT:
                                measure_loss = torch.abs(result.value - answer.value)
                                if measure_loss < 0.2:
                                    test_correct += 1 * measure_conf
                            test_total += 1 * measure_conf
                    except:
                        continue  # Skip errors in test set
            #self.train()  # Revert to train mode
            return float(test_correct) / test_total if test_total > 0 else 0.0

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

                images = sample["image"]
                program = sample["program"] if "program" in sample and self.cheat else None

                if isinstance(answer, bool): answer = Value(BOOL, answer)

                if isinstance(answer, (int, np.int64)): answer = Value(INT, answer)

                if isinstance(answer, float): answer = Value(FLOAT, answer)
                    
                if isinstance(answer, Value) and isinstance(answer.vtype, str):
                    if answer.vtype == "boolean": answer.vtype = BOOL
                    if answer.vtype == "float": answer.vtype = FLOAT
                    if answer.vtype == "int": answer.vtype = INT


                try:
                    working_loss = 0.
                    if program is None:
                        results, probs, programs, _ = self(query, grounding, answer.vtype)
                    else:  
                        results     =  [self.executor.additive_evaluation(program, grounding)]
                        probs       =  [torch.tensor(1.0)]
                        programs    =  [program]

                    if not results:
                        print(results)
                        self.executor.logger.warning(f"no parsing found for query:{query}")
                
                    for i, result in enumerate(results):
                        result, internal_loss = result
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

                            measure_loss = torch.nn.functional.mse_loss(result.value, torch.tensor(float(answer.value)) )
                            if measure_loss < 0.2: correct += 1 * measure_conf
                            else:
                                pass
                            
                        #print(measure_loss , internal_loss)
                        loss += measure_conf * (measure_loss + internal_loss)
                        total_count += 1 * measure_conf
                        working_loss += measure_conf * measure_loss 
                
                    batch_loss += working_loss
                    batch_count += 1
                
                    if batch_count == batch_size :#or idx == len(dataset) - 1:
                        batch_loss /= batch_count
                        try:
                            if not unify:

                                optim.zero_grad()
                                batch_loss.backward()
                                #check_gradient_flow(self)
                                optim.step()
                                batch_loss = 0.0
                                batch_count = 0
                                if check_model_nan_params(self):
                                    check_gradient_flow(self)
                        except: raise RuntimeError("No Valid Parse Found.")
                except UnificationFailure as Error:

                        if (Error.left_structure,value_types(Error.right_structure)) not in self.executor.ban_list:
                            
                            self.executor.logger.warning(f"unification failure on : {query} with {Error.left_structure}{value_types(Error.right_structure)}, construct freezed : {self.default_freeze}")
            
            #calculate test accuracy (if test_set exists) and update progress bar
            
            test_acc = evaluate_test_set()
            
            if not unify:
                avg_acc = float(correct) / total_count if total_count > 0 else 0.0
                avg_loss = loss / len(dataset) if len(dataset) > 0 else 0.0
                # Add test_acc to postfix
                #check_gradient_flow(self)
                epoch_bar.set_postfix({"avg_loss": f"{avg_loss.item():.4f}", "avg_acc": f"{avg_acc:.4f}", "test_acc": f"{test_acc:.4f}"})
            else: 
                avg_loss, avg_acc = -1, -1
                # Show test_acc even if unify=True
                epoch_bar.set_postfix({"test_acc": f"{test_acc:.4f}"})
            self.executor.logger.info(f"acc:{avg_acc}, test_acc: {test_acc}")
        return {"loss": avg_loss, "acc": avg_acc, "test_acc": test_acc}  # Add test_acc to return

    def infer_metaphor_expressions(self, meta_exprs: Union[str, Expression,List[Expression], List[str]]):
        if isinstance(meta_exprs, Expression) or isinstance(meta_exprs, str): meta_exprs = [meta_exprs]
        meta_exprs = [Expression.parse_program_string(str(expr)) if not isinstance(expr, Expression) else expr for expr in meta_exprs]
        metaphors = []
        for meta_expr in meta_exprs:
            infers = self.executor.infer_rewrite_expr(meta_expr)
            fixed_infers = self.executor.add_metaphors(infers)

            metaphors.append(fixed_infers)

        return metaphors

    def maximal_parse(self, sentence, tp = AnyType, forced = False):
        ### return a sorted list of tuple [parsed program, logit] of the possible parses

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
    
    """for interact and visualization purpose"""

    def process_query(self, query, grounding):
        from helchriss import stprint
        results = self.executor.evaluate(query, grounding = grounding)
        eval_info = self.executor.eval_info

        eval_info = add_coordinates_to_eval_info(eval_info)
        add_and_center_coordinates(eval_info)

        self.path_trees = eval_info["paths"]
        print("result",results[0])
        return eval_info
    
    def get_edge_paths(self, edge_id):
        """get path tree for a specific edge (with weights)"""
        path_tree = self.path_trees[edge_id.split("_")[0]]

        edges = []
        for u,v,w in path_tree["edges"]:
            edges.append([u,v,{"weight":w}])


        bind = {
            "nodes": path_tree["nodes"],
            "edges": edges
        }
        return bind

import random
from .utils import radial_tree_pos, balanced_tree_pos, hierarchy_pos
import networkx as nx


def add_and_center_coordinates(eval_info):
    """
    1. Add 'coordinate' key to all nodes (tree + paths) (random if missing)
    2. Center the FIRST tree node (super node) at (0, 0)
    3. Adjust all other nodes relative to the root to maintain layout
    
    Args:
        eval_info (dict): Input dict with tree (nodes/edges) and paths (dict of path entries)
    
    Returns:
        dict: Updated eval_info with centered coordinates
    """
    # Helper: Generate random (x,y) coordinates (0-1000, 2 decimal places)
    def _generate_random_coords():
        return (round(random.uniform(200, 300), 2), round(random.uniform(200, 600), 2))

    # --------------------------
    # Step 1: Add coordinates to ALL tree nodes (if missing)
    # --------------------------
    tree_nodes = eval_info.get('tree', {}).get('nodes', [])
    for node in tree_nodes:
        if 'coordinate' not in node:
            node['coordinate'] = _generate_random_coords()

    # --------------------------
    # Step 2: Center FIRST tree node (super node) at (0, 0)
    # --------------------------
    offset_x, offset_y = 0, 0
    if tree_nodes:  # Only if there are tree nodes
        # Get root node (first node = node2 in your example)
        root_node = tree_nodes[-1]
        root_x, root_y = root_node['coordinate']
        
        # Calculate offset to shift root to (0,0)
        offset_x = -root_x + 100
        offset_y = -root_y + 100
        
        # Shift ALL tree nodes by the offset (preserve relative positions)
        for node in tree_nodes:
            curr_x, curr_y = node['coordinate']
            node['coordinate'] = (round(curr_x + offset_x, 2), round(curr_y + offset_y, 2))

    # --------------------------
    # Step 3: Add coordinates to ALL path nodes (if missing) + shift to match root
    # --------------------------
    paths_dict = eval_info.get('paths', {})  # Paths are a dict (not list) in your new data
    for path_key in paths_dict:
        path_data = paths_dict[path_key]
        path_nodes = path_data.get('nodes', [])
        
        # Add missing coordinates + shift by root offset
        for node in path_nodes:
            if 'coordinate' not in node:
                node['coordinate'] = _generate_random_coords()
            
            # Shift path node to align with centered tree
            curr_x, curr_y = node['coordinate']
            node['coordinate'] = (round(curr_x + offset_x, 2), round(curr_y + offset_y, 2))

    return eval_info


def add_coordinates_to_eval_info(eval_info, scale=50):
    """
    Convert eval_info to NetworkX DiGraph, compute radial positions,
    and add coordinates back to original data structure.
    
    Args:
        eval_info: Your custom data structure containing tree and paths
        scale: Scaling factor for radial layout
        
    Returns:
        eval_info with added 'coords' key for each node/vertex
    """
    # 1. Extract tree structure and create DiGraph
    tree_data = eval_info['tree']
    G = nx.DiGraph()
    
    # Add nodes to graph
    for node in tree_data['nodes']:
        G.add_node(node['id'])
    
    # Add edges to graph (preserve weight attribute)
    for edge in tree_data['edges']:
        source, target, attrs = edge
        G.add_edge(source, target, **attrs)
    
    # 2. Compute radial positions for tree nodes
    #tree_pos = radial_tree_pos(G, scale=scale)
    #tree_pos = balanced_tree_pos(G)
    tree_pos = hierarchy_pos(G, width = 550, vert_gap=200)

    # 3. Add coordinates to tree nodes
    for node in tree_data['nodes']:
        node_id = node['id']
        if node_id in tree_pos:
            node['coordinate'] = tree_pos[node_id]  # (x, y) tuple
    
    # 4. Process paths (each path has its own vertices)
    paths_data = eval_info['paths']
    for path_node_id, path_data in paths_data.items():
        # Create subgraph for path vertices
        path_G = nx.DiGraph()
        
        # Add path vertices as nodes
        for vertex in path_data['nodes']:
            path_G.add_node(vertex['id'])
        
        # Add path edges
        for edge in path_data['edges']:
            source, target, weight = edge
            path_G.add_edge(source, target, weight=weight)
        
        if path_data['nodes']:
        # Compute radial positions for path vertices (scale down for paths)

           path_pos = hierarchy_pos(path_G, width = 5000, vert_gap= 350)  # smaller scale for paths
        
        # Add coordinates to path vertices
        for vertex in path_data['nodes']:
            vertex_id = vertex['id']
            if vertex_id in path_pos:
                vertex['coordinate'] = path_pos[vertex_id]
    
    return eval_info