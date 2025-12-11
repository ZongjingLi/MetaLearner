import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .embedding  import build_box_registry
from .entailment import build_entailment
from .symbolic import PredicateFilter
from helchriss.utils import freeze
from helchriss.utils.misc import *
from helchriss.utils.tensor import logit, expat
from helchriss.types import baseType, arrow
from helchriss.program import Primitive, Program
from helchriss.dsl.logic_types import boolean
from helchriss.algs.search.heuristic_search import run_heuristic_search
from dataclasses import dataclass
import copy
import re
from itertools import combinations

import random

def split_components(input_string):
    pattern = r'\([^)]*\)'
    return [match.strip('()') for match in re.findall(pattern, input_string)]

class UnknownArgument(Exception):
    def __init__(self):super()

class UnknownConceptError(Exception):
    def __init__(self):super()

@dataclass
class QuantizeTensorState(object):
      state: dict

def find_first_token(str, tok):
    start_index = 0
    while True:
        # Find the index of the next occurrence of the token
        next_index = str.find(tok, start_index)
        
        # If the token is not found, return -1
        if next_index == -1:
            return -1
        
        # Check if the token is followed by a "-"
        if next_index + len(tok) < len(str) and str[next_index + len(tok)] == "-":
            # If the token is followed by a "-", update the start index and continue searching
            start_index = next_index + 1
            continue
        
        # Otherwise, return the starting index of the token
        return next_index
    
def get_params(ps, token):

    start_loc = find_first_token(ps, token)

    ps = ps[start_loc:]
    count = 0
    outputs = ""
    idx = len(token) + 1
    while count >= 0:
         if ps[idx] == "(": count += 1
         if ps[idx] == ")": count -= 1
         outputs += ps[idx]
         idx += 1
    outputs = outputs[:-1]
    end_loc = idx + start_loc - 1
    components = ["({})".format(comp) for comp in split_components(outputs)]
    if len(components) == 0: components = [outputs]
    return components, start_loc, end_loc

def type_dim(rtype):
    if rtype in ["float", "boolean"]:
        return [1], rtype
    if "vector" in rtype:
        content = rtype[7:-1]
        coma = re.search(r",",content)

        vtype = content[:coma.span()[0]]
        vsize = [int(dim[1:-1]) for dim in content[coma.span()[1]:][1:-1].split(",")]
        return vsize, vtype
    else:
        print(f"unknown state type :{rtype}")
        return [1], rtype


class CentralExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, domain, concept_type = "cone", concept_dim = 100):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.domain = domain
        BIG_NUMBER = 100
        entries = 128

        self.entailment = build_entailment(concept_type, concept_dim)
        self.concept_registry = build_box_registry(concept_type, concept_dim, entries)

        # [Types]
        self.types = domain.types
        assert "state" in domain.types,domain.types
        for type_name in domain.types:
            baseType(type_name)
        self.state_dim, self.state_type = type_dim(domain.types["state"])

        # [Predicate Type Constraints]
        self.type_constraints = domain.type_constraints


        # [Predicates]
        self.predicates = {}
        self.predicate_output_types = {}
        self.predicate_params_types = {}
        for predicate in domain.predicates:
            predicate_bind = domain.predicates[predicate]
            predicate_name = predicate_bind["name"]
            params = predicate_bind["parameters"]
            rtype = predicate_bind["type"]
            
            """add the type annotation to all the predicates in the predicate section"""
            self.predicate_output_types[predicate_name] = rtype
            self.predicate_params_types[predicate_name] = [param.split("-")[1] if "-" in param else "any" for param in params]

            # check the arity of the predicate
            arity = len(params)
            if arity not in self.predicates:
                self.predicates[arity] = []
            
            #predicate_imp = PredicateFilter(predicate_name,arity)
            self.predicates[arity].append(Primitive(predicate_name,arrow(boolean, boolean),
            lambda x: {**x,
                   "from": predicate_name, 
                   "set":x["end"], 
                   "end": x[predicate_name] if predicate_name in x else x["state"]}
                )
            )
        # [Derived]
        self.derived = domain.derived
        for name in self.derived:
            params = self.derived[name]["parameters"]
            self.predicate_params_types[name] = [param.split("-")[1] if "-" in param else "any" for param in params]

            # check the arity of the predicate
            arity = len(params)
            if arity not in self.predicates:
                self.predicates[arity] = []
            
            #predicate_imp = PredicateFilter(predicate_name,arity)
            self.predicates[arity].append(Primitive(name,arrow(boolean, boolean), name))

        # [Actions]
        self.actions = domain.actions

        # [Word Vocab]
        #self.relation_encoder = nn.Linear(config.object_dim * 2, config.object_dim)

        self.concept_vocab = []
        for arity in self.predicates:
            for predicate in self.predicates[arity]:
                self.concept_vocab.append(predicate.name)

        """Neuro Component Implementation Registry"""
        self.implement_registry = {}
        for implement_key in domain.implementations:
            
            effect = domain.implementations[implement_key]
            self.implement_registry[implement_key] = Primitive(implement_key,arrow(boolean,boolean),effect)

        # copy the implementations from the registry

        # args during the execution
        self.kwargs = None 

        self.effective_level = BIG_NUMBER

        self.quantized = False
        """Embedding of predicates and actions, implemented using a simple embedding module"""
        self.predicate_embeddings = nn.Embedding(entries,2)
    
    def embeddings(self, arity):
        """return a tuple of predicate names and corresponding vectors"""
        names = [str(name) for name in self.predicates[arity]]
        embs = [self.get_predicate_embedding(name) for name in names]
        return names, torch.cat(embs, dim = 0)
    
    def get_predicate_embedding(self, name):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        predicate_index = self.concept_vocab.index(name)
        idx = torch.tensor(predicate_index).unsqueeze(0).to(device)
        return self.predicate_embeddings(idx)

    def check_implementation(self):
        warning = False
        for key in self.implement_registry:
            func_call = self.implement_registry[key]
            if func_call is None:warning = True
        if warning:
            print("Warning: exists predicates not implemented.")
            return False
    
    def redefine_predicate(self, name, func):
        for predicate in Primitive.GLOBALS:
            if predicate== name:
                Primitive.GLOBALS[name].value = func
        return True
 
    def evaluate(self, program, context):
        """program as a string to evaluate under the context
        Args:
            program: a string representing the expression for evaluation
            context: the execution context with predicates and executor
        Return:
            precond: a probability of this action is successfully activated.
            parameters changed
        """
        BIG_NUM = 1e6
        flat_string = program
        flag = True in [derive in flat_string for derive in self.derived]
        itr = 0
        """Replace all the derived expression in the program with primitives, that means recusion are not allowed"""
        import time
        start_time = time.time()
        last_string = flat_string
        while flag and itr < BIG_NUM:
            itr += 1
            for derive_name in self.derived:
                if not f"{derive_name} " in flat_string: continue
                formal_params = self.derived[derive_name]["parameters"]
                actual_params, start, end = get_params(flat_string, derive_name)

                """replace derived expressions with the primtives"""
                prefix = flat_string[:start];suffix = flat_string[end:]
                flat_string = "{}{}{}".format(prefix,self.derived[derive_name]["expr"],suffix)

                for i,p in enumerate(formal_params):flat_string = flat_string.replace(p.split("-")[0], actual_params[i])

            
            """until there are no more derived expression in the program"""
            flag = last_string != flat_string
            last_string = flat_string
        end_time = time.time()
        #print("time consumed by translate: {:.5f}".format(end_time - start_time))
        program = Program.parse(flat_string)

        outputs = program.evaluate(context)
        return outputs

    def symbolic_planner(self, start_state, goal_condition):
        pass
    
    def visualize(self, x, fname): return x
    
    def apply_action(self, action_name, params, context):
        """Apply the action with parameters on the given context
        Args:
            action_name: the name of action to apply
            params: a set of integers represent the index of objects in the scene
            context: given all the observable in a diction
        """

        context = copy.copy(context)
        assert action_name in self.actions
        action = self.actions[action_name] # assert the action must be in the action registry

        """Replace the formal parameters in the predcondition into lambda form"""
        formal_params = [p.split("-")[0] for p in action.parameters]
        
        num_objects = context["end"].size(0)

        context_params = {}
        for i,idx in enumerate(params):
            obj_mask = torch.zeros([num_objects])
            obj_mask[idx] = 1.0
            obj_mask = logit(obj_mask)
            context_param = {**context}
            context_param["end"] = context["end"]
            for key in context_param:
                if isinstance(context_param[key], torch.Tensor):
                    context_param[key] = context_param[key][idx]
            context_param["idx"] = idx
            context_params[i] = context_param#{**context, "end":idx}

        # handle the replacements of precondition and effects
        precond_expr = str(action.precondition)
        for i,formal_param in enumerate(formal_params):precond_expr = precond_expr.replace(formal_param, f"${i}")
        effect_expr = str(action.effect)
        for i,formal_param in enumerate(formal_params): effect_expr = effect_expr.replace(formal_param, f"${i}")

        """Evaluate the probabilitstic precondition (not quantized)"""
        precond = self.evaluate(precond_expr,context_params)["end"].reshape([-1])
        #print(precond_expr)
        #print(effect_expr)

        assert precond.shape == torch.Size([1]),print(precond.shape)
        if self.quantized: precond = precond > 0.0 
        else: precond = precond.sigmoid()

        """Evaluate the expressions"""
        effect_output = self.evaluate(effect_expr, context_params)
        if not isinstance(effect_output["end"], list):
            return -1 
        
        output_context = {**context}
        for assign in effect_output["end"]:
            #print(assign)
            condition = torch.min(assign["c"].sigmoid(), precond)
  
            apply_predicate = assign["to"] # name of predicate
            apply_index = assign["x"] # remind that x is the index
            source_value = assign["v"] # value to assign to x


            assign_mask = torch.zeros_like(output_context[apply_predicate]).to(self.device)

            if not isinstance(apply_index,list): apply_index = [apply_index]
            apply_index = list(torch.tensor(apply_index))
            
        
            assign_mask[apply_index] = 1.0

            output_context[apply_predicate] = \
            output_context[apply_predicate] * (1 - condition * assign_mask) + (condition * assign_mask) * source_value
        return precond, output_context

    def get_implementation(self, func_name):
        func = self.implement_registry[func_name]
        return func

    
    def get_type(self, concept):
        concept = str(concept)
        for key in self.type_constraints:
            if concept in self.type_constraints[key]: return key
        return False
    
    def build_relations(self, scene):
        end = scene["end"]
        features = scene["features"]
        N, D = features.shape
        cat_features = torch.cat([expat(features,0,N),expat(features,1,N)], dim = -1)
        relations = self.relation_encoder(cat_features)
        return relations
    
    def all_embeddings(self):
        return self.concept_vocab, [self.get_concept_embedding(emb) for emb in self.concept_vocab]

    def get_concept_embedding(self,concept):
        concept = str(concept)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        concept_index = self.concept_vocab.index(concept)
        idx = torch.tensor(concept_index).unsqueeze(0).to(device)

        return self.concept_registry(idx)
    
    def entail(self, feature, key): 
        if len(feature.shape) == 1: feature = feature.unsqueeze(0)
        return torch.einsum("nd,kd->n", feature, self.get_concept_embedding(key))

    
    def search_discrete_state(self, state, goal, max_expansion = 10000, max_depth = 10000):
        init_state = QuantizeTensorState(state = state)

        class ActionIterator:
            def __init__(self, actions, state, executor):
                self.actions = actions
                self.action_names = list(actions.keys())
                self.state = state
                self.executor = executor

                self.apply_sequence = []

                num_actions = self.state.state["end"].size(0)
                obj_indices = list(range(num_actions))
                for action_name in self.action_names:
                    params = list(range(len(self.actions[action_name].parameters)))
                    
                    for param_idx in combinations(obj_indices, len(params)):
                        #if action_name == "spreader" and 0 in param_idx and 3 in param_idx: print("GOOD:",action_name, list(param_idx))
                        #print(action_name, list(param_idx))
                        self.apply_sequence.append([
                            action_name, list(param_idx)
                        ])
                self.counter = 0

            def __iter__(self):
                return self
            
            def __next__(self):
                
                if self.counter >= len(self.apply_sequence):raise StopIteration
                context = copy.copy(self.state.state)
                
                action_chosen, params = self.apply_sequence[self.counter]
                #if action_chosen == "spreader" and 0 in params and 3 in params:print(action_chosen+str(params),context["red"] > 0)

                precond, state = self.executor.apply_action(action_chosen, params, context = context)
                
                #if action_chosen == "spreader" and 0 in params and 3 in params:print(state["red"] > 0)
                
                self.counter += 1
                state["executor"] = None

                return (action_chosen+str(params), QuantizeTensorState(state=state), -1 * torch.log(precond))
        
        def goal_check(searchState):
            return self.evaluate(goal,{0 :searchState.state})["end"] > 0.0


        def get_priority(x, y): return 1.0 + random.random()

        def state_iterator(state: QuantizeTensorState):
            actions = self.actions
            return ActionIterator(actions, state, self)
        
        states, actions, costs, nr_expansions = run_heuristic_search(
            init_state,
            goal_check,
            get_priority,
            state_iterator,
            False,
            max_expansion,
            max_depth
            )
        
        return states, actions, costs, nr_expansions
    

'''

class RewriteExecutor(CentralExecutor):
    """this is some kind of wrap out of an executor, equipped with a reductive graph"""
    def __init__(self, executor):
        super().__init__(None)
        self.base_executor : CentralExecutor = executor
        self.rewriter : NeuralRewriter = NeuralRewriter() # use this structur to store the caster and hierarchies
        self.inferer  = RuleBasedTransformInferer()
        self._gather_format = "{}:{}"

        """maintain the evaluation graph just to visualize and sanity check"""
        self.eval_graph : nx.DiGraph = nx.DiGraph()
        self.node_count = {}
        self.record = 1

    def init_graph(self):
        self.eval_graph = nx.DiGraph()
        self.node_count = {}
        self.record = 1

    def save_ckpt(self, ckpt_path = "tmp.ckpt"):
        self.base_executor.save_ckpt(ckpt_path)
        self.rewriter.save_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path = "tmp.ckpt"):
        self.base_executor.load_ckpt(ckpt_path)
        self.rewriter.load_ckpt(ckpt_path)
        return self

    def gather_format(self, name, domain): return self._gather_format.format(name, domain)
    
    @staticmethod
    def format(function : str): return function.split(":")[0]

    def infer_rewrite_expr(self, expr : Expression, reducer = None) -> List[Tuple[str, List[TypeBase], List[TypeBase]]]:
        """given an expression, use the unifer to infer if there exist casting of types or change in the local frames"""
        metaphor_exprs = []
        def dfs(expr : Expression):
            if isinstance(expr, FunctionApplicationExpression):
                func_name = expr.func.name

                source_arg_types : List[TypeBase] = [dfs(arg)[0]        for arg in expr.args] # A List of Types

                target_signatures = self.base_executor.function_signature(func_name)

                min_mismatch = len(source_arg_types) + 1
                best_matched = None # find the best matched function signature
                assert len(target_signatures) != 0,f"did not find any function {func_name}"

                for hyp_sign in target_signatures:

                    arg_types, out_type = hyp_sign
                    
                    mismatch_count = 0
                    for i,tp in enumerate(arg_types):
                        if source_arg_types[i].typename != arg_types[i].typename:
                            mismatch_count += 1
                    if mismatch_count < min_mismatch:
                        min_mismatch = mismatch_count
                        best_matched = hyp_sign


                if not min_mismatch == 0:
    
                    metaphor_exprs.append([func_name, best_matched[0], source_arg_types, best_matched[1]])
                output_type = best_matched[1]
    
                #print(output_type.alias, output_type.typename)
                return [output_type, best_matched[0]]
            
            else: raise NotImplementedError(f"did not write how to infer from {expr}")
        dfs(expr)

        return metaphor_exprs
    
    def add_metaphors(self, metaphors : List[Tuple[str, List[TypeBase], List[TypeBase]]], caster = None):
        if not isinstance(metaphors, List) : metaphors = [metaphors]
        output_metaphors = []
        for metaphor in metaphors:
            target_func, target_types, source_types, out_type = metaphor

            input_type  = source_types  #(y) self.function_input_type(*reduce_func.split(":"))      # actual input type for the function
            expect_type = target_types  #(x) expect input type for the function
            output_type = out_type      #(o) the output type for the target function

            ### 1) create the type casting rewrite rule and add a NeuralNet to fill the hole
            filler = fill_hole(input_type, output_type)
            self.base_executor.register_function(target_func, input_type, output_type, filler)
            output_metaphors.append(metaphor) ### add the extention of fill-hole
            
            ### 2) create the local frame that gathers other `source` functions to the `target` function

            if caster is None: caster = self.inferer.infer_caster(input_type, expect_type)
            rewrite_frame : LocalFrame = LocalFrame(target_func, expect_type, input_type, caster)

            reduce_hypothesis = self.base_executor.gather_functions(input_type, output_type)
            for reduce_func in reduce_hypothesis:

                rewrite_frame.add_source_caster(reduce_func, 0.0) # init the reduction `g`->`f` weight logits 0.0
                output_metaphors.append([reduce_func, target_types, source_types, out_type])

            hash_frame = target_func + str(hash((tuple(input_type) + tuple(expect_type))))
            
            self.rewriter.add_frame(hash_frame, rewrite_frame) # multiple frame lead to the same procedure

        return output_metaphors
    
    @property
    def types(self): return self.base_executor.types

    @property
    def functions(self): return self.base_executor.functions

    def function_out_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")

        return self.base_executor.function_out_type(func_name, domain)

    def function_input_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")
        return self.base_executor.function_input_type(func_name, domain)

    def display(self, fname=None):
        """Display the computational graph with proper visualization of nodes and edges."""
        import matplotlib.pyplot as plt
        import networkx as nx
        from networkx import spring_layout
        import numpy as np
        
        G = check_graph_node_names(self.eval_graph)
        H = G.copy()
        for n, d in H.nodes(data=True):
            # Keep only basic attributes for layout
            attrs_to_remove = ['inputs', 'output', 'args', 'weight']
            for attr in attrs_to_remove:
                d.pop(attr, None)
        
        for u, v, d in H.edges(data=True):
            d.pop('output', None)

    
        #from networkx.drawing.nx_agraph import graphviz_layout
        pos = balanced_tree_pos(H)

        #pos = hierarchy_pos(H)
        #pos = radial_tree_pos(H)
        #pos = nx.spring_layout(H, k=14, iterations=150, seed=42)
        #pos = graphviz_layout(G, prog='dot')  # Top-down tree
        #pos = graphviz_layout(G, prog='twopi')  # Radial tree
        edge_colors = []
        for u, v, d in G.edges(data=True):
            color = d.get('color', '#cccccc')
            edge_colors.append(color)

        
        node_colors = []
        node_weights = []
        
        for node, data in G.nodes(data=True):
            # Handle node colors
            color = data.get('color', '#1f77b4')  # default blue
            node_colors.append(color)
            
            # Handle node weights (for alpha transparency)
            weight = data.get('weight', 1.0)
            try:
                if hasattr(weight, 'item'):  # torch tensor
                    weight_val = float(weight.item())
                else:
                    weight_val = float(weight)
                # Normalize weight to [0.3, 1.0] range for visibility
                weight_val = max(0.01, min(1.0, abs(weight_val)))
            except (ValueError, TypeError):
                weight_val = 1.0
            node_weights.append(weight_val)
        
        # Create the plot
        plt.figure("Rewrite Computational Graph", figsize=(12, 8))

        

        # Draw nodes with individual colors and transparency
        from collections import OrderedDict
        pos= OrderedDict((node, pos[node]) for node in G.nodes)
        for i, (node, (x, y)) in enumerate(pos.items()):
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[node], 
                node_color=[node_colors[i]], 
                node_size=2000, 
                alpha=  node_weights[i],
                edgecolors='black',
                linewidths=1,
            )

        nx.draw_networkx_edges(
            G, pos, 
            edge_color=edge_colors, 
            arrows=True, 
            arrowsize=15, 
            arrowstyle='-|>',
            width=2,
            alpha=0.7
        )
        
        # Add node labels and information
        for i, (node, (x, y)) in enumerate(pos.items()):
            data = G.nodes[node]
            
            # Node weight label (top)
            weight = data.get('weight', 'N/A')
            try:
                if hasattr(weight, 'item'):
                    weight_str = f"{float(weight.item()):.3f}"
                else:
                    weight_str = f"{float(weight):.3f}"
            except (ValueError, TypeError):
                weight_str = str(weight)
            
            plt.text(x, y + 0.15, weight_str, 
                    fontsize=8, color='white', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
            
            # Node name (center)
            node_text_color = data.get("text_color", "black")
            # Truncate long node names
            display_name = node.split("&")[0]#node if len(str(node)) < 20 else str(node)[:17] + "..."
            plt.text(x, y, display_name, 
                    fontsize=7, color=node_text_color, weight='bold',
                    ha='center', va='center')
            
            # Output information (bottom)
            if 'output' in data and data['output'] is not None:
                output = data['output']
                if 1:
                    # Handle output value
                    if hasattr(output, 'value'):
                        if hasattr(output.value, 'item'):  # torch tensor
                            if sum(list(output.value.shape)) < 9:
                                val_str = f"{output.value}"
                            else:
                                from helchriss.utils import stprint_str
                                val_str = f"{output.value.shape}"
                        else: val_str = str(output.value)
                    else: val_str = str(output)
                    
                    # Handle output type
                    if hasattr(output, 'vtype'):
                        if hasattr(output.vtype, 'alias'):
                            type_str = str(output.vtype)
                        else:
                            type_str = str(output.vtype)
                    else:
                        type_str = "Unknown"
                    
                
                    out_label = f"V: {val_str}\nT: {type_str}"
                    
                
                plt.text(x, y - 0.2, out_label, 
                        fontsize=7, color=node_text_color,
                        ha='center', va='center')
        
        # Add edge labels
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            label_parts = []
            
            # Add weight if available
            if 'weight' in d:
                try:
                    weight = d['weight']
                    if hasattr(weight, 'item'):
                        weight_val = float(weight.item())
                    else:
                        weight_val = float(weight)
                    label_parts.append(f"W: {weight_val:.3f}")
                except (ValueError, TypeError):
                    label_parts.append(f"W: {d['weight']}")
            
            # Add output info if available
            if 'output' in d and d['output'] is not None:
                try:
                    output = d['output']
                    if hasattr(output, 'value'):
                        if hasattr(output.value, 'item'):
                            val_str = f"{float(output.value.item()):.3f}"
                        else:
                            val_str = str(output.value)[:10]
                        label_parts.append(f"V: {val_str}")
                except:
                    pass
            
            if label_parts:
                edge_labels[(u, v)] = '\n'.join(label_parts)
        
        # Draw edge labels with better positioning
        if 1 and edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, 
                edge_labels=edge_labels, 
                font_size=7,
                font_color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
        
        # Style the plot
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        
        # Remove axes
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False, 
                       labelbottom=False, bottom=False, top=False)
        
        plt.title("Computational Graph Visualization", fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save if filename provided
        if fname is not None:
            plt.savefig(f"{fname}.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            #print(f"Graph saved as {fname}.png")
        
        plt.show()
        return G, pos  # Return graph and positions for further use if needed

    def evaluate(self, expression, grounding):
        self.init_graph()
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)
        #print(expression)
        grounding = grounding if self._grounding is not None else grounding

        with self.with_grounding(grounding):
            outputs, out_name = self._evaluate(expression)

            self.eval_graph.add_node("outputs", weight = 1.0, inputs = outputs, color = "#0d0d0d", output = outputs)
            self.eval_graph.add_edge(out_name, "outputs", output = outputs, color = "#0d0d0d")

            return outputs
        
     
    def _evaluate(self, expr : Expression):
        """Internal implementation of the executor. This method will be called by the public method :meth:`execute`.
        This function basically implements a depth-first search on the expression tree.
        Args:
            expr: the expression to execute.

        Returns:
            The result of the execution.
        """

        if isinstance(expr, FunctionApplicationExpression):
            func_name = expr.func.name
            
            # recusive call self.evaluate(arg) to evaluate the args in the subtree
            args : List[Value] = []
            arg_names : List[str] = []
            for arg in expr.args: 
                arg_value, arg_name = self._evaluate(arg) # A List of Values
                args.append(arg_value)
                arg_names.append(arg_name)
            
            arg_types : List[TypeBase] = [arg.vtype for arg in args]
            sign = self.base_executor.signature(func_name, arg_types)


            count_func_sign = self.add_count_function_node(sign)
            self.eval_graph.add_node(count_func_sign)


            for arg_n in arg_names:
                self.eval_graph.add_edge(arg_n, count_func_sign, output = arg_value, color = "#0d0d0d")

            # weight of each rewrite is a basic-rewrite
            rewrite_distr, rewrite_graph = self.rewriter.rewrite_distr(func_name, args)

            self.add_rewrite_subgraph(rewrite_distr, rewrite_graph, sign)
            
            # expected execution over all basic-rewrites
            expect_output = 0.

            for (t_f, t_args, weight) in rewrite_distr:
                measure : Value = self.base_executor.execute(t_f.split("#")[0], t_args, grounding = self.grounding)

                expect_output += weight * measure.value

                ### add the output value for the evaluation graph
                func_sign = t_f
                node_count = self.node_count[func_sign]
                func_count_sign = f"{func_sign}_{node_count}"
                self.eval_graph.nodes[func_count_sign]["output"] = measure          


            return Value(measure.vtype, expect_output), count_func_sign

        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const
        elif isinstance(expr, VariableExpression):
            assert isinstance(expr.name, Value)
            return expr.const
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')
    
    

    def add_count_function_node(self, sign):
        nd = sign#self.format(node)

        if sign in self.node_count: self.node_count[sign] += 1
        else: self.node_count[sign] = 1
        #self.eval_graph.add_node(f"{nd}#{self.node_count[node]}")

        return f"{nd}_{self.node_count[sign]}"

    def add_rewrite_subgraph(self, distr, rewrite_graph, func_sign):
        reduce_funcs = distr
        _, reduce_edges = rewrite_graph
        node_count = self.node_count[func_sign]

        for reduce_func in reduce_funcs:
            node_sign, node_args, node_weight = reduce_func
            if node_sign == func_sign: # attach point is the origion of rewrite
                func_count_sign = f"{func_sign}_{node_count}"

                self.eval_graph.nodes[func_count_sign]["weight"] = node_weight
                self.eval_graph.nodes[func_count_sign]["args"]   = node_args
            
            else: # create new node for args
                node_count_sign = self.add_count_function_node(node_sign)

                self.eval_graph.add_node(node_count_sign, weight = node_weight, args = node_args, color = "#048393", text_color = '#0d0d0d')
            
        for edge in reduce_edges:
            src_node = f"{edge[0]}_{self.node_count[edge[0]]}"
            tgt_node = f"{edge[1]}_{self.node_count[edge[1]]}"

            if src_node != func_sign:
                    self.eval_graph.add_edge(tgt_node, src_node, weight = float(edge[2].detach()), color = "#048393")

'''