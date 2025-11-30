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