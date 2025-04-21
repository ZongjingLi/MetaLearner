'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-23 00:19:33
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-03-23 00:19:35
 # @Description:
'''
from typing import List, Union, Mapping, Dict, Any
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.knowledge.symbolic import Expression,FunctionApplicationExpression, ConstantExpression, VariableExpression
from helchriss.knowledge.symbolic import LogicalAndExpression, LogicalNotExpression,LogicalOrExpression
from helchriss.knowledge.symbolic import TensorState, concat_states
from helchriss.dsl.dsl_values import Value
from helchriss.domain import Domain

# this is the for type space, basically a wrapper class
from .types import *
from .unification import SparseWeightedGraph
from dataclasses import dataclass
import contextlib


class UnificationFailure(Exception):
    """
    Exception raised when unification between feature structures fails.
    
    This class captures information about why unification failed,
    which structures failed to unify, and at which feature the failure occurred.
    """
    
    def __init__(self, message=None, feature=None, left_structure=None, right_structure=None):
        """
        Initialize a UnificationFailure exception.
        
        Args:
            message (str): Description of why unification failed
            feature (str): The feature name where unification failed
            left_structure (dict): The left feature structure that failed to unify
            right_structure (dict): The right feature structure that failed to unify
        """
        self.feature = feature
        self.left_structure = left_structure
        self.right_structure = right_structure
        
        # Create a detailed message if none provided
        if message is None:
            if feature:
                message = f"Unification failed at feature '{feature}'"
                if left_structure and right_structure:
                    message += f": Cannot unify '{left_structure.get(feature)}' with '{right_structure.get(feature)}'"
            else:
                message = "Unification failed between incompatible structures"
        
        super().__init__(message)
    
    def __str__(self):
        """Return a string representation of the unification failure."""
        result = self.args[0]
        
        if self.left_structure and self.right_structure:
            result += "\nLeft structure: " + str(self.left_structure)
            result += "\nRight structure: " + str(self.right_structure)
            
        return result
    
    def get_conflict_path(self):
        """
        Return the path to the conflicting feature.
        
        Returns:
            str: A dot-notation path to the conflicting feature
        """
        if not self.feature:
            return None
        
        return self.feature
    
    def is_type_mismatch(self):
        """
        Check if the failure was due to type mismatch.
        
        Returns:
            bool: True if the failure was due to type mismatch
        """
        if not (self.feature and self.left_structure and self.right_structure):
            return False
        
        left_val = self.left_structure.get(self.feature)
        right_val = self.right_structure.get(self.feature)
        
        return (left_val is not None and right_val is not None and 
                type(left_val) != type(right_val))

class MetaphorExecutor(nn.Module):
    def __init__(self, domains : List[Union[CentralExecutor, Domain]], concept_dim = 128):
        super().__init__()
        self._gather_format = "{}:{}"
        self.domains = []
        self._types = []
        self._functions = []
        self.executors = nn.ModuleList([])
        
        self.casters = nn.ModuleList([]) # dynamic casting of different types
        self.chainer = SparseWeightedGraph([], []) # perform the chaining of functions

        self._grounding = None # just to save the current grounding
        self.concept_dim = concept_dim

        self.update_domains(domains)
    
    def gather_format(self, name, domain): return self._gather_format.format(name, domain)

    def update_domains(self, domains : List[Union[CentralExecutor, Domain]]):
        for i,item in enumerate(domains):
            if isinstance(item, Domain):
                self.domains.append(item)
                self.executors.append(CentralExecutor(item, concept_dim = self.concept_dim))
            elif isinstance(item, CentralExecutor):
                self.domains.append(item.domain)
                self.executors.append(item)
            else: assert 0, f"invalid input in the input domains item:{i} value:{item}"  
        
        for domain in self.domains:
            domain_name = domain.domain_name

            for type in domain.types:
                tp_signature = {
                    "name" : self.gather_format(type, domain_name),
                    "type_space" : TypeSpaceBase.parse_type(domain.types[type].replace("'",""))
                }
                self._types.append(tp_signature)
            for fun in domain.functions:
                function_name = self.gather_format(fun, domain_name)
                domain_func = domain.functions[fun]
                function_signature = {
                    "name" : function_name,
                    "parameters" : [self.gather_format(arg.split("-")[-1], domain_name) for arg in domain_func["parameters"]],
                    "type" :  self.gather_format(domain_func["type"], domain_name),
                }
                self._functions.append(function_signature)
                self.chainer.add_node(function_name)
                #self.chainer.add_edge(function_name, function_name, 1.)

    @property    
    def types(self): return self._types

    def get_type(self, name : str, domain_name : str = None) -> List[Dict]:
        """note that is the domain_name is None then it means you use the gathered name"""
        if domain_name:  query_name = self.gather_format(name, domain_name)
        else: query_name = name
        types = [tp for tp in self.types if tp["name"] == query_name]

        if len(types) > 1 : assert False, f"amibigous reference of function {query_name}"
        if len(types) == 0: raise Exception(f"no function found for {query_name}")
        return types[0]

    @property
    def functions(self): return self._functions

    def get_function(self, name : str, domain_name : str = None) -> List[Dict]:
        """note that is the domain_name is None then it means you use the gathered name"""
        if domain_name:  query_name = self.gather_format(name, domain_name)
        else: query_name = name
        functions =  [func for func in self.functions if func["name"] == query_name]
        if len(functions) > 1 : assert False, f"amibigous reference of function {query_name}"
        if len(functions) == 0: raise Exception(f"no function found for {query_name}")
        return functions[0]
    
    def add_domain(self, executor : Union[CentralExecutor, Domain]):self.update_domains([executor])

    def get_domain(self, name) -> CentralExecutor:
        for executor in self.executors:
            assert isinstance(executor, CentralExecutor), f"{executor} is not a central executor"
            executor.domain.domain_name
        assert False, f"there is no such domain called {name}"

    def get_executor(self, name : str):
        for executor in self.executors:
            assert isinstance(executor, CentralExecutor), "not a valid executor"
            if executor.domain.domain_name == name:
                return executor

    def add_caster(self, source_type, target_type, source_domain = None, target_domain = None):
        if source_domain is not None: source_type = self.gather_format(source_type, source_domain)
        if target_domain is not None: target_type = self.gather_format(target_type, target_domain)

        source_type_space = self.get_type(source_type)["type_space"]
        target_type_space = self.get_type(target_type)["type_space"]

        self.casters.append(TypeCaster(source_type_space, target_type_space))

    def cast_type(self, value, source_type, target_type, source_domain=None, target_domain=None):
        """
        Cast a value from source_type to target_type using the appropriate TypeCaster.
        Args:
            value: The value to be cast
            source_type: The source type name
            target_type: The target type name
            source_domain: Optional domain name for the source type
            target_domain: Optional domain name for the target type
        
        Returns:
            tuple: (cast_value, confidence) where:
                - cast_value is the value cast to the target type
                - confidence is a float between 0 and 1 indicating the confidence in the cast
            
        Raises:
            ValueError: If no suitable caster is found or if the types are ambiguous
        """
        # Convert type names to fully qualified names if domain is provided
        if source_domain is not None:
            source_type = self.gather_format(source_type, source_domain)
        if target_domain is not None:
            target_type = self.gather_format(target_type, target_domain)
    
        source_type_space = self.get_type(source_type)["type_space"]
    
        target_type_space = self.get_type(target_type)["type_space"]
    
        for caster in self.casters: # locate the caster of the type and make the inference
            if (caster.source_type == source_type_space and 
                caster.target_type == target_type_space):
                cast_value, confidence = caster(value)

                return cast_value, confidence

        raise ValueError(f"No caster found for conversion from {source_type} to {target_type}")
    
    @property
    def grounding(self): return self._grounding # the grounding stored in the current execution

    @contextlib.contextmanager
    def with_grounding(self, grounding : Any):
        """create the evaluation context"""
        old_grounding = self._grounding
        self._grounding = grounding
        try:
            yield
        finally:
            self._grounding = old_grounding
        
    def unify(self, types : List[str], args : List[Value]):
        unify_probs = 1.0 # joint prob of args can be unified
        unify_args : List[Value] = [] # a list of args with transformed values

        for i, arg in enumerate(args):
            assert isinstance(arg, Value), f"arg {i}: {arg} is not a Value class"
            target_type = types[i]
            source_type = arg.vtype
            source_value = arg.value

            if source_type != target_type:
                try:
                    target_type_value, cast_prob = self.cast_type(source_value, source_type, target_type)
                except ValueError as e:
                    raise UnificationFailure(left_structure = source_type, right_structure = target_type)
            else:
                target_type_value, cast_prob = source_value, 1.0
            
            unify_args.append(target_type_value)
            unify_probs = unify_probs * cast_prob

        return unify_args, unify_probs

    def chain_evaluate(self, function : str, args : List[Value], domain : str = None):
		# 1. gather all reachable nodes of the query function (abstractions)
		# 2. calculate the weight of each node is the far-reaching node
		#    the P(v) a node v is the far-reached is P(q->v) x (1-max_wP(v->w))
		#    it interprets as there is a path from q to v and no path from v to
		#    any other node w.
		# 3. for each reachable node, calculate the unfication of args to that
		#    function registry (type unfiy) obtain the cast args and corresponding
		#    probability of unification.
		# 4. obtain the expected output of the unification by evaluate each reachable
		#    function on the cast args, the weight of measure is the normalzied weight
		#    of the P(far-reach)*P(unify)
    
        #1. get a set of function nodes involved
        query_function     =  self.gather_format(function, domain)
        reach_weights      =  self.chainer.find_most_far_reaching_nodes(query_function)
        reachable_nodes    =  reach_weights["reachable_nodes"]
        far_reaching_probs =  reach_weights["far_reaching_scores"]


        #2.for each node, gather the type unification information
        reachable_measures = [] # the measurement tensors derived by each chaining node
        reachable_weights = [] # the probability of the node is the most far reaching node and can be unified.
        for reach_func in reachable_nodes:
            func_name, domain_name = reach_func.split(":")
            param_types = self.get_function(reach_func)["parameters"]
            output_type = self.get_function(reach_func)["type"]

            # unify the arguments and enumerate all possible chains
            unified_args, unify_prob = self.unify(param_types, args)

            domain_executor = self.get_executor(domain_name)
            with domain_executor.with_grounding(self.grounding):
                func  = domain_executor._function_registry[func_name]            
                res = func(*unified_args)

                assert isinstance(res, Value), "node wrong"
                reachable_measures.append(res.value)
                reachable_weights.append(unify_prob * far_reaching_probs[reach_func])

        """find the expected output of the chaining evaluation"""
        reachable_weights = torch.tensor(reachable_weights)
        normalized_weights = reachable_weights / torch.sum(reachable_weights)
        expected_measure = torch.zeros_like(reachable_measures[0])

        for measure, weight in zip(reachable_measures, normalized_weights):
            expected_measure += measure * weight
        return expected_measure

    def parse_expression(self, program_str): return Expression.parse_program_string(program_str)

    def evaluate(self, expression : Union[str, Expression], grounding):
        """this is just an override method"""
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)

        grounding = grounding if self._grounding is not None else grounding

        with self.with_grounding(grounding):
            return self._evaluate(expression)

    def _evaluate(self, expr):
        """tracking of equivalent symbols along the expressions"""
        if isinstance(expr, FunctionApplicationExpression):
            func_name, func_domain = expr.func.name.split(":")
            types = self.get_function(expr.func.name)["parameters"]

            args = [Value(types[i], self._evaluate(arg)) for i,arg in enumerate(expr.args)]
            chain_expecation = self.chain_evaluate(func_name, args, func_domain)
            
            return chain_expecation

        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const
        elif isinstance(expr, VariableExpression):
            return expr.name
        elif isinstance(expr, LogicalAndExpression):
            return expr.operands
        elif isinstance(expr, ListExpression):
            """f(x)->y, List[f]: List(x) -> List(y))
            List[List[f]]: List[List(x)] -? List[List(y)]
            """
            value_list = []
            func_name = expr.func.name.split(":")
            types = self.get_function(expr.func.name)["parameters"]
            
            return value_list
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')
