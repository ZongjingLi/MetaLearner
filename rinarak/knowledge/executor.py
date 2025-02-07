#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : central_executor.py
# Author : Yiqi Sun
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Distributed under terms of the MIT license.
from functools import reduce
import itertools
import re
import copy
import random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import build_box_registry
from .entailment import build_entailment
from .predicates import PredicateFilter
from rinarak.utils import freeze
from rinarak.utils.tensor import logit, expat
from rinarak.types import baseType, arrow
from rinarak.program import Primitive, Program, GlobalContext
from rinarak.dsl.logic_types import boolean
from rinarak.algs.search.heuristic_search import run_heuristic_search

__all__ = [
    'QuantizeTensorState',
    'CentralExecutor',
    'UnknownArgumentError',
    'UnknownConceptError'
]

# Constants
DEFAULT_CONCEPT_TYPE = "cone"
DEFAULT_CONCEPT_DIM = 100
MAX_FLATTEN_ITERATIONS = int(1e6)

@dataclass
class QuantizeTensorState:
    """Represents a quantized tensor state for search operations.
    
    Attributes:
        state: Dictionary containing the quantized state representation
    """
    state: Dict[str, Any]


class ExpressionParser:
    """Utility class for parsing and manipulating expressions."""
    
    @staticmethod
    def extract_components(expression: str) -> List[str]:
        """Get content inside parentheses."""
        # Matches content between parentheses: (content)
        return re.findall(r'\((.*?)\)', expression)

    @staticmethod
    def find_token_position(text: str, token: str) -> int:
        """Find token not followed by hyphen."""
        # Matches token not followed by hyphen
        match = re.search(rf'{re.escape(token)}(?!-)', text)
        return match.start() if match else -1

    @staticmethod
    def extract_parameters(expression: str, token: str) -> Tuple[List[str], int, int]:
        """Extract parameters following a token."""
        # Matches: token(param1)(param2)...
        match = re.search(rf'{re.escape(token)}\s*(\([^)]*\))+', expression)
        if not match:
            return [], -1, -1
            
        params_str = match.group(0)
        return (
            re.findall(r'\((.*?)\)', params_str),  # parameters
            match.start(),                         # start position
            match.end() - 1                        # end position
        )

class TypeParser:
    """Utility class for parsing type specifications."""
    
    @staticmethod
    def parse_type_dim(type_str: str) -> Tuple[List[int], str]:
        """Parse type string to get dimensions and type.
        
        Args:
            type_str: Type specification string
        Returns:
            Tuple of ([dimensions], type_name)
        """
        if type_str in ["float", "boolean"]:
            return [1], type_str
            
        if "vector" in type_str:
            content = type_str[7:-1]
            comma_pos = re.search(r",", content)
            
            value_type = content[:comma_pos.span()[0]]
            dimensions = [int(dim[1:-1]) for dim in content[comma_pos.span()[1]:][1:-1].split(",")]
            return dimensions, value_type
            
        return [1], type_str

def type_dim(type_str : str) : return TypeParser.parse_type_dim(type_str)

class ActionIterator:
    """Iterator for generating possible actions in a given state."""
    
    def __init__(self, actions: Dict, state: QuantizeTensorState, executor: 'CentralExecutor'):
        """Initialize the action iterator.
        
        Args:
            actions: Dictionary of available actions
            state: Current quantized state
            executor: Reference to the central executor
        """
        self.actions = actions
        self.action_names = list(actions.keys())
        self.state = state
        self.executor = executor
        self.apply_sequence = self._generate_action_sequence()
        self.counter = 0

    def _generate_action_sequence(self) -> List[List[Any]]:
        """Generate all possible action sequences."""
        sequences = []
        num_actions = self.state.state["end"].size(0)
        obj_indices = list(range(num_actions))
        
        for action_name in self.action_names:
            params = list(range(len(self.actions[action_name].parameters)))
            for param_idx in combinations(obj_indices, len(params)):
                sequences.append([action_name, list(param_idx)])
        
        return sequences

    def __iter__(self):
        return self
        
    def __next__(self):
        if self.counter >= len(self.apply_sequence):
            raise StopIteration
            
        context = copy.copy(self.state.state)
        action_chosen, params = self.apply_sequence[self.counter]
        
        precond, state = self.executor.apply_action(action_chosen, params, context=context)
        self.counter += 1
        state["executor"] = None
        
        return (action_chosen + str(params), 
                QuantizeTensorState(state=state), 
                -1 * torch.log(precond))

class PredicateEvaluator:
    def __init__(self, state_dict):
        self.state = state_dict
        
    @staticmethod
    def preprocess_state(expr: str, context: Dict[str, Any], executor : 'CentralExecutor') -> Dict[str, torch.Tensor]:
        """Extract and evaluate base predicates using executor with $0 placeholders.
        
        Args:
            expr: Expression to process
            context: Context dictionary
            executor: Executor instance with evaluate method
            
        Returns:
            Dictionary mapping predicates to their tensor values
        """
        state_map = {}
        atomic_predicates = set(re.findall(r'\((\w+)[^()]*\)', expr))
        
        # Get number of objects for sizing tensors
        n_objects = len(context.get('state', [])) or context.get('n_objects', 0)
        if not n_objects:
            raise ValueError("Cannot determine number of objects from context")
            
        # Evaluate each predicate with $0
        for pred in atomic_predicates:
            # Try evaluating nullary predicate

            try:
                result = executor.evaluate(f"({pred})", {0:context})
                
                val = result.get('end', 0.0)
                #print(pred,val)
                state_map[pred] = val
                continue
            except:
                pass
                
            # Try unary predicate with $0
            try:

                result = executor.evaluate(f"({pred} $0)", {0:context})

                val = result.get('end', 0.0)
                state_map[pred] = val
                #print(pred,val)
                continue
            except:
                pass
                
            # Try binary predicate with $0 $0
            try:
                result = executor.evaluate(f"({pred} $0 $0)", {0:context})
                val = result.get('end', 0.0)
                state_map[pred] = val
                #print(pred,val)
            except:
                
                pass

                
        # Add any pre-existing tensor predicates from context
        tensor_preds = {k: v for k, v in context.items() 
                       if isinstance(v, torch.Tensor)}
        state_map.update(tensor_preds)
        
        return state_map
    
    def evaluate(self, expr: str, bindings: Dict[str, int]) -> torch.Tensor:
        """Evaluate logical expression with variable bindings."""
        expr = expr.strip('()')
        
        if expr.startswith('and '):

            conjuncts = re.findall(r'\([^()]+\)', expr)
            if not conjuncts:
                conjuncts = [expr[4:]]

            return torch.prod(torch.stack([self.evaluate(conj, bindings) for conj in conjuncts]))
            
        elif expr.startswith('or '):
            disjuncts = re.findall(r'\([^()]+\)', expr)
            if not disjuncts:
                disjuncts = [expr[3:]]
            return torch.max(torch.stack([self.evaluate(disj, bindings) for disj in disjuncts]))
            
        elif expr.startswith('not '):
            inner = re.search(r'\((.*?)\)', expr)
            if inner:
                inner = inner.group(1)
            else:
                inner = expr[4:]
            return 1 - self.evaluate(inner, bindings)
            
        else:
            parts = expr.split()
            pred_name = parts[0]
            
            args = []
            for arg in parts[1:]:
                if arg.startswith('?'):
                    if arg not in bindings:
                        raise ValueError(f"Variable {arg} not found in bindings {bindings}")
                    args.append(bindings[arg])
                else:
                    args.append(int(arg))
            
            tensor = self.state[pred_name]
            if len(args) == 1:
                return tensor[args[0]]
            elif len(args) == 2:
                return tensor[args[0], args[1]]
            return tensor

class PredicateNetwork(nn.Module):
    """Generic neural network for predicates."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.model(x).sigmoid()

class CentralExecutor(nn.Module):
    """Central executor for handling domain operations and actions."""
    
    def __init__(self, domain, concept_type: str = DEFAULT_CONCEPT_TYPE, 
                 concept_dim: int = DEFAULT_CONCEPT_DIM):
        """Initialize the central executor.
        
        Args:
            domain: Domain specification object
            concept_type: Type of concept representation
            concept_dim: Dimension of concept space
        """
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.domain = domain
        self.concept_registry = build_box_registry(concept_type, concept_dim, 128)
        self.neural_registry = nn.ModuleDict({})
        self.initialize_domain_components()
        
        

    def initialize_domain_components(self):
        """Initialize all domain-specific components and registries.
        
        This method serves as the main initialization pipeline, calling all setup
        methods in the correct order to ensure proper domain initialization.
        """
        self._setup_types()
        self._setup_predicates()
        self._setup_constants()
        self._setup_derived_predicates()
        self._setup_implementations()
        self.quantized = False
        
    def _setup_types(self):
        """Set up domain types and their constraints.
        
        Initializes:
            - Base types from domain specification
            - State dimensions and type
            - Type constraints
            
        Raises:
            AssertionError: If state type is not defined in domain
        """
        self.types = self.domain.types
        assert "state" in self.types, "State type must be defined in domain"
        
        # Initialize base types
        for type_name in self.types:
            baseType(type_name)
            
        # Parse state type and dimensions
        self.state_dim, self.state_type = TypeParser.parse_type_dim(self.types["state"])
        self.state_dim = self.state_dim[-1]
        # Store type constraints
        self.type_constraints = self.domain.type_constraints
        
    def _setup_predicates(self):
        """Set up domain predicates and their properties.
        
        Initializes:
            - Predicate registry
            - Output types for predicates
            - Parameter types for predicates
            - Concept vocabulary
        """
        # Initialize storage structures
        self.predicates = {}
        self.predicate_output_types = {}
        self.predicate_params_types = {}
        self.concept_vocab = []
        
        # Register each predicate from domain
        for pred_name, pred_bind in self.domain.predicates.items():
            self._register_predicate(
                name=pred_bind["name"],
                parameters=pred_bind["parameters"],
                return_type=pred_bind["type"]
            )
            
    def _register_predicate(self, name: str, parameters: List[str], return_type: str):
        """Register a single predicate with its properties.
        
        Args:
            name: Name of the predicate
            parameters: List of parameter specifications
            return_type: Return type of the predicate
            
        Updates:
            - predicate_output_types
            - predicate_params_types
            - predicates
            - concept_vocab
        """
        # Store predicate types
        self.predicate_output_types[name] = return_type
        self.predicate_params_types[name] = [
            param.split("-")[1] if "-" in param else "any"
            for param in parameters
        ]

        # Determine arity and function type
        output_dim,_ = type_dim(return_type)

        output_dim = output_dim[-1]

        arity = len(parameters)
        if arity == 0 or arity == 1:
            function_type = arrow(boolean, boolean)
            input_dim = self.state_dim  # Input shape: (N, d), Output: (N,)

        elif arity == 2:
            function_type = arrow(boolean, boolean, boolean)
            input_dim = 2 * self.state_dim  # Input shape: (N, N, 2d), Output: (N, N)

        else:  # Arity > 2
            function_type = reduce(lambda acc, _: arrow(boolean, acc), range(arity - 1), boolean)
            input_dim = arity * self.state_dim  # (N, N, ..., N, arity*d)


        # Create and store the neural network

        self.neural_registry[name] = PredicateNetwork(input_dim, output_dim)

        # Create lambda functions dynamically based on arity
        predicate_value = self._generate_lambda_function(name, arity)

        # Register predicate by arity
        if arity not in self.predicates:
            self.predicates[arity] = []
            
        self.predicates[arity].append(
            Primitive(
                name=name,
                ty=function_type,
                value=predicate_value
            )
        )

        # Add to concept vocabulary
        self.concept_vocab.append(name)

    def _generate_lambda_function(self, name, arity):
        """Generate a nested lambda function to support multi-argument predicates.

        Args:
            name: Predicate name
            arity: Number of arguments

        Returns:
            A nested lambda function matching the arity.
        """
        if arity == 0 or arity == 1:
            return lambda x: self._execute_predicate(name, x)
        
        # Create nested lambda functions for arity > 1
        return self._create_nested_lambda(name, arity)

    def _create_nested_lambda(self, name, arity):
        """Recursively generates nested lambdas based on arity."""
        if arity == 2:
            return lambda x: lambda y: self._execute_predicate(name, x, y)
        elif arity == 3:
            return lambda x: lambda y: lambda z: self._execute_predicate(name, x, y, z)
        else:
            return reduce(lambda acc, _: lambda *args: acc(lambda x: args + (x,)), range(arity), self._execute_predicate)

    def _execute_predicate(self, name, *args):
        """Execute the predicate by running the corresponding neural network.

        Args:
            name: Predicate name
            args: Tuple containing `executor` key with input states

        Returns:
            Tensor with predicate outputs
        """
        if len(args) == 1:
            input_tensor = args[0]["state"]  # Shape (N, d)

        # Handle binary predicates (N, N, d) → (N, N)
        elif len(args) == 2:

            N = args[0]["state"].shape[0]
            M = args[1]["state"].shape[0]
            #print()
            state1 = args[0]["state"].unsqueeze(1).repeat(1, M, 1)  # (N, 1, d) → (N, N, d)
            state2 = args[1]["state"].unsqueeze(0).repeat(N, 1, 1)  # (1, N, d) → (N, N, d)
            input_tensor = torch.cat([state1, state2], dim=-1)  # (N, N, 2d)

        # Handle ternary predicates (N, N, N, d) → (N, N, N)
        elif len(args) == 3:
            N = args[0]["state"].shape[0]
            M = args[1]["state"].shape[1]
            M = args[2]["state"].shape[2]
            state1 = args[0]["state"].unsqueeze(1).unsqueeze(2).repeat(N, M, K, 1)  # (N, 1, 1, d) → (N, N, N, d)
            state2 = args[1]["state"].unsqueeze(0).unsqueeze(2).repeat(N, M, K, 1)  # (1, N, 1, d) → (N, N, N, d)
            state3 = args[2]["state"].unsqueeze(0).unsqueeze(1).repeat(N, M, K, 1)  # (1, 1, N, d) → (N, N, N, d)
            input_tensor = torch.cat([state1, state2, state3], dim=-1)  # (N, N, N, 3d)

        else:
            raise ValueError(f"Unsupported arity: {len(args)}")
        # Ensure predicate exists
        if name not in self.neural_registry:
            raise ValueError(f"Predicate {name} is not registered with a neural network.")

        # Run the neural network
        nn_model = self.neural_registry[name]

        return {"end":nn_model(input_tensor).squeeze()}  # Ensure correct shape


    def _setup_constants(self):
        """Set up domain constants as nullary predicates.
        
        Processes each constant in the domain and registers it
        as a nullary predicate (predicate with no parameters).
        """
        for const_name in self.domain.constants:
            self._register_constant(const_name)
            
    def _register_constant(self, name: str):
        """Register a single constant as a nullary predicate.
        
        Args:
            name: Name of the constant to register
            
        Updates:
            - predicate_output_types
            - predicate_params_types
            - predicates
        """
        # Register constant types
        self.predicate_output_types[name] = 'all'
        self.predicate_params_types[name] = []
        
        # Create nullary predicates list if needed
        if 0 not in self.predicates:
            self.predicates[0] = []
            
        # Add constant as nullary predicate
        self.predicates[0].append(
            Primitive(
                name=name,
                type_signature=arrow(boolean),
                value={
                    "from": name,
                    "set": "nullary",
                    "end": GlobalContext.get_context().get(name, "nullary")
                }
            )
        )
        
    def _setup_derived_predicates(self):
        """Set up derived predicates from domain specification.
        
        Processes each derived predicate, registering its parameters
        and implementation.
        """
        self.derived = self.domain.derived
        
        for name, derived in self.derived.items():
            params = derived["parameters"]
            
            # Store parameter types
            self.predicate_params_types[name] = [
                param.split("-")[1] if "-" in param else "any"
                for param in params
            ]
            
            # Register by arity
            arity = len(params)
            if arity not in self.predicates:
                self.predicates[arity] = []
                
            # Add derived predicate
            self.predicates[arity].append(
                Primitive(
                    name=name,
                    type_signature=arrow(boolean, boolean),
                    value=name
                )
            )
            
    def _setup_implementations(self):
        """Set up implementation registry for domain operations.
        
        Creates primitives for each implementation in the domain,
        storing them in the implementation registry.
        """
        self.implement_registry = {
            key: Primitive(
                name=key,
                type_signature=arrow(boolean, boolean), # this is a fake type
                value=effect
            )
            for key, effect in self.domain.implementations.items()
        }
    def update_registry(self, new_implementations):
        """Update the registry with new implementations and update Primitive.GLOBALS.
        
        Args:
            new_implementations: Dictionary mapping predicate names to their new Primitive implementations
        """
        # First update the implementation registry
        self.implement_registry.update(new_implementations)
        
        # Then update the Primitive.GLOBALS with new values
        for name, primitive in new_implementations.items():
            if name in Primitive.GLOBALS:
                # Keep the original type but update the value function
                original_primitive = Primitive.GLOBALS[name]

                Primitive.GLOBALS[name] = Primitive(
                    name=name,
                    ty=original_primitive.tp,  # Keep original type
                    value=self._create_value_wrapper(primitive.value)
                )
            else:
                # If it's a new primitive, add it to GLOBALS
                Primitive.GLOBALS[name] = Primitive(
                    name=name,
                    ty=primitive.tp,
                    value=self._create_value_wrapper(primitive.value)
                )

    def _create_value_wrapper(self, implementation_func):
        """Create a wrapper that handles the conversion between primitive call formats.
        
        Args:
            implementation_func: The implementation function from the new registry
            
        Returns:
            A wrapped function that handles the conversion
        """
        def wrapper(*args):
            # Convert args to state format if needed
            converted_args = [
                {"state": arg} if not isinstance(arg, dict) else arg 
                for arg in args
            ]
            
            # Call implementation and get result
            result = implementation_func(*converted_args)
            
            # If result is a function (for curried functions), wrap it too
            if callable(result):
                return self._create_value_wrapper(result)
            
            # Extract the final value from the result dict
            if isinstance(result, dict) and "end" in result:
                return result
                
            return result
            
        return wrapper

    def search_discrete_state(self, state: Dict, goal: str, 
                            max_expansion: int = 10000, 
                            max_depth: int = 10000) -> Tuple:
        """Search for a sequence of actions to reach a goal state.
        
        Args:
            state: Initial state dictionary
            goal: Goal state specification
            max_expansion: Maximum number of state expansions
            max_depth: Maximum search depth
            
        Returns:
            Tuple of (states, actions, costs, number_of_expansions)
        """
        init_state = QuantizeTensorState(state=state)
        
        def goal_check(search_state: QuantizeTensorState) -> bool:
            return self.evaluate(goal, {0: search_state.state})["end"] > 0.0

        def get_priority(x: Any, y: Any) -> float:
            return 1.0 + random.random()

        def state_iterator(state: QuantizeTensorState) -> ActionIterator:
            return ActionIterator(self.actions, state, self)
        
        return run_heuristic_search(
            init_state,
            goal_check,
            get_priority,
            state_iterator,
            False,
            max_expansion,
            max_depth
        )
    
    def evaluate(self, program: str, context: dict, scene: bool = False) -> dict:
        """Evaluate a program string in a given context.
        
        The evaluation pipeline consists of:
        1. Process state observations
        2. Setup global context
        3. Update predicate definitions
        4. Handle scene-level evaluation
        5. Process and evaluate program
        
        Args:
            program: Program string to evaluate
            context: Context dictionary containing state and variables
            scene: Whether this is a scene-level evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        
        # Process and evaluate program
        program = self._process_program(program)
        res = Program.parse(program).evaluate(context)
        return res
    
    def get_predicate_arity(self, predicate_name: str) -> int:
        """Get the arity of a predicate by name.
    
        Args:
            predicate_name: Name of the predicate
        
        Returns:
            Arity of the predicate, or -1 if not found
        """
        for arity, predicates in self.predicates.items():
            if any(pred.name == predicate_name for pred in predicates):
                return arity
        return -1
    
    def _process_program(self, program: str) -> str:
        """Process program string before evaluation.
        
        This includes:
        1. Expanding derived expressions
        2. Converting nullary predicates to unary form
        
        Args:
            program: Original program string
            
        Returns:
            Processed program string
        """
        # First expand derived expressions
        program = self._expand_derived_expressions(program)
        
        # Then convert nullary predicates
        program = self._convert_nullary_predicates(program)
        
        return program
        
    def _convert_nullary_predicates(self, program: str) -> str:
        """Convert nullary predicates to unary form.
        
        Handles:
        - Simple nullary predicates: (hand-free) -> (hand-free $0)
        - Nested expressions
        - Lambda abstractions
        - Function applications
        
        Args:
            program: Program string in lambda calculus format
            
        Returns:
            Modified program with nullary predicates converted to unary form
        """
        # First get all nullary predicates
        nullary_preds = set()
        if 0 in self.predicates:
            nullary_preds = {pred.name for pred in self.predicates[0]}
        
        # If no nullary predicates, return unchanged
        if not nullary_preds:
            return program
            
        # Helper to process a single expression
        def process_expression(expr: str) -> str:
            # Handle lambda abstractions
            if expr.startswith('lambda'):
                parts = expr.split('.', 1)
                if len(parts) == 2:
                    lambda_head, lambda_body = parts
                    return f"{lambda_head}.{process_expression(lambda_body)}"
            
            # Handle function applications
            for pred in nullary_preds:
                # Look for "(pred)" pattern not followed by any argument
                pattern = rf'\({pred}\)(?!\s*\$)'
                # Replace with "(pred $0)"
                expr = re.sub(pattern, f'({pred} $0)', expr)
                
                # Also handle when pred appears alone (not in application)
                pattern = rf'\b{pred}\b(?!\s*[\$\w\()])'
                expr = re.sub(pattern, f'({pred} $0)', expr)
            
            return expr
            
        # Process the entire program
        processed = program
        prev_processed = None
        
        # Keep processing until no more changes
        while processed != prev_processed:
            prev_processed = processed
            processed = process_expression(processed)
            
        return processed
        
    def _substitute_parameters(self, expr: str, formal_params: List[str], actual_params: List[str]) -> str:
        """Substitute parameters in an expression.
        
        Also ensures that any nullary predicates in the substituted expression
        are properly handled.
        
        Args:
            expr: Expression to substitute in
            formal_params: List of formal parameter names
            actual_params: List of actual parameter values
            
        Returns:
            Expression with parameters substituted
        """
        # First do normal substitution
        result = expr
        for formal, actual in zip(formal_params, actual_params):
            result = result.replace(formal, actual)
            
        # Then ensure any nullary predicates are properly handled
        result = self._convert_nullary_predicates(result)
        
        return result
    
    def _expand_derived_expressions(self, program: str) -> str:
        """Expand all derived expressions in program.
        
        Args:
            program: Program containing derived expressions
            
        Returns:
            Program with derived expressions expanded
        """
        iteration = 0
        
        while any(derive in program for derive in self.derived) and iteration < MAX_FLATTEN_ITERATIONS:
            last_program = program
            program = self._expand_single_iteration(program)
            
            # Break if no changes
            if program == last_program:
                break
                
            iteration += 1
        
        return program
    
    def _expand_single_iteration(self, program: str) -> str:
        """Perform single iteration of derived expression expansion.
        
        Args:
            program: Current program string
            
        Returns:
            Program with one round of expansions
        """
        for derive_name, derived in self.derived.items():
            # Skip if derive not present
            if f"{derive_name} " not in program:
                continue
            
            # Extract and validate parameters
            actual_params, start, end = ExpressionParser.extract_parameters(program, derive_name)
            if start == -1:
                continue
            
            # Substitute parameters
            expr = self._substitute_parameters(derived["expr"], derived["parameters"], actual_params)
            
            # Update program
            program = f"{program[:start]}{expr}{program[end:]}"
            
        return program
    
    def evaluate_predicates(self, state_dict: Dict[str, torch.Tensor], expr: str, param_bindings: Dict[str, int]) -> torch.Tensor:
        """Evaluate predicates in the expression using PredicateEvaluator."""
        evaluator = PredicateEvaluator(state_dict)
        return evaluator.evaluate(expr, param_bindings)
        
    def apply_action(self, action_name: str, params: List[int], context: Dict[str, Any] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply an action to the current state."""
        if context is None:
            context = {}
            
        action = self.domain.actions[action_name]
        bindings = dict(zip(action.parameters, params))
        
        # Preprocess state to create tensor map
        state_map = PredicateEvaluator.preprocess_state(str(action.precondition), context, self)

        
        # Create evaluator with preprocessed state
        evaluator = PredicateEvaluator(state_map)

        precond_score = evaluator.evaluate(str(action.precondition), bindings)
        print(action.precondition, precond_score)
        
        if torch.all(precond_score < 0.5):
            return precond_score, context
            
        # Create new state and process effects
        new_state = dict(context)
        effect_expr = str(action.effect)
        #print("Effect:",effect_expr)
        
        # Parse and apply effects
        if effect_expr.startswith('(and-do'):
            effects = re.findall(r'\(assign [^()]+\)', effect_expr)
            #print(effects)
        else:
            effects = [effect_expr]
            
        n_objects = len(context.get('state', []))
        
        for effect in effects:
            # Extract predicate name, arguments, and value using regex
            match = re.match(r'\(assign \((\w+)(.*?)\) (\w+)\)', effect)
            if match:
                pred_name, args_str, value = match.groups()
                # Get argument indices using bindings
                args = [bindings.get(arg.strip(), int(arg.strip())) 
                       for arg in args_str.split() if arg.strip()]
                
                # Create appropriate tensor based on predicate arity
                if not args:  # Nullary
                    new_state[pred_name] = torch.tensor([float(value == 'true')])
                elif len(args) == 1:  # Unary
                    tensor = torch.zeros(n_objects, dtype=torch.float32)
                    tensor[args[0]] = float(value == 'true')
                    new_state[pred_name] = tensor
                elif len(args) == 2:  # Binary
                    tensor = torch.zeros((n_objects, n_objects), dtype=torch.float32)
                    tensor[args[0], args[1]] = float(value == 'true')
                    new_state[pred_name] = tensor
                    
        return precond_score, new_state
class UnknownArgumentError(Exception):
    """Raised when an unknown argument is encountered."""
    pass

class UnknownConceptError(Exception):
    """Raised when an unknown concept is referenced."""
    pass