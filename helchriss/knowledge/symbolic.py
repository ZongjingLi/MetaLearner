# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:07:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-19 04:42:09
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from helchriss.dklearn.nn import FCBlock
from helchriss.utils.tensor import logit
from helchriss.utils import indent_text
from helchriss.dsl.dsl_values import Value, MultidimensionalArrayInterface, TensorValueDict, QINDEX,\
    StateObjectReference, ListType, StateObjectList, MaskedTensorStorage, TensorValue
from helchriss.dsl.dsl_types import AutoType, ObjectType

from termcolor import colored
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Mapping, List, Iterable, Tuple, Sequence
import re

class DSLExpressionError(Exception):
    pass

"""Allowed expression like List[f](x) this expression suggest such that the f(x) is element wise implementation
List[red](scene()) this suggest that scene is a list of elements and color can red need cast
so the syntatic type will be List[red]()
"""

"""
an expression is the data class of tree with the following nodes.
"""
import re
from typing import List, Union, Any, Optional, Dict

class Expression:
    def evaluate(self, inputs, executor):
        return inputs
    
    def __repr__(self):
        return f"Expression()"
        
    @staticmethod
    def parse_program_string(program: str) -> 'Expression':
        """
        Parse a string representation of a program into an Expression tree.
        Supports logical operators (and, or, not) and function names with domain notation (name::domain).
        
        Args:
            program: The program string to parse
            
        Returns:
            An Expression object representing the parsed program
        """
        # Remove whitespace
        program = program.strip()
        
        if not program:
            return Expression()
        
        # First check for assignment
        if ":=" in program:
            var_name, value_expr = program.split(":=", 1)
            return ValueAssignmentExpression(var_name.strip(), 
                                            Expression.parse_program_string(value_expr.strip()))
        
        # Parse logical expressions
        return Expression._parse_logical_expression(program)
    
    @staticmethod
    def _parse_logical_expression(expr: str) -> 'Expression':
        """Parse expressions with 'or' operators at the top level"""
        if not expr.strip():
            return Expression()
        
        # Split by 'or' but respect parentheses
        or_parts = Expression._split_by_operator(expr, "or")
        
        if len(or_parts) > 1:
            # We have 'or' operators
            operands = [Expression._parse_and_expression(part) for part in or_parts]
            return LogicalOrExpression(operands)
        else:
            # No 'or' operators, try 'and'
            return Expression._parse_and_expression(expr)
    
    @staticmethod
    def _parse_and_expression(expr: str) -> 'Expression':
        """Parse expressions with 'and' operators"""
        if not expr.strip():
            return Expression()
        
        # Split by 'and' but respect parentheses
        and_parts = Expression._split_by_operator(expr, "and")
        
        if len(and_parts) > 1:
            # We have 'and' operators
            operands = [Expression._parse_not_expression(part) for part in and_parts]
            return LogicalAndExpression(operands)
        else:
            # No 'and' operators, try 'not'
            return Expression._parse_not_expression(expr)
    
    @staticmethod
    def _parse_not_expression(expr: str) -> 'Expression':
        """Parse expressions with 'not' operators"""
        expr = expr.strip()
        if not expr:
            return Expression()
        
        # Check for 'not' operator
        if expr.startswith("not "):
            operand = Expression._parse_not_expression(expr[4:].strip())
            return LogicalNotExpression(operand)
        else:
            # No 'not' operator, parse atoms (parenthesized expressions, function calls, etc.)
            return Expression._parse_atom(expr)
    
    @staticmethod
    def _parse_atom(expr: str) -> 'Expression':
        """Parse atomic expressions (parenthesized expressions, function calls, variables, constants)"""
        expr = expr.strip()
        if not expr:
            return Expression()
        
        # Parenthesized expressions
        if expr.startswith("(") and expr.endswith(")"):
            # Check if the outer parentheses are actually enclosing the entire expression
            paren_level = 0
            for i, char in enumerate(expr):
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                    # If we reach level 0 before the last character, these aren't the outer parentheses
                    if paren_level == 0 and i < len(expr) - 1:
                        break
            
            # If the parentheses enclose the entire expression, remove them and parse the content
            if paren_level == 0:
                return Expression._parse_logical_expression(expr[1:-1])
        
        # Function application
        # Modified regex to properly handle identifiers with double colon notation (like point::Path)
        func_match = re.match(r'^([a-zA-Z0-9_:]+)\((.*)\)$', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            
            # Parse arguments
            args = Expression._parse_arguments(args_str)
            return FunctionApplicationExpression(VariableExpression(func_name), args)
        
        # Constants (uppercase identifiers with at least 2 characters)
        # Modified to consider colons as part of the identifier
        if re.match(r'^[A-Z][A-Z0-9_:]*$', expr) and len(expr) >= 2:
            return ConstantExpression("Any", expr)
        
        # Numeric constant
        if re.match(r'^-?\d+(\.\d+)?$', expr):
            return ConstantExpression("Any", float(expr) if '.' in expr else int(expr))
        
        # Variable (can include domain notation)
        return VariableExpression(expr)
    
    @staticmethod
    def _parse_arguments(args_str: str) -> List['Expression']:
        """Parse function arguments, respecting nested parentheses and logical operators"""
        args = []
        if not args_str.strip():
            return args
        
        # Track nested parentheses and quotes to split arguments correctly
        current_arg = ""
        paren_level = 0
        
        for char in args_str:
            if char == ',' and paren_level == 0:
                # End of an argument
                args.append(Expression._parse_logical_expression(current_arg.strip()))
                current_arg = ""
            else:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                current_arg += char
        
        # Add the last argument
        if current_arg:
            args.append(Expression._parse_logical_expression(current_arg.strip()))
        
        return args
    
    @staticmethod
    def _split_by_operator(expr: str, operator: str) -> List[str]:
        """
        Split expression by operator (and, or) while respecting parentheses
        
        Args:
            expr: The expression to split
            operator: The operator to split by ('and' or 'or')
            
        Returns:
            List of subexpressions
        """
        parts = []
        current_part = ""
        paren_level = 0
        word_boundary_pattern = r'(?<!\w)' + re.escape(operator) + r'(?!\w)'
        
        i = 0
        while i < len(expr):
            # Check for operator with word boundaries
            match = re.match(word_boundary_pattern, expr[i:])
            
            if paren_level == 0 and match:
                # Found the operator outside parentheses with proper word boundaries
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
                i += len(operator)
            else:
                if expr[i] == '(':
                    paren_level += 1
                elif expr[i] == ')':
                    paren_level -= 1
                current_part += expr[i]
                i += 1
        
        # Add the last part
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts

class VariableExpression(Expression):
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"VariableExpression({self.name})"

class ConstantExpression(Expression):
    def __init__(self, type_name: str, value: Any):
        self.type_name = type_name
        self.const = Value(type_name,value)
    
    def __repr__(self):
        return f"ConstantExpression({self.type_name}, {self.const})"

class FunctionApplicationExpression(Expression):
    def __init__(self, function: Expression, arguments: List[Expression]):
        self.func = function
        self.args = arguments
    
    def __repr__(self):
        return f"FunctionApplicationExpression({self.func}, {self.args})"

class ValueAssignmentExpression(Expression):
    def __init__(self, variable_name: str, value_expression: Expression):
        self.variable_name = variable_name
        self.value_expression = value_expression
    
    def __repr__(self):
        return f"ValueAssignmentExpression({self.variable_name}, {self.value_expression})"

class LogicalAndExpression(Expression):
    def __init__(self, operands: List[Expression]):
        self.operands = operands
    
    def __repr__(self):
        return f"LogicalAndExpression({self.operands})"

class LogicalOrExpression(Expression):
    def __init__(self, operands: List[Expression]):
        self.operands = operands
    
    def __repr__(self):
        return f"LogicalOrExpression({self.operands})"

class LogicalNotExpression(Expression):
    def __init__(self, operand: Expression):
        self.operand = operand
    
    def __repr__(self):
        return f"LogicalNotExpression({self.operand})"
    
class ListExpression(Expression):
    def __init__(self, function, arg):
        self.function = function
        self.arg = arg

    def __repr__(self):
        return f"List[{self.function}]({self.arg})"


class State:
    """a state that maps a given feature name to a value"""
    def __init__(self):
        """ construct a symbolic or hybrid state
        Args:
            data: a diction that maps the data of each state to the actual value
        """

""""""

class TensorStateBase(State):
    """a state representation maps a given feature name to the tensor feature"""
    def __init__(self, data):
        super(data)
    
    @property
    def batch_dims(self) -> int: raise NotImplementedError()

    @property
    def features(self) -> MultidimensionalArrayInterface: raise NotImplementedError

    def __getitem__(self, name): return self.features[name]

    def clone(self) -> 'TensorStateBase': raise NotImplementedError()
    
    def __str__(self) -> str: raise NotImplementedError()

    def __repr__(self): return self.__str__()

class TensorState(TensorStateBase):
    """A state representation is essentially a mapping from feature names to tensors."""
    def __init__(self, features: Optional[Union[Mapping[str, Any], TensorValueDict]] = None, batch_dims: int = 0, internals: Optional[Dict[str, Any]] = None):
        """Initialize a state.

        Args:
            features: the features of the state.
            batch_dims: the number of batch dimensions.
            internals: the internal state of the state.
        """

        if features is None:
            features = dict()
        if internals is None:
            internals = dict()

        if isinstance(features, TensorValueDict):
            self._features = features
        else:
            self._features = TensorValueDict(features)
        self._batch_dims = batch_dims

        self._internals = dict(internals)

    @property
    def batch_dims(self) -> int:
        """The number of batchified dimensions. For the basic State, it should be 0."""
        return self._batch_dims

    @property
    def features(self) -> TensorValueDict:
        return self._features

    @property
    def internals(self) -> Dict[str, Any]:
        """Additional internal information about the state."""
        return self._internals

    def clone(self) -> 'TensorState':
        return type(self)(features=self._features.clone(), batch_dims=self._batch_dims, internals=self.clone_internals())

    def clone_internals(self):
        """Clone the internals."""
        return self.internals.copy()

    def summary_string(self) -> str:
        """Get a summary string of the state. The main difference between this and __str__ is that this function only formats the shape of intermediate tensors."""
        fmt = f'''{type(self).__name__}{{
  states:
'''
        for p in self.features.all_feature_names:
            feature = self.features[p]
            fmt += f'    {p}: {feature.format(content=False)}\n'
        fmt += self.extra_state_str_after()
        fmt += '}'
        return fmt

    def __str__(self):
        fmt = f'''{type(self).__name__}{{\n'''
        fmt += self.extra_state_str_before()
        fmt += '  states:\n'
        for p in self.features.all_feature_names:
            tensor = self.features[p]
            fmt += f'    - {p}'
            fmt += ': ' + indent_text(str(tensor), level=2).strip() + '\n'
        fmt += self.extra_state_str_after()
        fmt += '}'
        return fmt

    def extra_state_str_before(self) -> str:
        """Extra state string before the features."""
        return ''

    def extra_state_str_after(self) -> str:
        """Extra state string."""
        return ''

def _pad_tensor(tensor: torch.Tensor, target_shape: Iterable[int], dtype: TensorValue, batch_dims: int, constant_value: float = 0.0):
    target_shape = tuple(target_shape)
    paddings = list()
    for size, max_size in zip(tensor.size()[batch_dims:], target_shape):
        paddings.extend((max_size - size, 0))
    if tensor.dim() - batch_dims == len(target_shape):
        pass
    elif tensor.dim() - batch_dims == len(target_shape) + dtype.ndim():
        paddings.extend([0 for _ in range(dtype.ndim() * 2)])
    else:
        raise ValueError('Shape error during tensor padding.')
    paddings.reverse()  # no need to add batch_dims.
    return F.pad(tensor, paddings, "constant", constant_value)
def concat_tvalues(*args: TensorValue):  # will produce a Value with batch_dims == 1, but the input can be either 0-batch or 1-batch.
    assert len(args) > 0
    include_tensor_optimistic_values = any([v.tensor_optimistic_values is not None for v in args])
    include_tensor_quantized_values = any([v.tensor_quantized_values is not None for v in args])

    # Sanity check.
    for value in args[1:]:
        assert value.is_torch_tensor
        assert value.dtype == args[0].dtype
        assert value.batch_variables == args[0].batch_variables
        if include_tensor_optimistic_values:
            pass  # we have default behavior for None.
        if include_tensor_quantized_values:
            assert value.tensor_quantized_values is not None

    device = args[0].tensor.device

    # Collect all tensors.
    all_tensor = [v.tensor for v in args]
    all_tensor_mask = [v.tensor_mask for v in args]
    all_tensor_optimistic_values = [v.tensor_optimistic_values for v in args]
    all_tensor_quantized_values = [v.tensor_quantized_values for v in args]

    target_shape = tuple([
        max([v.get_variable_size(i) for v in args])
        for i in range(args[0].nr_variables)
    ])
    for i in range(len(args)):
        tensor, tensor_mask, tensor_optimistic_values, tensor_quantized_values = all_tensor[i], all_tensor_mask[i], all_tensor_optimistic_values[i], all_tensor_quantized_values[i]
        all_tensor[i] = _pad_tensor(tensor, target_shape, args[i].dtype, args[i].batch_dims)

        if tensor_mask is None:
            tensor_mask = torch.ones(target_shape, dtype=torch.bool, device=device)
        else:
            tensor_mask = _pad_tensor(tensor_mask, target_shape, args[i].dtype, args[i].batch_dims)
        all_tensor_mask[i] = tensor_mask

        if include_tensor_optimistic_values:
            if tensor_optimistic_values is None:
                tensor_optimistic_values = torch.zeros(target_shape, dtype=torch.int64, device=device)
            else:
                tensor_optimistic_values = _pad_tensor(tensor_optimistic_values, target_shape, args[i].dtype, args[i].batch_dims)
            all_tensor_optimistic_values[i] = tensor_optimistic_values

        if include_tensor_quantized_values:
            tensor_quantized_values = _pad_tensor(tensor_quantized_values, target_shape, args[i].dtype, args[i].batch_dims)
            all_tensor_quantized_values[i] = tensor_quantized_values

        if args[0].batch_dims == 0:
            all_tensor[i] = all_tensor[i].unsqueeze(0)
            all_tensor_mask[i] = all_tensor_mask[i].unsqueeze(0)
            all_tensor_optimistic_values[i] = all_tensor_optimistic_values[i].unsqueeze(0) if all_tensor_optimistic_values[i] is not None else None
            all_tensor_quantized_values[i] = all_tensor_quantized_values[i].unsqueeze(0) if all_tensor_quantized_values[i] is not None else None
        else:
            assert args[0].batch_dims == 1

    masked_tensor_storage = MaskedTensorStorage(
        torch.cat(all_tensor, dim=0),
        torch.cat(all_tensor_mask, dim=0),
        torch.cat(all_tensor_optimistic_values, dim=0) if include_tensor_optimistic_values else None,
        torch.cat(all_tensor_quantized_values, dim=0) if include_tensor_quantized_values else None
    )
    return TensorValue(args[0].dtype, args[0].batch_variables, masked_tensor_storage, batch_dims=1)


ObjectNameArgument = Union[Iterable[str], Mapping[str, ObjectType]]
ObjectTypeArgument = Optional[Iterable[ObjectType]]


class NamedObjectStateMixin(object):
    """A state type mixin with named objects."""

    def __init__(self, object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None):
        """A state type mixin with named objects.
        The object names can be either a list of names, or a mapping from names to :class:`ObjectType`'s.

            - If the `object_names` is a list of names, then the user should also specify a list of object types.
            - If the `object_names` is a mapping from names to :class:`ObjectType`'s, then the `object_types` argument should be None.

        Args:
            object_names: the object names.
            object_types: the object types.
        """
        if isinstance(object_names, Mapping):
            assert object_types is None, 'object_types should be None if object_names is a mapping.'
            self.object_names = tuple(object_names.keys())
            self.object_types = tuple(object_names.values())
        else:
            assert object_types is not None, 'object_types should not be None if object_names is not a mapping.'
            self.object_names = tuple(object_names)
            self.object_types = tuple(object_types)

        self.object_type2name: Dict[str, List[str]] = dict()
        self.object_name2index: Dict[Tuple[str, str], int] = dict()
        self.object_name2defaultindex: Dict[str, Tuple[str, int]] = dict()

        for name, obj_type in zip(self.object_names, self.object_types):
            self.object_type2name.setdefault(obj_type.typename, list()).append(name)
            self.object_name2index[name, obj_type.typename] = len(self.object_type2name[obj_type.typename]) - 1
            self.object_name2defaultindex[name] = obj_type.typename, len(self.object_type2name[obj_type.typename]) - 1
            for t in obj_type.iter_parent_types():
                self.object_type2name.setdefault(t.typename, list()).append(name)
                self.object_name2index[name, t.typename] = len(self.object_type2name[t.typename]) - 1

    @property
    def nr_objects(self) -> int:
        """The number of objects in the current state."""
        return len(self.object_types)

    def get_typed_index(self, name: str, typename: Optional[str] = None) -> int:
        """Get the typed index of the object with the given name.
        There is an additional typename argument to specify the type of the object.
        Because the same object can have multiple types (due to inheritence), the object can have multiple typed indices, one for each type.
        When the typename is not specified, the default type of the object is used (i.e., the most specific type).

        Args:
            name: the name of the object.
            typename: the typename of the object. If not specified, the default type of the object is used (i.e. the most specific type).

        Returns:
            the typed index of the object.
        """
        if typename is None or typename == AutoType.typename:
            return self.object_name2defaultindex[name][1]
        return self.object_name2index[name, typename]

    def get_default_typename(self, name: str) -> str:
        """Get the typename of the object with the given name."""
        return self.object_name2defaultindex[name][0]

    def get_name(self, typename: str, index: int) -> str:
        """Get the name of the object with the given type and index."""
        return self.object_type2name[typename][index]

    def get_objects_by_type(self, typename: str) -> List[str]:
        """Get the names of the objects with the given type."""
        return self.object_type2name[typename]

    def get_nr_objects_by_type(self, typename: str) -> int:
        """Get the number of objects with the given type."""
        return len(self.object_type2name[typename])

    def get_state_object_reference(self, dtype: Union[ObjectType, str], index: Optional[int] = None, name: Optional[str] = None) -> StateObjectReference:
        """Get the object reference with the given type and index."""
        if isinstance(dtype, str):
            dtype, typename = ObjectType(dtype), dtype
        else:
            typename = dtype.typename

        if index is not None:
            return StateObjectReference(self.get_name(typename, index), index, dtype)
        if name is not None:
            return StateObjectReference(name, self.get_typed_index(name, typename), dtype)
        raise ValueError('Either indices or names should be specified.')

    def get_state_object_list(self, dtype: Union[ObjectType, str], indices: Optional[Union[Sequence[int], slice]] = None, names: Optional[Sequence[str]] = None) -> StateObjectList:
        """Get a list of object references with the given type and indices."""
        if isinstance(dtype, str):
            dtype, typename = ObjectType(dtype), dtype
        else:
            typename = dtype.typename
        if indices is not None:
            if isinstance(indices, slice):
                if indices != QINDEX:
                    raise ValueError('Only QINDEX is allowed for the indices of StateObjectList.')
                return StateObjectList(ListType(dtype), QINDEX)
            return StateObjectList(ListType(dtype), [self.get_state_object_reference(typename, index) for index in indices])
        if names is not None:
            return StateObjectList(ListType(dtype), [self.get_state_object_reference(typename, name=name) for name in names])
        raise ValueError('Either indices or names should be specified.')

class NamedObjectTensorState(TensorState, NamedObjectStateMixin):
    """A state type with named objects."""

    def __init__(self, features: Optional[Union[Mapping[str, Any], MultidimensionalArrayInterface]], object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None, batch_dims: int = 0, internals: Optional[Mapping[str, Any]] = None):
        """Initialize the state.

        Args:
            features: the features of the state.
            object_types: the types of the objects.
            object_names: the names of the objects. If the object_names is a mapping, the object_types should be None.
            batch_dims: the number of batchified dimensions.
            internals: the internals of the state.
        """

        TensorState.__init__(self, features, batch_dims, internals)
        NamedObjectStateMixin.__init__(self, object_names, object_types)

    def clone(self) -> 'NamedObjectTensorState':
        return type(self)(features=self._features.clone(), object_types=self.object_types, object_names=self.object_names, batch_dims=self._batch_dims, internals=self.clone_internals())

    def extra_state_str_before(self) -> str:
        """Extra state string: add the objects."""
        if self.object_names is not None:
            typename2objects = dict()
            for name, dtype in zip(self.object_names, self.object_types):
                typename2objects.setdefault(dtype.typename, list()).append(name)
            objects_str = '; '.join([f'{typename}: [{", ".join(names)}]' for typename, names in typename2objects.items()])
        else:
            objects_str = ', '.join(self.object_names)
        return '  objects: ' + objects_str + '\n'

class QuantizedTensorState:
    def __init__(self, data):
        self.data = data
    
    def quantize(self):
        return self.data

def concat_states(*args: TensorState) -> TensorState:
    """Concatenate a list of states into a batch state.

    Args:
        *args: a list of states.

    Returns:
        a new state, which is the concatenation of the input states.
        This new state will have a new batch dimension.
    """

    if len(args) == 0:raise ValueError('No states to concatenate.')

    all_features = list(args[0].features.all_feature_names)

    # 1. Sanity checks.
    for state in args[1:]:
        assert len(all_features) == len(state.features.all_feature_names)
        for feature in all_features:
            assert feature in state.features.all_feature_names

    # 2. Put the same feature into a list.
    features = {feature_name: list() for feature_name in all_features}
    for state in args:
        for key, value in state.features.tensor_dict.items():
            features[key].append(value)

    # 3. Actually, compute the features.
    feature_names = list(features.keys())
    for feature_name in feature_names:
        features[feature_name] = concat_tvalues(*features[feature_name])

    # 4. Create the new state.
    state = args[0]
    kwargs: Dict[str, Any] = dict()
    if isinstance(state, NamedObjectTensorState):
        kwargs = dict(object_types=state.object_types, object_names=state.object_names)

    kwargs['features'] = features
    kwargs['batch_dims'] = args[0].batch_dims + 1
    return type(state)(**kwargs)

"""action precondition and effect parts,"""

class Precondition:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self, bool_expression) -> bool:
        return 0
    
    def __str__(self): return str(self.bool_expression)

class Effect:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self) -> bool:
        return 0
    
    def __str__(self):return str(self.bool_expression)
    
    def split_effect(self):
        """split the effect into two parts. effect+, effect-
        Returns:
            effect+: the added predicates to the known state
            effect-: the removed predicates to the known state
        """
        return 0

class Action:
    def __init__(self, action_name, parameters, precondition, effect, sampler = None):
        """ construct an symboic action with preconditinos and effects
        Args:
            action_name: the name of the action
            precondition: a boolean expression that is callable for an input predicate state
            effect: a set of assignment expressions to known predicates
        """
        super().__init__()
        self.action_name = action_name
        self.parameters = parameters
        if isinstance(precondition, Precondition):
            self.precondition = precondition
        else: self.precondition = Precondition(precondition)

        if not isinstance(effect, Effect):
            self.effect = Effect(effect)
        else: self.effect = effect

        self.sampler  = sampler
    
    def apply(self, state):
        if self.precondition(state):
            return
