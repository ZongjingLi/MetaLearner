import torch
import torch.nn as nn
from abc import abstractmethod
from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
from typing import List, Tuple,Union, Any, Dict, Tuple, Optional, Callable, Type
from dataclasses import dataclass, field
from helchriss.dsl.dsl_types import VectorType, ListType, EmbeddingType, TupleType, FixedListType, ArrowType, BatchedListType
import copy

__all__ = ["PatternVar","match_pattern", "CasterRegistry", "type_dim", "fill_hole", "infer_caster"]

"""Match the Pattern for Tree Regular Language"""
class PatternVar(TypeBase):
    """variable in the pattern"""
    def __init__(self, var_name: str):
        super().__init__(f"${var_name}")  # $ as the mark for the variable
        self.var_name = var_name

    def __eq__(self, other: TypeBase) -> bool:  return True 

def match_pattern(
    target_type: TypeBase,
    pattern: TypeBase,
    bindings: Optional[Dict[str, TypeBase]] = None
) -> Optional[Dict[str, TypeBase]]:
    """ if the pattern matches the
    Args:
        target_type: the type to match
        pattern: pattern to match
        bindings: the known pattern to match
    
    Returns:
        variable binding that maps %var to the actual type or value
    """
    bindings = bindings or {}

    # Case 1：pattern is single variable -> binding
    if isinstance(pattern, PatternVar):
        var_name = pattern.var_name
        if var_name in bindings: ### check if the variable binding is consistent
            if bindings[var_name] != target_type: return None
        else: bindings[var_name] = target_type
        return bindings

    # Case 2：target type inconsistent with the  -> failed to match
    if type(target_type) != type(pattern): return None

    try: # compare the type name
        if not (hasattr(target_type, 'element_type') or hasattr(target_type, 'element_types')):
            return bindings if target_type == pattern else None
    except: return None

    # nested type -> recursive match the subtype
    # 4.1 ListType（uniform sequence, single element type）
    if isinstance(target_type, (ListType, FixedListType, VectorType, BatchedListType)):
        # check the subtyping
        sub_bindings = match_pattern(
            target_type.element_type,
            pattern.element_type,
            copy.deepcopy(bindings)
        )
        if sub_bindings is None: return None
        if isinstance(target_type, VectorType):
            if target_type.dim != pattern.dim and not isinstance(pattern.dim, PatternVar):
                return None
        if isinstance(target_type, FixedListType):
            if target_type.typename != pattern.typename:  # 包含length信息
                return None
        return sub_bindings

    # 4.2 TupleType with multiple elemnt types
    if isinstance(target_type, TupleType):
        if len(target_type.element_types) != len(pattern.element_types):
            return None
        # recusrive match each element type
        new_bindings = copy.deepcopy(bindings)
        for t_elem, p_elem in zip(target_type.element_types, pattern.element_types):
            elem_bindings = match_pattern(t_elem, p_elem, new_bindings)
            if elem_bindings is None:
                return None
            new_bindings.update(elem_bindings)
        return new_bindings

    # 4.3 EmbeddingType : space_name and dim
    if isinstance(target_type, EmbeddingType):
        if (target_type.space_name != pattern.space_name or 
            target_type.dim != pattern.dim):
            return None
        return bindings

    # 4.4 ArrowType : match the firs second
    if isinstance(target_type, ArrowType):
        first_bindings = match_pattern(target_type.first, pattern.first, copy.deepcopy(bindings))
        if first_bindings is None:
            return None
        second_bindings = match_pattern(target_type.second, pattern.second, first_bindings)
        return second_bindings

    return bindings if target_type == pattern else None

class TransformRule:
    def __init__(self, source_pattern: TypeBase, transform_func: Callable[[Dict[str, TypeBase]], TypeBase], target_pattern = None):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.transform_func = transform_func

    def apply(self, source_type, target_type):
        source_vars = match_pattern(source_type, self.source_pattern)
        if source_vars is None : return None
        if self.target_pattern is not None and not match_pattern(target_type, self.target_pattern): return None
        return self.transform_func(source_vars)


def find_transform_path(
    initial_type: TypeBase, 
    target_type: TypeBase,
    rules: List[TransformRule],
    max_depth: int = 5
) -> Optional[List[Tuple[TransformRule, Dict[str, TypeBase]]]]:
    """    
    Args:
        initial_type: the initial type to transform
        target_type:  the taraget type to transform
        rules:        rules to transform
        max_depth:    depth constraints
    
    Returns:
        return the possible (rules, bindings) tuples
    """
    # the bfs queue of transform rules
    queue = [(initial_type, [], 0)]
    visited = set()

    while queue:
        current_type, path, depth = queue.pop(0)
        if current_type == target_type: return path

        if depth >= max_depth: continue
        type_key = current_type.typename
        if type_key in visited: continue
        visited.add(type_key)

        for rule in rules:
            bindings = match_pattern(current_type, rule.pattern)
            if bindings is None: continue

            try: new_type = rule.transform_func(bindings)
            except Exception: continue

            new_path = path + [(rule, bindings)]
            queue.append((new_type, new_path, depth + 1))
    return None


class RuleBasedTransformInferer:
    def __init__(self):
        self.rules = [] # tuples of (TransformRule, priority)
        self._register_default_rules()
    
    def _register_default_rules(self):
        pass
    
    def register_rule(self, rule: TransformRule, priority = 0.0):
        self.rules.append([rule, priority])
        self.rules.sort(key=lambda r: r[1], reverse=True)
    
    def infer_caster(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> nn.Module:
        print("input types:")
        [print(a) for a in input_types]
        print("output types:")
        [print(a) for a in output_types]
        for rule in self.rules:
            if rule.can_apply(input_types, output_types):
                return rule.create_caster(input_types, output_types)
        
        return fallback_infer_caster(input_types, output_types)

def type_dim(tp : TypeBase) -> int:
    if isinstance(tp, TypeBase) and tp.typename in ["int", "float", "boolean", "bool"]: return 1
    elif isinstance(tp, VectorType): return int(tp.dim)
    elif isinstance(tp, EmbeddingType): return int(tp.dim)
    elif isinstance(tp, ListType): type_dim(tp.element_type)
    elif isinstance(tp, TupleType):return sum(type_dim(elem) for elem in tp.element_types)
    elif isinstance(tp, FixedListType):
        length = tp.length if isinstance(tp.length, int) else 10  # 默认长度
        return length * type_dim(tp.element_type)
    raise NotImplementedError(f"dim of type {tp} cannot be inferred")


def fill_hole(arg_types : List[TypeBase], out_type : TypeBase) -> nn.Module:
    complexity = analyze_type_complexity(arg_types + [out_type])
    hidden_size = min(64 * (complexity // 2 + 1), 512)
    num_layers = min(complexity // 3 + 2, 5)
    
    in_dim = sum([type_dim(tp) for tp in arg_types])
    out_dim = type_dim(out_type)

    layers = []
    layers.append(nn.Linear(in_dim, hidden_size))
    layers.append(nn.ReLU())
    
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_size, out_dim))
    
    net = nn.Sequential(*layers)
    filler = MLPFiller(arg_types, out_type, net)
    return filler

def analyze_type_complexity(types: List[TypeBase]) -> int:
    complexity = 0
    
    for tp in types:
        if isinstance(tp, ListType) or isinstance(tp, TupleType):
            complexity += 2
            try:
                complexity += analyze_type_complexity([tp.element_type])
            except: complexity + 1
            try:
                complexity += analyze_type_complexity(tp.element_types)
            except: complexity += 1
        elif isinstance(tp, VectorType) or isinstance(tp, EmbeddingType):
            complexity += 1
        else:
            complexity += 0
    
    return complexity

class MLPCaster(nn.Module):
    def __init__(self, input_dims : List[int], output_dims : List[int]):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.total_input_dim = sum(input_dims)
        self.total_output_dim = sum(output_dims)
        
        self.net = nn.Sequential(
            nn.Linear(self.total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.total_output_dim)
        )
        self.logit_net = nn.Sequential(
            nn.Linear(self.total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.output_dims))
        )
    
    def forward(self, *args):
        flat_args = [arg.value.reshape(-1) for arg in args[0]]
        cat_args = torch.cat(flat_args, dim=0)


        output = self.net(cat_args)
        outputs = [t.reshape([d]) for t, d in zip(torch.split(output, self.output_dims), self.output_dims)]

        logit_output = self.logit_net(cat_args)

        tuple_output = [(o, logit_output[i]) for i,o in enumerate(outputs)]
        return tuple_output

def infer_mlp_caster(input_type : List[TypeBase], output_types : List[TypeBase]) -> nn.Module:
    input_dims = [type_dim(tp) for tp in input_type]
    output_dims = [type_dim(tp) for tp in output_types]
    return MLPCaster(input_dims, output_dims)
