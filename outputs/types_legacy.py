import copy
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import FLOAT, ListType, VectorType, EmbeddingType, TypeBase
from helchriss.dsl.dsl_types import VectorType, ListType, EmbeddingType, TupleType, FixedListType, ArrowType, BatchedListType, BOOL
from typing import List, Tuple,Union, Any, Dict, Tuple, Optional, Callable, Type




__all__ = ["PatternVar","match_pattern", "TransformRule", "RuleBasedTransform", "get_transform_rules"]

"""Match the Pattern for Tree Regular Language"""
class PatternVar(TypeBase):
    """variable in the pattern"""
    def __init__(self, var_name: str):
        super().__init__(f"${var_name}")  # $ as the mark for the variable
        self.var_name = var_name

    def __eq__(self, other: TypeBase) -> bool:  return True 

class TransformRule:
    def __init__(self, 
                 source_pattern: TypeBase,
                 transform_func: Callable[[Dict[str, TypeBase]],TypeBase],
                 target_pattern: TypeBase = None):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.transform_func = transform_func

    def apply(self, source_type, target_type):
        source_vars = match_pattern(source_type, self.source_pattern)
        if source_vars is None : return None
        if self.target_pattern is not None and not match_pattern(target_type, self.target_pattern): return None
        return self.fill_in_func(source_vars)

class FillerRule:
    def __init__(self,
                 source_pattern : TypeBase,
                 target_pattern : TypeBase,
                 filler):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.filler = filler
        assert self.filler is not None, f"filler {source_pattern} -> {target_pattern} is None"
    
    def applicable(self, source_type : TypeBase, target_type):
        source_vars = match_pattern(source_type, self.source_pattern)
        #print(source_vars, source_type, self.source_pattern)
        target_vars = match_pattern(target_type, self.target_pattern)
        #print(target_vars, target_type, self.target_pattern)
        return source_vars is not None and target_vars is not None
    
    def fill_in(self, source_type : TypeBase, target_type : TypeBase):
        source_vars = match_pattern(source_type, self.source_pattern)
        #print(source_vars, source_type, self.source_pattern)
        target_vars = match_pattern(target_type, self.target_pattern)
        #print(target_vars, target_type, self.target_pattern)
        
        return self.filler({**source_vars, ** target_vars})

def match_pattern(
    target_type: TypeBase,
    pattern: TypeBase,
    bindings: Optional[Dict[str, TypeBase]] = None
) -> Optional[Dict[str, TypeBase]]:
    """Match target type against pattern and return consistent variable bindings."""
    bindings = bindings or {}

    # Case 1: Pattern is a variable -> bind and check consistency
    if isinstance(pattern, PatternVar):
        var_name = pattern.var_name
        if var_name in bindings:
            if bindings[var_name] != target_type:
                return None
        else:
            bindings[var_name] = target_type
        return bindings

    # Case 2: List/FixedList/Vector types (uniform sequence types)
    target_is_seq = isinstance(target_type, (ListType, FixedListType, VectorType))
    pattern_is_seq = isinstance(pattern, (ListType, FixedListType, VectorType))
    
    if target_is_seq and pattern_is_seq:
        # 1. Validate sequence type compatibility (e.g., List ↔ FixedList is allowed only if intentional)
        # Strict check: same sequence subclass (adjust if you want loose matching)
        if type(target_type) != type(pattern):
            return None

        # 2. Match element type recursively
        elem_bindings = match_pattern(
            target_type.element_type,
            pattern.element_type,
            copy.deepcopy(bindings)
        )
        #print("Elem:", elem_bindings)
        if elem_bindings is None:
            return None

        # 3. Handle FixedList-specific length check (supports PatternVar)

        if isinstance(target_type, ListType):

            length_bindings = match_pattern(
                target_type.element_type,
                pattern.element_type,
                copy.deepcopy(elem_bindings)
            )
            return length_bindings
            #elem_bindings = length_bindings
            #print("LEN",elem_bindings)


        # 4. Handle Vector-specific dim check (supports PatternVar)
        if isinstance(target_type, VectorType):
            dim_bindings = match_pattern(
                target_type.dim,
                pattern.dim,
                copy.deepcopy(elem_bindings)
            )
            if dim_bindings is None:
                return None
            elem_bindings = dim_bindings


        return elem_bindings

    # Case 3: TupleType (multiple element types)
    if isinstance(target_type, TupleType) and isinstance(pattern, TupleType):
        if len(target_type.element_types) != len(pattern.element_types):
            return None
        
        new_bindings = copy.deepcopy(bindings)
        #print("start:", target_type.element_types, pattern.element_types)
        for t_elem, p_elem in zip(target_type.element_types, pattern.element_types):
            #print("enter tuple",t_elem, p_elem, new_bindings)

            elem_bindings = match_pattern(t_elem, p_elem, new_bindings)
            #print("tuple pair:",t_elem, p_elem)
            if elem_bindings is None:
                #print("EXIT")
                return None
            new_bindings.update(elem_bindings)
            #print('tuple:',elem_bindings)
        return new_bindings

    # Case 4: EmbeddingType (fixed to support PatternVar)
    if isinstance(target_type, EmbeddingType) and isinstance(pattern, EmbeddingType):
        # Match space_name (supports PatternVar)
        space_bindings = match_pattern(
            target_type.space_name,
            pattern.space_name,
            copy.deepcopy(bindings)
        )
        if space_bindings is None:
            return None
        
        # Match dim (supports PatternVar)
        dim_bindings = match_pattern(
            target_type.dim,
            pattern.dim,
            copy.deepcopy(space_bindings)
        )

        return dim_bindings if dim_bindings is not None else None

    # Case 5: ArrowType (function/arrow types)
    if isinstance(target_type, ArrowType) and isinstance(pattern, ArrowType):
        first_bindings = match_pattern(target_type.first, pattern.first, copy.deepcopy(bindings))
        if first_bindings is None:
            return None
        second_bindings = match_pattern(target_type.second, pattern.second, first_bindings)
        return second_bindings

    # Case 6: Exact match for primitive types
    return bindings if target_type == pattern else None

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
            bindings = match_pattern(current_type, rule.source_pattern)
            if bindings is None: continue
            new_type = rule.transform_func(bindings)
            new_path = path + [(rule, bindings)]
            queue.append((new_type, new_path, depth + 1))
    return None

class MLPArgumentCaster(nn.Module):
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
            nn.Linear(64, 1)
        )
    
    def forward(self, *args):
        flat_args = [arg.value.reshape(-1) for arg in args[0]]
        cat_args = torch.cat(flat_args, dim=0)
        output = self.net(cat_args)
        outputs = [t.reshape([d]) for t, d in zip(torch.split(output, self.output_dims), self.output_dims)]
        logit_output = self.logit_net(cat_args)
        args = [o for i,o in enumerate(outputs)]
        logits = torch.sum(logit_output)
        return args, logits

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


class ConvexConstruct:
    def __init__(self, functions: List[Union[Callable, "ConvexConstruct"]], weights: np.ndarray = None):
        self.functions = functions  # a function or a convex construct
        self.num_functions = len(functions)

        if weights is None:
            self.weights = np.ones(self.num_functions) / self.num_functions
        else:
            assert len(weights) == self.num_functions, f"num weights {len(weights)} numfunctions {self.num_functions}"
            assert np.all(weights >= 0)
            self.weights = weights / np.sum(weights)  # 归一化确保凸组合条件
    
    def __call__(self, *args, **kwargs) -> Any:
        outputs = []
        for func in self.functions:
            outputs.append(func(*args, **kwargs))

        output_type = type(outputs[0])
        weighted_sum = output_type(0)
        for a_i, f_i_output in zip(self.weights, outputs):
            weighted_sum += a_i * f_i_output
        
        return weighted_sum
    
    def normalize_weights(self):
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / np.sum(self.weights)
    
    def get_top_p_functions(self, p: float) -> "ConvexConstruct":
        assert 0 < p <= 1, "p need to within the (0,1]"
        top_k = int(self.num_functions * p)
        top_k = max(top_k, 1)

        top_indices = np.argsort(self.weights)[-top_k:]
        top_functions = [self.functions[i] for i in top_indices]
        top_weights = self.weights[top_indices]
        
        return ConvexConstruct(top_functions, top_weights)

class RuleBasedTransform:

    def __init__(self, transform_rules : List[TransformRule], filler_rules : List[FillerRule]):
        self.transform_rules = transform_rules
        self.filler_rules = filler_rules
        self.k = 3
    
    def add_rule(self, rule : TransformRule): self.transform_rules.append(rule)


    """infer the possible set of fillers from signature """
    def infer_fn_prototypes(self,input_type : List[TypeBase], output_type : TypeBase):
        #print( TupleType(input_type), output_type)
        
        #paths = find_transform_path(
        #    TupleType(input_type),
        #    output_type, rules=self.transform_rules,
        #    max_depth=self.k)

        input_types = TupleType(input_type) if len(input_type) > 1 else input_type[0]

        fillers = []

        
        for fill_rule in self.filler_rules:
            #print(TupleType(input_type), output_type)
            if fill_rule.applicable(input_types, output_type):

                fillers.append( fill_rule.fill_in(input_types, output_type) )
        assert fillers, f"not found for {input_types} to {output_type}"
        return fillers
        #
    

    """infer argument transformation using the predefined transformation rules"""
    def infer_args_caster(self,input_type : List[TypeBase], output_types : List[TypeBase]) -> nn.Module:
        input_dims = [type_dim(tp) for tp in input_type]
        output_dims = [type_dim(tp) for tp in output_types]
        return MLPArgumentCaster(input_dims, output_dims)

import torch
import torch.nn as nn
from typing import List, Callable

class ListFloatFiller(nn.Module):
        def __init__(self,n,m):
            super().__init__()
            # Linear layer: n+m input dim → 1 output dim
            hidden_dim = 128
            self.linear = nn.Sequential(
                nn.Linear(n+m, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.ReLU(),
            )

        def forward(self, list1: List[torch.Tensor], list2: List[torch.Tensor], **kwargs) -> torch.Tensor:
            # Step 1: Compute expectation (mean) over each list (collapse list to single tensor)
            # list1: [B1, 1+n] → mean over list → [1+n]; list2: [B2, 1+m] → mean → [1+m]
            exp1 = list1.mean(dim=0)  # Expectation over list1
            exp2 = list2.mean(dim=0)  # Expectation over list2
            
            # Step 2: Strip logit dimension (first dim) → [n] and [m]
            feat1 = exp1[1:]  # Drop logit (1st dim) → shape [n]
            feat2 = exp2[1:]  # Drop logit (1st dim) → shape [m]
            
            combined = torch.cat([feat1, feat2], dim=0)  # Shape [n+m]
            
            output = self.linear(combined)  # Shape [1]
            return Value(FLOAT,output)

def obj_list_to_float_filler(binds: dict) -> Callable[[], nn.Module]:
    """
    Factory function returning a custom nn.Module class that:
    - Takes two list inputs (list of [1+n] and [1+m] dim tensors)
    - Computes expectation (mean) over each list
    - Combines to n+m dim tensor, then projects to 1 dim via Linear
    
    Args:
        binds: Dictionary with "n" and "m" keys (defines tensor dimensions)
    
    Returns:
        Callable that instantiates the custom nn.Module
    """
    n = int(binds["n"]);m = int(binds["m"])
    
    return ListFloatFiller(n,m)


class FloatFiller(nn.Module):
        def __init__(self,k):
            super().__init__()
            # Linear layer: k input dim → 1 output dim
            hidden_dim = 128
            self.feat_net = nn.Sequential(
                nn.Linear(k+0, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, hidden_dim),

            )
            self.decode_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, 1),
            )
              

        def forward(self, input_list: List[torch.Tensor], **kwargs) -> Value:
            # Step 1: Compute expectation (mean) over the input list
            # input_list: list of [1+k] tensors → stack + mean → [1+k]
            stacked = input_list  # Shape: [num_tensors, 1+k]
            #exp = stacked.mean(dim=0)         # Expectation over list → [1+k]
            

            feat = self.feat_net(stacked[:,1:])  # Drop logit → shape [k]
            prob = stacked[:,0:1].sigmoid()
            # Step 3: Project k-dim to 1 dim via linear layer
            decoded = self.decode_net(feat) * prob
            #print("prob:", prob.reshape([-1]))
            #print("num cast:",decoded.reshape([-1]))
            output_tensor = torch.max(decoded)  # Shape [1]


            
            # Step 4: Wrap in Value (matches your original return)
            return Value(FLOAT, output_tensor)


def single_obj_list_to_float_filler(binds: dict) -> nn.Module:
    """
    Factory function returning a custom nn.Module that:
    - Takes ONE list input (list of [1+k] dim tensors, k = binds["k"])
    - Computes expectation (mean) over the list
    - Strips logit dimension (first dim) to get k-dim tensor
    - Projects k-dim tensor to 1 dim via Linear
    
    Args:
        binds: Dictionary with "k" key (defines tensor dimension: [1+k])
    
    Returns:
        Instantiated nn.Module (matches your original return FloatFiller())
    """
    k = int(binds["k"])  # k = dimension after stripping logit (1+k total)
    # Return INSTANTIATED module (matches your original return FloatFiller())
    return FloatFiller(k)

def get_transform_rules():
    object_dim = 128
    def obj_tuple_to_embedding_tp(binding):

        return EmbeddingType("object", binding["n1"] + binding["n2"])

    rule_obj_tuple_to_embedding_transform = TransformRule(
        source_pattern = TupleType([
            ListType(EmbeddingType("object", PatternVar("n1"))),
            ListType(EmbeddingType("object", PatternVar("n2")))
        ]
        ),
        transform_func =obj_tuple_to_embedding_tp,
    )




    transform_rules = [
        rule_obj_tuple_to_embedding_transform
    ]

    def emb2emb_filler(binds):
        return nn.Linear(binds["n"], binds["m"])
    emb2emb_fill_rule = FillerRule(
        EmbeddingType(PatternVar("K"), PatternVar("n")),
        EmbeddingType(PatternVar("C"), PatternVar("m")),
        emb2emb_filler
    )

    def emb2vector_filler(binds):
        return nn.Linear(binds["n"], binds["m"])
    emb2vector_fill_rule = FillerRule(
        EmbeddingType(PatternVar("K"), PatternVar("n")),
        VectorType(PatternVar("C"), PatternVar("m")),
        emb2vector_filler
    )

    def obj_list_to_emb_filler(binds):

        return nn.Linear(int(binds["n1"]) + int(binds["n2"]), int(binds["m"]))
    obj_list_to_emb_fill_rule = FillerRule(
        TupleType([
            ListType(EmbeddingType("object", PatternVar("n1"))),
            ListType(EmbeddingType("object", PatternVar("n2")))
        ]
        ),
        EmbeddingType(PatternVar("name"), PatternVar("m")),
        obj_list_to_emb_filler
    )


    obj_list_to_float_fill_rule = FillerRule(
        TupleType([
            ListType(TupleType([BOOL, EmbeddingType("object", PatternVar("n")) ]) ),
            ListType(TupleType([BOOL, EmbeddingType("object", PatternVar("m")) ]) )
        ]
        ),
       FLOAT,
        obj_list_to_float_filler
    )
    #print(ListType(TupleType([BOOL, EmbeddingType("object", PatternVar("k")) ]) ),)
    single_obj_list_to_float_fill_rule = FillerRule(
            ListType(TupleType([BOOL, EmbeddingType("object", PatternVar("k")) ]) ),
            FLOAT,
            single_obj_list_to_float_filler
    )

    filler_rules = [
        emb2emb_fill_rule,
        emb2vector_fill_rule,
        obj_list_to_emb_fill_rule,
        obj_list_to_float_fill_rule,
        single_obj_list_to_float_fill_rule
    ]
    return transform_rules, filler_rules