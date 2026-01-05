import copy
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import FLOAT, ListType, VectorType, EmbeddingType, TypeBase
from helchriss.dsl.dsl_types import VectorType, ListType, EmbeddingType, TupleType, FixedListType, ArrowType, BatchedListType, BOOL
from typing import List, Tuple, Union, Any, Dict, Optional, Callable, Type

class FCBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
    
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


    def __repr__(self): return str(self)

    def __str__(self): return f"{self.input_dim}_{self.num_layers}_{self.output_dim}"

class PatternVar(TypeBase):
    def __init__(self, var_name: str):
        super().__init__(f"${var_name}")
        self.var_name = var_name

    def __eq__(self, other: TypeBase) -> bool: return True 

def _match_attribute(pattern_attr: Any, target_attr: Any, var_binds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    if isinstance(pattern_attr, PatternVar):
        var_name = pattern_attr.var_name
        if var_name in var_binds:
            return var_binds[var_name] == target_attr, var_binds
        var_binds[var_name] = target_attr
        return True, var_binds
    
    if pattern_attr is None:
        return True, var_binds
    
    return pattern_attr == target_attr, var_binds

def match_pattern(pattern: TypeBase, target: TypeBase, var_binds: Optional[Dict[str, TypeBase]] = None) -> Tuple[bool, Dict[str, TypeBase]]:
    if var_binds is None:
        var_binds = {}
    
    if isinstance(pattern, PatternVar):
        var_name = pattern.var_name
        if var_name in var_binds:
            return var_binds[var_name] == target, var_binds
        var_binds[var_name] = target
        return True, var_binds
    
    if type(pattern) != type(target):
        return False, var_binds
    
    basic_types = (VectorType,)
    if any(isinstance(pattern, tp) for tp in basic_types):
        for attr in pattern.__dict__:
            pattern_attr_val = getattr(pattern, attr)
            target_attr_val = getattr(target, attr)
            
            if pattern_attr_val is None or target_attr_val is None:
                continue
            
            if pattern_attr_val != target_attr_val:
                return False, var_binds
        return True, var_binds
    
    if isinstance(pattern, EmbeddingType):
        if pattern.space_name != target.space_name:
            return False, var_binds
        
        dim_match, var_binds = _match_attribute(pattern.dim, target.dim, var_binds)
        if not dim_match:
            return False, var_binds
        
        return True, var_binds
    
    if isinstance(pattern, ListType):
        elem_match, var_binds = match_pattern(pattern.element_type, target.element_type, var_binds)
        if not elem_match:
            return False, var_binds

        if isinstance(pattern, FixedListType) and isinstance(target, FixedListType):
            if pattern.length != target.length:
                return False, var_binds

        if isinstance(pattern, BatchedListType) and isinstance(target, BatchedListType):
            if pattern.batch_dim != target.batch_dim:
                return False, var_binds
        return True, var_binds
    
    if isinstance(pattern, TupleType):
        if len(pattern.element_types) != len(target.element_types):
            return False, var_binds
        for pat_elem, tgt_elem in zip(pattern.element_types, target.element_types):
            elem_match, var_binds = match_pattern(pat_elem, tgt_elem, var_binds)
            if not elem_match:
                return False, var_binds
        return True, var_binds

    if isinstance(pattern, ArrowType):
        in_match, var_binds = match_pattern(pattern.input_type, target.input_type, var_binds)
        if not in_match:
            return False, var_binds
        out_match, var_binds = match_pattern(pattern.output_type, target.output_type, var_binds)
        return out_match, var_binds
    
    return True, var_binds

class TypeTransformRule:
    def __init__(self, pattern_type : TypeBase, apply_fn: Callable[[Dict[str, TypeBase]], Tuple[TypeBase, Callable]],
                 name: Optional[str] = None)-> Tuple[bool, Dict[str, TypeBase]]:
        self.pattern_type = pattern_type
        self.apply_fn     = apply_fn
        self.name = name or f"Rule({pattern_type.name})"

    def match(self, other_type : TypeBase):
        return match_pattern(self.pattern_type, other_type)
    
    def apply(self, var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]: return self.apply_fn(var_binds)

class TypeTransformRuleBackward:
    def __init__(self, tgt_pattern: TypeBase, type_fn: Callable, apply_fn: Callable, name: str = None):
        self.tgt_pattern = tgt_pattern
        self.apply_fn = apply_fn 
        self.type_fn = type_fn 
        self.name = name or f"BackwardRule({str(tgt_pattern)})"

    def match(self, other_type: TypeBase) -> tuple[bool, Dict[str, Any]]:
        return match_pattern(self.tgt_pattern, other_type)
    
    def sub_goals(self, other_type: TypeBase) -> List[TypeBase]:
        match_success, var_binds = self.match(other_type)
        if not match_success: return []
        return self.type_fn(var_binds, other_type)
    
    def apply(self, var_binds: Dict[str, TypeBase]) -> Callable:

        return self.apply_fn(var_binds)
    


def _embed_to_float_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    dim = int(var_binds.get("dim", 128))

    return FLOAT, FCBlock(dim, 1)

def _embed_to_bool_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    embed_dim = int(var_binds.get("dim", 128))
    return BOOL, FCBlock(embed_dim, 1)

def _float_to_embed_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    return EmbeddingType("Latent", 128), FCBlock(1, 128)

def _bool_to_embed_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    return EmbeddingType("Latent", 128), FCBlock(1, 128)

def _embed_to_latent_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    in_dim = int(var_binds.get("dim", 128))
    out_dim = 128
    return EmbeddingType("Latent", 128), FCBlock(in_dim, out_dim)

def _listA_to_listB_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    list_a_type = var_binds.get("L1", ListType(PatternVar("A")))
    elem_b_type = var_binds.get("B", EmbeddingType())
    
    class ListAToListB(nn.Module):
        def __init__(self):
            super().__init__()
            self.elem_transform = FCBlock(elem_b_type.dim or 128, elem_b_type.dim or 128)
        
        def forward(self, x: List[Any]) -> List[Any]:
            return [self.elem_transform(elem) if isinstance(elem, torch.Tensor) else elem for elem in x]

    return list_a_type, ListAToListB()

def _listA_to_embed_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    list_a_type = var_binds.get("L", ListType(PatternVar("A")))
    
    class ListAToEmbed(nn.Module):
        def __init__(self, embed_dim: int = 128):
            super().__init__()
            self.embed_dim = embed_dim
            self.fc = FCBlock(embed_dim, embed_dim)
        
        def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
            concatenated = torch.cat(x, dim=0)
            return self.fc(concatenated)

    return list_a_type, ListAToEmbed()

def _tuple_of_embeds_to_embed_apply(var_binds: Dict[str, TypeBase]) -> Tuple[TypeBase, Callable]:
    # Get the tuple type from var binds and calculate total dimension

    total_dim = sum(val for key,val in var_binds.items())
    #target_embed_space = var_binds.get("space", "latent")
    
    # Module to flatten tuple of embeddings into a single embedding
    class TupleOfEmbedsToEmbed(nn.Module):
        def __init__(self, input_total_dim: int, output_dim: int):
            super().__init__()
            self.fc = FCBlock(input_total_dim, output_dim)
        
        def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
            # Concatenate all embedding tensors in the tuple
            concatenated = torch.cat(list(x), dim=-1)
            # Map to target embedding dimension
            return self.fc(concatenated)
    
    # Return target embedding type and the conversion module
    target_embed = EmbeddingType("TupleFix", total_dim)
    return target_embed, TupleOfEmbedsToEmbed(total_dim, total_dim)


def objects_to_embed_apply(var_binds):
    elem_b_type = var_binds.get("dim", -1)
    return EmbeddingType("objects")

VAR = TupleType([BOOL, EmbeddingType("object", PatternVar("dim"))] )
objects_to_embed = TypeTransformRule(
    pattern_type=ListType(VAR),
    apply_fn = objects_to_embed_apply,
    name = "objects_to_embed"
)

class SetPool(nn.Module):
    def __init__(self,k, max = True):
        super().__init__()
        hidden_dim = 128
        self.max = max
        self.feat_net = nn.Sequential(
                nn.Linear(k, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, hidden_dim),)
        self.decode_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, 1),)

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
        if self.max:
            output_tensor = torch.max(decoded)  # Shape [1]
        else:
            output_tensor = torch.mean(decoded)  # Shape [1]
            
        # Step 4: Wrap in Value (matches your original return)
        return Value(FLOAT, output_tensor)

def object_set_max_apply(var_binds):

    dim = var_binds.get("dim", -1)

    return FLOAT, SetPool(dim)


object_set_to_max = TypeTransformRule(
    pattern_type=ListType(VAR),
    apply_fn = object_set_max_apply,
    name = "max_objects"
)

tuple_of_embeds_to_embed = TypeTransformRule(
    pattern_type=TupleType([EmbeddingType(PatternVar("space_name"), PatternVar("dim"))]),  # Matches tuple with any Embedding elements
    apply_fn=_tuple_of_embeds_to_embed_apply,
    name="TupleOfEmbeddings_To_SingleEmbedding"
)

# Rule 2: Specific Tuple[Embedding[object, 96]] (example)
tuple_single_embed_96_to_embed = TypeTransformRule(
    pattern_type=TupleType([EmbeddingType(PatternVar("space_name"), PatternVar("dim"))]),
    apply_fn=_tuple_of_embeds_to_embed_apply,
    name="Tuple[Embedding[object,96]]_To_Embedding[96]"
)

# Rule 3: Specific Tuple[Embedding[object,96], Embedding[object,32]] (example)
tuple_double_embed_96_32_to_embed = TypeTransformRule(
    pattern_type=TupleType([
        EmbeddingType(PatternVar("space_name"), PatternVar("dim1")),
        EmbeddingType(PatternVar("space_name"), PatternVar("dim2"))]),
    apply_fn=_tuple_of_embeds_to_embed_apply,
    name="Tuple[Embedding[object,96],Embedding[object,32]]_To_Embedding[128]"
)

latent_to_float = TypeTransformRule(
    pattern_type=EmbeddingType("Latent", PatternVar("dim")),
    apply_fn=_embed_to_float_apply,
    name="Embed_to_Float"
)

latent_to_bool = TypeTransformRule(
    pattern_type=EmbeddingType("Latent", PatternVar("dim")),
    apply_fn=_embed_to_bool_apply,
    name="Embed_to_Bool"
)

float_to_embed = TypeTransformRule(
    pattern_type=FLOAT,
    apply_fn=_float_to_embed_apply,
    name="Float_to_Embed"
)

bool_to_embed = TypeTransformRule(
    pattern_type=BOOL,
    apply_fn=_bool_to_embed_apply,
    name="Bool_to_Embed"
)

embed_to_latent = TypeTransformRule(
    pattern_type=EmbeddingType(PatternVar("E"), PatternVar("dim")),
    apply_fn=_embed_to_latent_apply,
    name="Embed_to_Latent"
)

listA_to_listB = TypeTransformRule(
    pattern_type=ListType(PatternVar("A")),
    apply_fn=_listA_to_listB_apply,
    name="ListA_to_ListB"
)

listA_to_embed = TypeTransformRule(
    pattern_type=ListType(PatternVar("A")),
    apply_fn=_listA_to_embed_apply,
    name="ListA_to_Embed"
)

from helchriss.utils.tensor import Id

def _make_identity(var_binds):
    tp = var_binds.get("tp", -1)
    #print("tp:",tp, var_binds)
    return tp, Id()

id_rule = TypeTransformRule(
    pattern_type=PatternVar("tp"),
    apply_fn= _make_identity,
    name = "id_rule"
)

def _make_canon_object_embeder(var_binds):
    dim = var_binds.get("dim", -1)
    return FCBlock(128, dim)

canon_object_embedding_rule = TypeTransformRuleBackward(
    tgt_pattern = EmbeddingType("object", PatternVar("dim")),
    type_fn = lambda x : EmbeddingType("Latent", 128),
    apply_fn= _make_canon_object_embeder
)



"""backward rules to decompose"""
def _backward_embed_to_latent128_type_fn(var_binds: Dict[str, Any], target_type: EmbeddingType) -> List[TypeBase]:
    sub_goal_type = EmbeddingType("Latent", 128)
    return [sub_goal_type]

def _backward_embed_to_latent128_apply_fn(var_binds: Dict[str, Any]) -> Callable:
    target_space = var_binds.get("space_name", "Latent")
    target_dim = var_binds.get("dim", 128)
    
    return FCBlock(input_dim=128, output_dim=target_dim)

backward_embed_to_latent128 = TypeTransformRuleBackward(
    tgt_pattern=EmbeddingType(PatternVar("space_name"), PatternVar("dim")),
    type_fn=_backward_embed_to_latent128_type_fn,
    apply_fn=_backward_embed_to_latent128_apply_fn,
    name="Backward_AnyEmbed_To_Latent128_WithMLPFiller"
)



default_constructor_rules = [
    latent_to_float,
    latent_to_bool,
    embed_to_latent,
    #listA_to_listB,
    #listA_to_embed,

    #tuple_of_embeds_to_embed,
    #tuple_single_embed_96_to_embed,
    #tuple_double_embed_96_32_to_embed

    backward_embed_to_latent128,
    object_set_to_max,

    id_rule,

]