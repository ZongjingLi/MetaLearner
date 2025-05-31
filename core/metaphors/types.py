from abc import abstractmethod
from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
from typing import List, Tuple,Union, Any
import torch
import torch.nn as nn
import re

__all__ = ["type_dim", "fill_hole", "infer_caster"]

def parse_type_declaration(type_str):
    """
    Parse a type declaration string and extract the prefix and shape.
    
    Args:
        type_str (str): A string like "vector[float,[1]]" or "bool[float,[1,32,6]]"
    
    Returns:
        tuple: (prefix, shape) where prefix is the type name and shape is the dimension list
    """
    # Match the prefix and the shape part
    match = re.match(r'([a-zA-Z_]+)(?:\[.*?\])*?\[([^,\]]*,)*?(\[[0-9,]+\])\]', type_str)
    
    if not match:
        # If no shape part found, just extract the prefix part
        prefix_match = re.match(r'([a-zA-Z_]+)', type_str)
        return prefix_match.group(1) if prefix_match else type_str, None
        
    prefix = match.group(1)
    shape_str = match.group(3)
    
    # Convert shape string to actual list
    try:
        shape = eval(shape_str)
    except:
        shape = shape_str
        
    return prefix, shape

class BaseCaster(nn.Module):    
    def forward(self, args : List[Value]):
        tensor_args =[(arg.value).reshape([1,-1]) for arg in args]
        return self.cast(tensor_args)

    @abstractmethod
    def cast(self, input) -> List[Tuple[Any, torch.Tensor]]:
        raise NotImplementedError()


class LinearCaster(BaseCaster):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.linear_units = nn.ModuleList([])
        self.cast_units = nn.ModuleList([])
        for i in range(len(in_dims)):
            self.linear_units.append(nn.Linear(in_dims[i], out_dims[i]))
            self.cast_units.append(nn.Linear(in_dims[i], 1))
    
    def cast(self, args):
        return [(
            self.linear_units[i](arg).flatten(),
            torch.log(torch.sigmoid(self.cast_units[i](arg).flatten()))) for i,arg in enumerate(args)]

class MLPCaster(BaseCaster):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.mlp_units = nn.ModuleList()
        self.cast_units = nn.ModuleList()
        for in_dim, out_dim in zip(in_dims, out_dims):
            self.mlp_units.append(nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            ))
            self.cast_units.append(nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ))

    def cast(self, args):
        return [(
            self.mlp_units[i](arg).flatten(),
            torch.log(torch.sigmoid(self.cast_units[i](arg).flatten()))
        ) for i, arg in enumerate(args)]


def infer_caster(input_type : List[TypeBase], output_type : List[TypeBase]):
    in_prefix, in_shapes = list(), list()
    for arg in input_type:
        prefix, shape = parse_type_declaration(arg.typename)
        in_prefix.append(prefix)
        in_shapes.append(shape)
    
    out_prefix, out_shapes = list(), list()
    for arg in output_type:
        prefix, shape = parse_type_declaration(arg.typename)
        out_prefix.append(prefix)
        out_shapes.append(shape)

    input_pure_vector = sum([prefix != "vector" for prefix in in_prefix]) == 0
    output_pure_vector = sum([prefix != "vector" for prefix in out_prefix]) == 0
    if input_pure_vector and output_pure_vector:
        input_dims = [sum(list(shape)) for shape in in_shapes]
        output_dims = [sum(list(shape)) for shape in out_shapes]

        return MLPCaster(input_dims, output_dims)

    raise NotImplementedError("failed to infer the caster type")


class MLPFiller(nn.Module):
    def __init__(self, input_types : List[TypeBase], out_type : Union[TypeBase, List[TypeBase]], net : nn.Module):
        super().__init__()
        self.input_types = input_types
        self.out_types = out_type
        self.net = net
    
    @property
    def singular(self): return len(self.out_types) == 1

    def forward(self, *args):
        neural_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor): neural_args.append(arg)
            else: neural_args.append(torch.tensor(arg))
        cat_args = torch.cat([arg.reshape([1,-1]) for arg in neural_args], dim = -1)
        output = self.net(cat_args).reshape([-1])
        return Value(self.out_types, output)

def type_dim(tp : TypeBase):
    if tp.typename in ["int", "float", "boolean", "bool"]: return 1
    if "vector" in tp.typename:
        dim = 1        
        for d in [int(x) for x in re.findall(r'\d+', tp.typename)]: dim *= d
        return dim
    raise NotImplementedError(f"dim of type {tp} cannot be inferred")

def fill_hole(arg_types : List[TypeBase], out_type : TypeBase) -> nn.Module:
    in_dim = sum([type_dim(tp) for tp in arg_types])
    out_dim = type_dim(out_type)
    net = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim)
        )
    filler = MLPFiller(arg_types, out_type, net)
    return filler

