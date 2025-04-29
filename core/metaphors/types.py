from abc import abstractmethod
from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
import re
from typing import List, Tuple, Any
import torch
import torch.nn as nn


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
    def cast(self, input):
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

def infer_caster(input_type : List[TypeBase], output_type : TypeBase):
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
        """TODO: ignore the individual arg separation"""
        return MLPCaster(input_dims, output_dims)
        return LinearCaster(input_dims, output_dims)

    return -1
