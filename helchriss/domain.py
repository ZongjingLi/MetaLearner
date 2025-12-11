
import os
from lark import Lark, Tree, Transformer, v_args
from typing import Set, Tuple, Dict, List, Sequence, Union, Any, Optional
from .knowledge import State, Precondition, Effect, Action
from .types import baseType
from lark import Lark, Tree, Transformer, v_args
from typing import Callable, Any, List, Dict, Tuple, Optional, Union

from helchriss.dsl.dsl_types import (
    TypeBase, UnionType, ObjectType, SequenceType, TupleType,
    UniformSequenceType, ListType, FixedListType, VectorType, FunctionType as BaseFunctionType,
    AnyType, AutoType, DependentFunction, INT, FLOAT, BOOL, TYPE, ArrowType, EmbeddingType, STR
)
from helchriss.dsl.dsl_values import Value


class Domain:
    def __init__(self, grammar_file: Optional[str] = None):
        self.lark = self._load_grammar(grammar_file) if grammar_file else None
        
        self.domain_name: Optional[str] = None
        self.type_aliases: Dict[str, Tuple[List[Tuple[str, TypeBase]], TypeBase]] = {} 
        self.constants: Dict[str, Tuple[Any, TypeBase]] = {} 
        self.structures: List[Tuple[str, List[Tuple[str, TypeBase]], List[Tuple[str, TypeBase]]]] = [] 
        self.functions: Dict[str, DependentFunction] = {}
        self.verbose = 0

    def _load_grammar(self, grammar_file: str) -> Lark:
        """load the functional grammar file"""
        with open(grammar_file, 'r') as f:
            return Lark(f.read(), start='start', parser='lalr')

    def parse(self, code: str) -> None:
        tree = self.lark.parse(code)
        transformer = LKTransformer(self)
        transformer.transform(tree)

    def define_type(self, alias_name: str, params: List[Tuple[str, TypeBase]], actual_type: TypeBase) -> None:
        """define a type alias with parameters"""
        self.type_aliases[alias_name] = (params, actual_type)
        if self.verbose:print(f"define the type name alias : {alias_name}{{{', '.join(f'{n} : {t}' for n,t in params)}}} - {actual_type}")

    def define_structure(self, struct_name: str, params: List[Tuple[str, TypeBase]], fields: List[Tuple[str, TypeBase]]) -> None:
        """define structures as dependent product"""
        self.structures.append((struct_name, params, fields))
        if self.verbose:print(f"define structure of : {struct_name}{{{', '.join(f'{n} : {t}' for n,t in params)}}} containing: {fields}")

    def define_function(self, func_name: str, dep_params: List[Tuple[str, TypeBase]], args: List[Tuple[str, TypeBase]], ret_type: TypeBase, impl: str) -> None:
        """define funtctions"""

        dep_func = DependentFunction(dep_params, args, ret_type, impl)
        self.functions[func_name] = dep_func
        if self.verbose:print(f"define function: {func_name}{{{', '.join(f'{n} : {t}' for n,t in dep_params)}}}({', '.join(f'{n}:{t}' for n,t in args)}) -> {ret_type}")

    def print_summary(self) -> None:
        print(f"\nDomain: {self.domain_name}")
        print("types:")
        for alias, (params, typ) in self.type_aliases.items():

            print(f"  {alias}{{{', '.join(f'{n} : {t}' for n,t in params)}}} -> {typ}")
        print("structure:")
        for name, params, fields in self.structures:
            print(f"  {name}{{{params}}} := {fields}")
        print("functions:")
        for (name,function) in self.functions.items():
            print(f"  {name} {function}")


class LKTransformer(Transformer):
    def __init__(self, domain: Domain):
        if isinstance(domain, str) : domain = Domain("helchriss/dlt.grammar")
        self.domain = domain
        self.type_map = {
            "int": INT,
            "float": FLOAT,
            "boolean": BOOL,
            "Type": TYPE,
            "str" : STR
        }

    def IDENTIFIER(self, args): return str(args)


    def domain_name(self, args):
        return str(args[0])

    def domain_definition(self, args) -> None:
        self.domain.domain_name = str(args[0])

    def type_definitions(self, args) -> None:
        for def_tree in args:
            alias_name, params, actual_type = def_tree
            self.domain.define_type(alias_name, params, actual_type)

    def type_definition(self, args) -> Tuple[str, List[Tuple[str, TypeBase]], TypeBase]:
        if len(args) == 2:  # no parameters（as u_expr - u_var -> bool）
            alias_name, actual_type = args
            return (alias_name, [], actual_type)
        else:  # parametric type（as q_var{dim:int} - Vector[float, dim]）
            alias_name, params, actual_type = args
            return (alias_name, params, actual_type)

    def parametric_type_alias(self, args) -> Tuple[str, List[Tuple[str, TypeBase]], TypeBase]:
        alias_name, params, actual_type = args
        return (alias_name, params, actual_type)

    def type_parameters(self, args) -> List[Tuple[str, TypeBase]]:
        args = args[0]
        params = args[:-1] # var names
        rtype = args[-1]   # vtype
        return [(param, rtype) for param in params]
    
    def type_parameter(self, args): return args

    def base_type(self, args):
        type_name = str(args[0])
        if type_name in self.type_map: return self.type_map[type_name]
        if type_name in self.domain.type_aliases: return self.domain.type_aliases[type_name][1]
        return TypeBase(type_name)

    def dimension(self, args): return str(args[0])

    def TYPE_KIND(self, args) -> TypeBase: return self.type_map[str(args)]

    def type_expression(self, args):
        """
        type_expression: base_type
               | type_expression "->" type_expression  // Function type (e.g., A -> B)
               | "Vector" "[" type_expression "," dimension "]"  // Vector type (e.g., Vector[float, dim])
               | "List" "[" dimension "," type_expression "]"  // List type (e.g., List[n, u_var])
               | "Tuple" "[" type_expression ("," type_expression)+ "]"  // Tuple type (e.g., Tuple[u_var, boolean])
               | "(" type_expression ")"  // Parenthesized type
        """

        if len(args) == 1 and isinstance(args[0], TypeBase):  return args[0]
        if len(args) == 3:  return args[1] # braket
        if len(args) == 2: return ArrowType(args[0], args[1])#BaseFunctionType()
        raise NotImplementedError(f"failed to parse the type expression : {args}")
    
    def universe_type(self, args): return TYPE

    def vector_type(self, args): return VectorType(args[0], args[1])

    def embedding_type(self, args): 
        return EmbeddingType(args[0], args[1])

    def embedding_name(self, args): return args[0]
 
    def list_type(self, args):
        if len(args) == 2:
            return FixedListType(args[0], args[1])
        if len(args) == 1: return ListType(args[0])

    def tuple_type(self, args): return TupleType(args)

    def actual_type(self, args) -> TypeBase:
        if len(args) == 1 and isinstance(args[0], TypeBase): return args[0]
        else: raise ValueError(f"Failed to parse args: {args}")

    def alias_name(self, args): return str(args[0])

    def structure_definitions(self, args) -> None:
        for struct_tree in args:
            struct_name, params, fields = struct_tree
            self.domain.define_structure(struct_name, params, fields)

    def structure_definition(self, args) -> Tuple[str, List[Tuple[str, TypeBase]], List[Tuple[str, TypeBase]]]:
        struct_name, params, fields = args
        return (struct_name, params, fields)

    def function_definitions(self, args) -> None:
        for func_tree in args:
            func_name, dep_params, args_list, ret_type, impl = func_tree
            self.domain.define_function(func_name, dep_params, args_list, ret_type, impl)

    def function_definition(self, args) -> Tuple[str, List[Tuple[str, TypeBase]], List[Tuple[str, TypeBase]], TypeBase, str]:
        if len(args) == 4: args.insert(1,[])
        func_name = str(args[0])
        dep_params = args[1]
        alias_args_list = args[2]
        actual_args_list = []
        for alias in alias_args_list:

            if str(alias[1]) in self.domain.type_aliases:
                #print(alias[1], self.domain.type_aliases[str(alias[1])])
                actual_args_list.append([alias[0], self.domain.type_aliases[str(alias[1])][1] ])
                #print(alias[0], self.domain.type_aliases[str(alias[1])][1]   )
            else: actual_args_list.append(alias)

        ret_type = args[3]
        if str(ret_type) in self.domain.type_aliases:
            ret_type = self.domain.type_aliases[str(ret_type)][1]
        else: ret_type = ret_type

        impl = str(args[-1])        
        return (func_name, dep_params, actual_args_list, ret_type, impl)

    def dependent_types(self, args): return args
    
    def dependent_param(self, args): return args
    
    def typed_args(self, args):
        params = []
        for arg in args:
            for bind in arg: params.append(bind)
        return params

    def typed_arg(self, args): 
        elem_type = args[-1]
        var_names = args[:-1]
        return [(var, elem_type) for var in var_names]

    def CODE_BLOCK(self, args): return args

    def implementation(self, args): return args[0]


_icc_parser = Domain("helchriss/dlt.grammar")

def load_domain_string(domain_string, default_parser = _icc_parser):

    tree = default_parser.lark.parse(domain_string)
    lk_transformer = LKTransformer("helchriss/dlt.grammar")
    lk_transformer.transform(tree)
    return lk_transformer.domain