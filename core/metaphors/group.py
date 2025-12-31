'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-23 00:19:33
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-03-23 00:19:35
 # @Description:
'''
import torch
import torch.nn as nn
from helchriss.domain import Domain
from helchriss.knowledge.executor import CentralExecutor
from helchriss.dsl.dsl_values import Value, value_types
from helchriss.dsl.dsl_types import TypeBase, AnyType, INT, FLOAT, TupleType

from typing import List, Union, Mapping, Dict, Any, Tuple, Callable

from pathlib import Path
import dill
import inspect
from dataclasses import dataclass
from .types import ConvexConstruct

@dataclass
class FunctionSignature:
    fn          : str
    domain      : str # (domain) namespace of the function
    input_types : List[TypeBase]
    output_type : List[TypeBase]

    def __str__(self) -> str:
        return f"Function {self.fn} at {self.domain} ({self.input_types}) -> {self.output_type}"


class ExecutorGroup(CentralExecutor):
    """Storage of Domain Executors or some Extention Functions"""
    def __init__(self, domains : List[Union[CentralExecutor, Domain]]):
        super().__init__(None)  

        self.executor_group = nn.ModuleList([])
        for domain_executor in domains:
            if isinstance(domain_executor, CentralExecutor):
                executor = domain_executor
            elif isinstance(domain_executor, Domain):
                executor = CentralExecutor(domain_executor)
            else : raise Exception(f"input {domain_executor} is not a Domain or FunctionExecutor")
            executor.refs["executor_parent"] = self
            self.executor_group.append(executor)

        self.extended_registry = nn.ModuleDict({})

    
    def save_ckpt(self, path: str):
        path = Path(path)
        (path / "domains").mkdir(parents=True, exist_ok=True)
        (path / "extended").mkdir(parents=True, exist_ok=True)
        
        for executor in self.executor_group:
            torch.save(executor.state_dict(), path / "domains" / f"{executor.domain.domain_name}.pth")
        
        for name, module in self.extended_registry.items():
            #print(module)
            #torch.save(module, path / "extended" / f"{name}.ckpt")

            with open(path / "extended" / f"{name}.ckpt", "wb") as f:
                dill.dump(module, f)


    def load_ckpt(self, path: str):
        path = Path(path)
        for executor_file in (path / "domains").glob("*.pth"):
            name = executor_file.stem
            for executor in self.executor_group:
                if executor.domain.domain_name == name:
                    executor.load_state_dict(torch.load(executor_file,  weights_only=True))
                    break

        for extended_file in (path / "extended").glob("*.ckpt"):
            name = extended_file.stem
            with open(extended_file, "rb") as f:
                self.extended_registry[name] = dill.load(f)



    def format(self, function : str, domain : str) -> str: return f"{function}:{domain}"

    @staticmethod
    def signature(function : str, types : List[TypeBase]):
        typenames = [f"{tp.typename}" for tp in types]
        type_sign = "->".join(typenames)
        return f"{function}#{type_sign}"

    @staticmethod
    def parse_signature(signature: str) ->Tuple[str, List[TypeBase],TypeBase]:
        parts = signature.split('#')
        if len(parts) != 2: raise ValueError(f"Invalid signature format: {signature}")
    
        function_name = parts[0]
        type_signature = parts[1]
    
        type_specs = type_signature.split('->')
        all_types = []
        
        for type_spec in type_specs:
            type_parts = type_spec.split('-')
            if len(type_parts) != 1: raise ValueError(f"Invalid type specification: {type_spec}")
        
            typename = type_parts[0]
            all_types.append(TypeBase(typename))
    
        input_types = all_types[:-1]


        output_types = all_types[-1]  # Return as list as requested
        return function_name, input_types, output_types

    def domain_function(self, func : str) -> bool: return ":" in func

    def freeze_extended(self, freeze = True):
        for param in self.extended_registry.parameters():
            param.requires_grad = freeze

    @property
    def functions(self) -> List[Tuple[str, List[TypeBase], TypeBase]]:
        functions = []
        for sign in self.extended_registry:
            f_sign, in_types, out_type = self.parse_signature(sign)

            functions.append([f_sign, in_types, out_type])
        for executor in self.executor_group:
            assert isinstance(executor, CentralExecutor), f"{executor} is not a executor"
            for func_name in executor._function_registry:

                functions.append([
                    self.format(func_name,executor.domain.domain_name),
                    executor.function_input_types[func_name],
                    executor.function_output_type[func_name]])

        return functions

    def function_signature(self, func : str) -> List[Tuple[List[TypeBase], TypeBase]]:
        hyp_sign = []
        for function in self.functions:
            f_sign, in_types, out_type = function
            if func == f_sign:
                hyp_sign.append([in_types, out_type])
        return hyp_sign
        
    def gather_functions(self, input_types, output_type : Union[TypeBase, bool]) -> List[str]:
        compatible_fns = []
        for function in self.functions:
            fn_sign, in_types, out_type = function

            if in_types == input_types and out_type == output_type:


                compatible_fns.append(fn_sign)
        return compatible_fns
    
    def get_functions(self, input_types, output_type):
        compatible_fns = []
        for function in self.functions:
            fn, in_types, out_type = function
            fn_signature = self.signature(fn, in_types)
            if isinstance(in_types, List): in_types = TupleType(in_types)
            
            if in_types == input_types and out_type == output_type:
                print(fn, "In Group Executor")
                print("CMP:",in_types , input_types)
                print("CMP:",out_type , output_type)
                compatible_fns.append(self.get_function_call(fn_signature))
        return compatible_fns

    def register_extended_function(self, func : str, in_types : List[TypeBase], out_type : TypeBase, implement : nn.Module):
        signature = self.signature(func, in_types + [out_type])
        self.extended_registry[signature] = implement

    def infer_domain(self, func: str) -> str:
        for executor in self.executor_group: 
            assert isinstance(executor, CentralExecutor), "not a function executor"
            if func in executor._function_registry: return executor.domain.domain_name

    def get_function_call(self, signature: str, grounding = {}):
        fn_call = None
        func, _, _ = self.parse_signature(signature)
        # 1) check if this is a domain function
        if self.domain_function(func):
            func_name, domain_name = func.split(":")
            for executor in self.executor_group:
                assert isinstance(executor, CentralExecutor), f"{executor} is not a executor"
                if executor.domain.domain_name == domain_name: # find the correct executor
                    executor._grounding = grounding
                    fn_call = executor._function_registry[func_name]


        # 2) check if this is an extention function
        for sign in self.extended_registry:
            if signature in sign:fn_call = self.extended_registry[sign]
        return fn_call

    def execute(self, func : str, args : List[Value], arg_types : List[TypeBase],  grounding = None) -> Value:
        self._grounding = grounding
        """a function could be in some domain or in extention registry"""
        arg_types = value_types(args)
        signature = self.signature(func, arg_types)

        func_call = self.get_function_call(signature, grounding)

        ### collect arguments and evaluate on the function call
        args = [arg.value for arg in args]
        kwargs = {"arg_types" : arg_types}
        
        sig = inspect.signature(func_call)
        has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD 
                     for param in sig.parameters.values())

        has_kwargs = has_kwargs and not isinstance(func_call, ConvexConstruct)
        if func_call is not None:
            if has_kwargs: return func_call(*args, **kwargs)
            else: return func_call(*args)
        else:raise ModuleNotFoundError(f"{signature} is not found.")

