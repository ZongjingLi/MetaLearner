# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-03-16 22:28:20
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-03-16 22:41:56
import torch
import torch.nn as nn
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import functools

from .symbolic import Expression, FunctionApplicationExpression, ConstantExpression
from abc import abstractmethod

from helchriss.logger import get_logger
from helchriss.dsl.dsl_types import FuncType
from helchriss.dsl.dsl_values import Value, ProbValue

from helchriss.dsl.dsl_types import TypeBase, ListType, TupleType
logger = get_logger(__file__)

class FunctionExecutor(nn.Module):
    def __init__(self, domain : 'Domain' = None, concept_dim = 128):
        super().__init__()
        if domain is None:
            logger.warning("The input domain is empty, creating an empty domain")
            self.parser = Expression # Expression class allows prasing, but other parsers should be allowed
            self._grounding = None
            return
        
        self._domain : 'Domain' = domain
        self._domain = domain
        self.concept_dim = concept_dim

        self.parser = Expression # Expression class allows prasing, but other parsers should be allowed
        self._function_registry = dict() # allowed neural registry

        self.function_output_type = {}
        self.function_input_types = {}

        for function_name, function in domain.functions.items():

            if hasattr(self, function_name):
                self.register_function(function_name, self.unwrap_values(getattr(self, function_name)))
                logger.info('Function {} automatically registered.'.format(function_name))


            self.function_output_type[function_name] = function["type"]
            self.function_input_types[function_name] = [arg.split("-")[-1] for arg in function["parameters"]]


        self._grounding = None
    
    @property
    def domain(self) -> 'Domain' : return self._domain

    @property
    def types(self) : return self.domain.types
    
    def register_function(self, name: str, func: Callable):
        """Register an implementation for a function.

        Args:
            name: the name of the function.
            func: the implementation of the function.
        """
        self._function_registry[name] = func

    def get_function_implementation(self, name: str) -> Callable:
        """Get the implementation of a function. When the executor does not have an implementation for the function,
        the implementation of the function in the domain will be returned. If that is also None, a `KeyError` will be
        raised.

        Args:
            name: the name of the function.

        Returns:
            the implementation of the function.
        """

        if name in self._function_registry:
            return self._function_registry[name]
        raise KeyError(f'No implementation for function {name}.')

    def init_domain_functions(self, domain):
        return 

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

    def parse_expression(self, expr):
        """Parse the expression from a program string
        Args:
            expr: the string of the program expression
        Returns:
            the parsed expression
        """
        if self.parser is None: raise Exception("Parser is not availale")
        return self.parser.parse_program_string(expr)
    
    def evaluate(self, expression, grounding):
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)

        grounding = grounding if self._grounding is not None else grounding

        with self.with_grounding(grounding):
            return self._evaluate(expression)

    
    def _evaluate(self, expr : Expression):
        """Internal implementation of the executor. This method will be called by the public method :meth:`execute`.
        This function basically implements a depth-first search on the expression tree.

        Args:
            expr: the expression to execute.

        Returns:
            The result of the execution.
        """

        if isinstance(expr, FunctionApplicationExpression):
            func = self._function_registry[expr.func.name]
            args = [self._evaluate(arg) for arg in expr.args]
            return func(*args)
        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')

    def unwrap_values(self, func_or_ftype: Callable) -> Callable:
        """A function decorator that automatically unwraps the values of the arguments of the function.
        Basically, this decorator will unwrap the values of the arguments of the function, and then wrap the result with the
        :class:`concepts.dsl.value.Value` class.

        There are two ways to use this decorator. The first way is to use it as a function decorator:
        In this case, the wrapped function should have the same name as the DSL function it implements.

            >>> domain = FunctionDomain()
            >>> # Assume domain has a function named "add" with two arguments.
            >>> executor = FunctionDomainExecutor(domain)
            >>> @executor.unwrap_values
            >>> def add(a, b):
            >>>     return a + b
            >>> executor.register_function('add', add)

        The second way is to use it as function that generates a function decorator:

            >>> domain = FunctionDomain()
            >>> # Assume domain has a function named "add" with two arguments.
            >>> executor = FunctionDomainExecutor(domain)
            >>> @executor.unwrap_values(domain.functions['add'].ftype)
            >>> def add(a, b):
            >>>     return a + b
            >>> executor.register_function('add', executor.unwrap_values(add))

        Args:
            func_or_ftype: the function to wrap, or the function type of the function to wrap.

        Returns:
            The decorated function or a function decorator.
        """
        FunctionType = CentralExecutor # TODO: remove this

        if isinstance(func_or_ftype, FunctionType):
            ftype = func_or_ftype
        else:
            if func_or_ftype.__name__ not in self.domain.functions:
                raise NameError(f'Function {func_or_ftype.__name__} is not registered in the domain.')

            func_dict = self.domain.functions[func_or_ftype.__name__]

            ftype = FuncType(func_dict["parameters"], func_dict["type"])

        def wrapper(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                args = [arg.value if isinstance(arg, Value) else arg for arg in args]
                kwargs = {k: v.value if isinstance(v, Value) else v for k, v in kwargs.items()}
                rv = func(*args, **kwargs)

                if isinstance(ftype.return_type, tuple):
                    return tuple(
                        Value(ftype.return_type[i], rv[i]) if not isinstance(rv[i], Value) else rv[i] for i in range(len(rv))
                    )
                elif isinstance(rv, Value):
                    return rv
                else:
                    return Value(ftype.return_type, rv)
            return wrapped

        if isinstance(func_or_ftype, FunctionType):
            return wrapper
        else:
            return wrapper(func_or_ftype)
        

class TensorStateExecutor(FunctionExecutor):
    def __init__(self, domain, concept_dim = 128):
        super().__init__(domain, concept_dim = concept_dim)


class CentralExecutor(FunctionExecutor):
    def __init__(self, domain, concept_dim = 128):
        super().__init__(domain, concept_dim = concept_dim)

