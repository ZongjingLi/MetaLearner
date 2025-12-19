import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import ListType, TupleType
from helchriss.knowledge.symbolic import FunctionApplicationExpression, VariableExpression
first_order_logic_domain_str = """
(domain :: Create)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    Object - Embedding[object, 256] ;; the type of certain object
    Expr - str
    Index - int
)
(def function
    create (x : List[Object])(y :List[Expr]) : List[Object] := by pass
    isinstance (x : List[Object]) (y : List[Expr]) :float := by pass
)
"""

create_domain = load_domain_string(first_order_logic_domain_str)
#fol_domain.print_summary()

class CreateExecutor(CentralExecutor):
    """extracts objects tagged in the grounding and implement the logic inference module recurrsively"""
    def __init__(self, domain):
        super().__init__(domain)
    
    def create(self, objects, relations):
        # sort the input arguments by type and arity into objects, relations of different arity

        # call the constraint energy landscape for each relation called
        return

    def isinstance(self, objects, relations):
        return



create_executor = CreateExecutor(create_domain)
