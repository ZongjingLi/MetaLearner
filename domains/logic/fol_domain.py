import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import ListType, TupleType
from helchriss.knowledge.symbolic import FunctionApplicationExpression, VariableExpression
first_order_logic_domain_str = """
(domain :: Logic)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    Object - Embedding[object, 64] ;; the type of certain object
    Expr - str
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    exists (x : List[Object]) : boolean := by pass
    forall (x : List[Object]) : boolean := by pass
    iota   (x : List[Object]) : List[Object] := by pass
    filter (x : List[Tuple[boolean,Object]]) (y : Expr) : List[Object] := by pass

    negate (x : boolean) : boolean := by pass
    logic_and (x y : boolean) : boolean := by pass
    logic_or  (x y : boolean) : boolean := by pass

    count (x : List[Object]) : integer := by pass
    scene : List[object] := by pass
)
"""

fol_domain = load_domain_string(first_order_logic_domain_str)
#fol_domain.print_summary()

class FOLExecutor(CentralExecutor):
    """extracts objects tagged in the grounding and implement the logic inference module recurrsively"""
    def __init__(self, domain):
        super().__init__(domain)

    def ancestor_executor(self):
        ancestor = self
        while (ancestor is not None) and ancestor.has_parent_executor() :
            ancestor = ancestor.parent_executor()
        return ancestor 

    def filter(self, vars, expr, **kwargs):
        logits = [] # logits of reference in the scene.
        objects = []
        local_loss = 0.

        for var in vars:
            vtp = kwargs["arg_types"][0]
            assert isinstance(vtp, ListType), f"{vtp}"
            assert isinstance(vtp.element_type, TupleType), f"{vtp.element_type}"
            obj_tp = vtp.element_type.element_types[1]

            var_logit, obj = var[:1], var[1:]
            if len(obj.shape) == 1: obj = obj[None,...]

            logic_expr = FunctionApplicationExpression(VariableExpression(expr), [ VariableExpression(Value(obj_tp,obj)) ] )

            class_logit, loss = self.ancestor_executor().evaluate(logic_expr, self.grounding)

            logits.append(torch.min(class_logit.value, var_logit))
            objects.append(obj)

            local_loss += loss
        logits = torch.stack(logits)
        objects = torch.cat(objects, dim = 0)

        reference_set = torch.cat([logits, objects], dim = 1)


        return reference_set#, local_loss
    
    def relate(self, anchor_vars, ref_vars, expr):
        return

    def exists(self, objects):

        return torch.max(objects[: ,0])

    def forall(self, objects): return torch.min(objects[:, 0])
    
    def iota(self, objects): return torch.cat([
        torch.logit(torch.softmax(objects[:, 0], dim = 0).reshape([-1,1])),
          objects[:,1:]
    ], dim = -1)

    def negate(self, logit): return -logit

    def logic_and(self, logit1, logit2): return torch.min(logit1, logit2)

    def logic_or(self, logit1, logit2): return torch.max(logit1, logit2)

    def count(self, objects): return torch.sum(torch.sigmoid(objects[:,0]))


fol_executor = FOLExecutor(fol_domain)
