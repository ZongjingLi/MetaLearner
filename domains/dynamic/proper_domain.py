import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import ListType, TupleType
from helchriss.knowledge.symbolic import FunctionApplicationExpression, VariableExpression
first_order_logic_domain_str = """
(domain :: Tower)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    Object - Embedding[object, 64] ;; the type of certain object
    Expr - str
    ObjSet - List[Tuple[boolean,Embedding[object, 64]]]
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    exists (x : ObjSet) : boolean := by pass
    forall (x : ObjSet) : boolean := by pass
    iota   (x : ObjSet) : ObjSet := by pass
    filter (x : ObjSet) (y : Expr) : ObjSet := by pass

    negate (x : boolean) : boolean := by pass
    logic_and (x y : boolean) : boolean := by pass
    logic_or  (x y : boolean) : boolean := by pass

    count (x : ObjSet) : integer := by pass
    scene : ObjSet := by pass
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

        ancestor_executor = self.ancestor_executor()
        node_id = f"node{ancestor_executor.node_count}"
        fn = "Eval"

        for var in vars:
            vtp = kwargs["arg_types"][0]
            assert isinstance(vtp, ListType), f"{vtp}"
            assert isinstance(vtp.element_type, TupleType), f"{vtp.element_type}"
            obj_tp = vtp.element_type.element_types[1]

            var_logit, obj = var[:1], var[1:]
            if len(obj.shape) == 1: obj = obj[None,...]

            logic_expr = FunctionApplicationExpression(VariableExpression(expr), [ VariableExpression(Value(obj_tp,obj)) ] )
            class_logit, subloss, son_id, paths = ancestor_executor._evaluate(logic_expr)


            edge_info = (node_id, son_id, {"weight":float(torch.exp(torch.tensor(-subloss)) )})
            ancestor_executor.eval_info["tree"]["edges"].append(edge_info)

            logits.append(torch.min(class_logit.value, var_logit))
            objects.append(obj)
            local_loss += subloss


        logits = torch.stack(logits)
        objects = torch.cat(objects, dim = 0)
        reference_set = torch.cat([logits, objects], dim = 1)
        output, _, paths = reference_set, 0.0, {"nodes":[], "edges":[]}

        """add the edge node and eval node"""
        node_info = {"id":node_id, "fn" : fn, "value": str(output), "type": "List"}
        ancestor_executor.eval_info["tree"]["nodes"].append(node_info)
        ancestor_executor.eval_info["paths"][f"{node_id}"] = paths # no rewrite 
        ancestor_executor.prev_node = node_info

        return reference_set#, local_loss
    

    def relate(self, anchor_vars, ref_vars, expr):
        logits = [] # logits of reference in the scene.
        objects = []
        local_loss = 0.

        reference_set = None
        return reference_set

    def exists(self, objects): return torch.max(objects[: ,0])

    def forall(self, objects): return torch.min(objects[:, 0])
    
    def iota(self, objects): return torch.cat([
        torch.logit(torch.softmax(objects[:, 0], dim = 0).reshape([-1,1])),
          objects[:,1:]
    ], dim = -1)

    def negate(self, logit): return -logit

    def logic_and(self, logit1, logit2): return torch.min(logit1, logit2)

    def logic_or(self, logit1, logit2): return torch.max(logit1, logit2)

    def count(self, objects):
        #print("logits",torch.sum(torch.sigmoid(objects[:,0])))
        return torch.sum(torch.sigmoid(objects[:,0]))


fol_executor = FOLExecutor(fol_domain)
