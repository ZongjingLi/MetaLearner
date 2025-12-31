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
    Object - Embedding[object, 96] ;; the type of certain object
    Expr - str
    ObjSet - List[Tuple[boolean,Embedding[object, 96]]]
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    exists (x : ObjSet) : boolean := by pass
    forall (x : ObjSet) : boolean := by pass
    iota   (x : ObjSet) : ObjSet := by pass
    filter (x : ObjSet) (y : Expr) : ObjSet := by pass
    relate (x y : ObjSet) (y : Expr) : ObjSet := by pass

    negate (x : boolean) : boolean := by pass
    logic_and (x y : boolean) : boolean := by pass
    logic_or  (x y : boolean) : boolean := by pass

    count (x : ObjSet) : integer := by pass
    scene : ObjSet := by pass
)
"""

fol_domain = load_domain_string(first_order_logic_domain_str)
#fol_domain.print_summary()

def merge_paths(paths, sub_paths):

    if sub_paths["nodes"]:
        print(list(node["id"] for node in sub_paths["nodes"]))
        min_id = min(node["id"] for node in sub_paths["nodes"])

        sub_paths["edges"].append((min_id, "init", {"type": "init_connection"}))

    existing_ids = {n['id'] for n in paths['nodes']}
    id_map, new_nodes = {}, []
    for node in sub_paths['nodes']:
        orig_id, new_id = node['id'], node['id']
        i = 0
        while new_id in existing_ids:
            prefix = orig_id.rstrip('0123456789')
            num = int(orig_id[len(prefix):]) if orig_id[len(prefix):] else 0
            new_id = f"{prefix}{num + len(existing_ids) + i}"
            i += 1
        id_map[orig_id] = new_id
        new_nodes.append({**node, 'id': new_id})
        existing_ids.add(new_id)
    

    paths['nodes'].extend(new_nodes)
    paths['edges'].extend([(id_map.get(a,a), id_map.get(b,b), d) for a,b,d in sub_paths['edges']])


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

        paths = {"nodes":[{"id": "init","fn":"init_fn", "value":"pseudo","type":"pseudo type", "weight": 1.0}], "edges":[]}

        vtp = kwargs["arg_types"][0]
        assert isinstance(vtp, ListType), f"{vtp}"
        assert isinstance(vtp.element_type, TupleType), f"{vtp.element_type}"
        obj_tp = vtp.element_type.element_types[1]

        var_logit, obj = vars[:,:1], vars[:,1:]
        if len(obj.shape) == 1: obj = obj[None,...]

        logic_expr = FunctionApplicationExpression(VariableExpression(expr), [ VariableExpression(Value(obj_tp,obj)) ] )
        class_logit, subloss, son_id, sub_paths = ancestor_executor._evaluate(logic_expr)
        merge_paths(paths, sub_paths)
    
        if not isinstance(subloss, torch.Tensor): subloss = torch.tensor(subloss)
        edge_info = (node_id, son_id, {"weight":float(torch.exp(-subloss) )})
        ancestor_executor.eval_info["tree"]["edges"].append(edge_info)

        logits.append(torch.min(class_logit.value, var_logit))
        objects.append(obj)
        local_loss += subloss
        """after the process of batchwise"""

        logits = torch.stack(logits).reshape([-1,1])
        objects = torch.cat(objects, dim = 0)


        reference_set = torch.cat([logits, objects], dim = 1)
        output, _, paths = reference_set, 0.0, paths

        """add the edge node and eval node"""
        node_info = {"id":node_id, "fn" : fn, "value": str(output), "type": "List"}
        ancestor_executor.eval_info["tree"]["nodes"].append(node_info)
        ancestor_executor.eval_info["paths"][f"{node_id}"] = paths # no rewrite 
        ancestor_executor.prev_node = node_info

        return reference_set#, local_loss
    

    def relate(self, anchor_vars, ref_vars, expr, **kwargs):
        logits = []     # logits of reference in the scene.
        relations = []  # 
        local_loss = 0. # the loss by executing the function

        n, m = anchor_vars.size(0), ref_vars.size(0)
        ancestor_executor = self.ancestor_executor()
        node_id = f"node{ancestor_executor.node_count}"
        fn = "Tensor"

        for anchor_var in anchor_vars:
            for ref_var in ref_vars:
                vtp = kwargs["arg_types"][0]
                assert isinstance(vtp, ListType), f"{vtp}"
                assert isinstance(vtp.element_type, TupleType), f"{vtp.element_type}"
                anchor_tp = vtp.element_type.element_types[1]
                refer_tp =  kwargs["arg_types"][1].element_type.element_types[1]


                anchor_var_logit, anchor_feature = anchor_var[:1], anchor_var[1:]
                refer_var_logit , refer_feature  = ref_var[:1],    ref_var[1:]

                relation_feature = torch.cat([anchor_feature, refer_feature], dim = -1)
                relation_type    = TupleType([anchor_tp, refer_tp])
    
                if len(relation_feature.shape) == 1: relation_feature = relation_feature[None,...]

                logic_expr = FunctionApplicationExpression(VariableExpression(expr), 
                                        [ VariableExpression(Value(anchor_tp,anchor_feature)),
                                          VariableExpression(Value(refer_tp,refer_feature)) ] )
                class_logit, subloss, son_id, paths = ancestor_executor._evaluate(logic_expr)


                #edge_info = (node_id, son_id, {"weight":float(torch.exp(torch.tensor(-subloss)) )})
                #ancestor_executor.eval_info["tree"]["edges"].append(edge_info)

                logits.append(class_logit.value )
                relations.append(relation_feature)
                local_loss += subloss


        matrix  = torch.stack(logits).reshape([n,m])
        #print(expr)
        #print((matrix > 0).detach().numpy())



        anchor_expanded = anchor_vars[:,0:1].T.repeat(n,1)  # [n,m] (repeat anchor logit for all j)
        ref_expanded = ref_vars[:,0:1].repeat(1,m)       # [n,m] (repeat reference logit for all i)

        joint_validity = torch.min(torch.min(anchor_expanded, matrix), ref_expanded)  # [n,m]
        
        logits = joint_validity.max(dim=1).values[..., None]  # [m,1]
        objects = ref_vars[:,1:]

        #print("anchor:",anchor_vars[:,0:1].flatten() > 0)
        #print("refere:", ref_vars[:,0:1].flatten() > 0)
        #print((logits > 0).detach().numpy())


        reference_set = torch.cat([logits, objects], dim = 1)
        output, _, paths = reference_set, 0.0, {"nodes":[], "edges":[]}

        """add the edge node and eval node"""
        #node_info = {"id":node_id, "fn" : fn, "value": str(output), "type": "List"}
        #ancestor_executor.eval_info["tree"]["nodes"].append(node_info)
        #ancestor_executor.eval_info["paths"][f"{node_id}"] = paths # no rewrite 
        #ancestor_executor.prev_node = node_info

        return reference_set#, local_loss

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
