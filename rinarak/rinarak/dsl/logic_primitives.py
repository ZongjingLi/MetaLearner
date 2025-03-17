from .logic_types import *

infinity = 1e9
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# [Existianial quantification, exists, forall]
operator_exists = Primitive(
    "exists",
    arrow(fuzzy_set, boolean),
    lambda x:{**x,
    "end":torch.max(x["end"], dim = -1).values})

operator_forall = Primitive(
    "forall",
    arrow(fuzzy_set, boolean),
    lambda x:{**x,
    "end":torch.min(x["end"], dim = -1).values})

operator_equal_concept = Primitive(
    "equal_concept",
    arrow(fuzzy_set, fuzzy_set, boolean),
    "Not Implemented"
)

# make reference to the objects in the scene
operator_related_concept = Primitive(
    "relate",
    arrow(fuzzy_set, fuzzy_set, boolean),
    "Not Implemented"
)

def type_filter(objset,concept,exec):
    if concept in objset["context"]: return torch.min(objset["context"][concept], objset["end"])

    filter_logits = torch.zeros_like(objset["end"])
    parent_type = exec.get_type(concept)
    for candidate in exec.type_constraints[parent_type]:
        filter_logits += exec.entailment(objset["context"]["features"],
            exec.get_concept_embedding(candidate)).sigmoid()

    div = exec.entailment(objset["context"]["features"],
            exec.get_concept_embedding(concept)).sigmoid()

    filter_logits = logit(div / filter_logits)
   
    return torch.min(objset["end"],filter_logits)


# end points to train the clustering methods using uniform same or different.
operator_uniform_attribute = Primitive("uniform_attribute",
                                       arrow(fuzzy_set, boolean),
                                       "not")

operator_equal_attribute = Primitive("equal_attribute",
                                     arrow(fuzzy_set, boolean, boolean),
                                     "not"
                                     )

def condition_assign(x, y):
    """evaluate the expression x, y and return the end as an assignment operation"""
    return {**x, "end": [{"x": x["idx"], "v" : y["end"], "c": torch.tensor(infinity, device = device), "to": x["from"]}]}

operator_assign_attribute = Primitive("assign",arrow(boolean, boolean, boolean),
                                      lambda x: lambda y: condition_assign(x, y))

def condition_if(x, y):
    """x as the boolean expression to evaluate, y as the code blocks"""
    outputs = []
    for code in y["end"]:
        code_condition = code["c"] if isinstance(code["c"], torch.Tensor) else torch.tensor(code["c"])
        assign_operation = {
            "x": code["x"],
            "v": code["v"],
            "c": torch.min(code_condition, torch.max(x["end"])),
            "to": code["to"],
        }
        outputs.append(assign_operation)
    return {**x, **y, "end":outputs}

operator_if_condition = Primitive("if", arrow(boolean, boolean, boolean),
                                  lambda x: lambda y: condition_if(x, y))

operator_pi = Primitive("pi", arrow(boolean), {"end":torch.tensor(3.14), "set":torch.tensor(1.0)})

operator_true = Primitive("true", arrow(boolean), {"end":torch.tensor(14.), "set":torch.tensor(1.0)})

operator_true = Primitive("false", arrow(boolean), {"end":-1. * torch.tensor(14.), "set":torch.tensor(1.0)})