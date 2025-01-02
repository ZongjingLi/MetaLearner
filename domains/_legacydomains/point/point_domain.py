
from rinarak.knowledge.executor import CentralExecutor
from domains.utils import domain_parser, load_domain_string, build_domain_dag

domain_string_point = """
(domain Point)
(:type
    state - vector[float,3]
    position - vector[float, 3]
)
(:predicate
    is-large ?x-state -> boolean
    pos ?x-state -> position
    
    left ?x-pos ?y-pos -> boolean
    right ?x-pos ?y-pos -> boolean

    near ?x-pos ?y-pos -> boolean
    far ?x-pos ?y-pos -> boolean

    above ?x-state ?y-state -> boolean
    below ?x-state ?y-state -> boolean
)
(:action
    (
        name: transport
        parameters: ?o1 ?o2
        precondition: (true)
        effect: (
            assign (g ?o1) ?o2
        )
    )
)
"""

def left(A, B, threshold = 0.0, gamma = 1.0):
    n, d = A.shape
    n = A.shape[0]
    m = B.shape[0]
    A_expanded = A.unsqueeze(1).expand(n, m, d)  # Shape [n, b, 3]
    B_expanded = B.unsqueeze(0).expand(n, m, d)  # Shape [n, b, 3]

    delta = A_expanded - B_expanded  # Shape [n, b, 3]
    left_logits = (-delta[:, :, 0] - threshold ) / gamma  # Negative x-difference indicates A is to the left of B
    return left_logits

def right(A, B, threshold = 0.0, gamma = 1.0):
    n, d = A.shape
    n = A.shape[0]
    m = B.shape[0]
    A_expanded = A.unsqueeze(1).expand(n, m, d)  # Shape [n, b, 3]
    B_expanded = B.unsqueeze(0).expand(n, m, d)  # Shape [n, b, 3]

    delta = A_expanded - B_expanded  # Shape [n, b, 3]
    left_logits = (-delta[:, :, 0] - threshold ) / gamma  # Negative x-difference indicates A is to the left of B
    return left_logits

def far(A, B, threshold = 0.0, gamma = 1.0):
    n, d = A.shape
    n = A.shape[0]
    m = B.shape[0]
    A_expanded = A.unsqueeze(1).expand(n, m, d)  # Shape [n, b, 3]
    B_expanded = B.unsqueeze(0).expand(n, m, d)  # Shape [n, b, 3]
    delta = A_expanded - B_expanded  # Shape [n, b, 3]

    distances = torch.sqrt(torch.sum(delta ** 2, dim=2))
    far_logits = (distances - threshold)  # Positive distance is used as logits (farther = more positive)
    return far_logits

def near(A, B, threshold = 0.0, gamma = 1.0):
    n, d = A.shape
    n = A.shape[0]
    m = B.shape[0]
    A_expanded = A.unsqueeze(1).expand(n, m, d)  # Shape [n, b, 3]
    B_expanded = B.unsqueeze(0).expand(n, m, d)  # Shape [n, b, 3]
    delta = A_expanded - B_expanded  # Shape [n, b, 3]
    distances = torch.sqrt(torch.sum(delta ** 2, dim=2))
    near_logits = -distances + threshold
    return near_logits

def above(A, B, threshold = 0.0, gamma = 1.0):
    n, d = A.shape
    n = A.shape[0]
    m = B.shape[0]
    A_expanded = A.unsqueeze(1).expand(n, m, d)  # Shape [n, b, 3]
    B_expanded = B.unsqueeze(0).expand(n, m, d)  # Shape [n, b, 3]
    delta = A_expanded - B_expanded
    above_logits = delta[:, :, 2] - threshold 
    return above_logits

def below(A, B, threshold = 0.0, gamma = 1.0):
    n, d = A.shape
    n = A.shape[0]
    m = B.shape[0]
    A_expanded = A.unsqueeze(1).expand(n, m, d)  # Shape [n, b, 3]
    B_expanded = B.unsqueeze(0).expand(n, m, d)  # Shape [n, b, 3]
    delta = A_expanded - B_expanded
    below_logits = -delta[:, :, 2]  + threshold 
    return below_logits

"""a universal source domain for realization of rough spatial relations"""
domain_point = load_domain_string(domain_string_point, domain_parser)
predicate_graphs_point = build_domain_dag(domain_point)
point_executor = CentralExecutor(domain_point)
point_executor.redefine_predicate(
    "is-large", lambda x : {**x , "end": x["state"].norm(dim = -1) - 1.0})
point_executor.redefine_predicate(
    "left", lambda x: lambda y : {**x , "end": left(x["state"], y["state"]) })

point_executor.redefine_predicate(
    "right", lambda x: lambda y : {**x , "end": right(x["state"], y["state"]) })

point_executor.redefine_predicate(
    "far", lambda x: lambda y : {**x , "end": far(x["state"], y["state"]) })
point_executor.redefine_predicate(
    "near", lambda x: lambda y : {**x , "end": near(x["state"], y["state"]) })

point_executor.redefine_predicate(
    "above", lambda x: lambda y : {**x , "end": above(x["state"], y["state"]) })
point_executor.redefine_predicate(
    "below", lambda x: lambda y : {**x , "end": below(x["state"], y["state"]) })

def visualize_state(state, save_name = "temp"):
    return 0
