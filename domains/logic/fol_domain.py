import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

first_order_logic_domain_str = """
(domain :: FirstOrderLogic)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    var{dim : int} - Vector[float, dim] ;; as a quantified variable
    u_expr{var : Type} - var -> boolean ;; given an unquantified variable output a boolean value
    b_expr{var : Type} - var -> var -> boolean ;; arrow helps to make complex type A \to B make a complex type
    object - Tuple[boolean, Embedding[object, 64]]
    shape  - Tuple[boolean, Embedding[shape, 32]]
    color - Embedding[color, 3]
    
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    exists (x : List[object]) : boolean := by pass
    forall (x : List[object]) : boolean := by pass
    iota   (x : List[object]) : List[object] := by pass

    negate (x : boolean) : boolean := by pass
    logic_and (x y : boolean) : boolean := by pass
    logic_or  (x y : boolean) : boolean := by pass

    assert {var : Type} (x : var) (y : var -> boolean) : boolean := by pass
    
    count (x : List[object]) : integer := by pass
    scene : List[object] := by pass

    red      (x : List[object]) : List[object] := by pass
    green    (x : List[object]) : List[object] := by pass
    blue     (x : List[object]) : List[object] := by pass
    
    circle   (x : List[shape]) : List[shape] := by pass
    square   (x : List[shape]) : List[shape] := by pass
    triangle (x : List[shape]) : List[shape] := by pass
)
"""

from helchriss.dklearn.nn.mlp import FCBlock

fol_domain = load_domain_string(first_order_logic_domain_str)
fol_domain.print_summary()

class FOLExecutor(CentralExecutor):

    def __init__(self, domain):
        super().__init__(domain)
        self.red_mlp    = FCBlock(64,2,64, 1)
        self.green_mlp  = FCBlock(64,2,64, 1)
        self.blue_mlp   = FCBlock(64,2,64, 1)

    def scene(self): return self.grounding["objects"]

    def exists(self, objects): return torch.max(objects[: ,0])

    def forall(self, objects): return torch.min(objects[:, 0])
    
    def iota(self, objects): return torch.cat([
        torch.logit(torch.softmax(objects[:, 0], dim = 0).reshape([-1,1])),
          objects[:,1:]
    ], dim = -1)

    def negate(self, logit): return -logit

    def logic_and(self, logit1, logit2): return torch.min(logit1, logit2)

    def logic_or(self, logit1, logit2): return torch.max(logit1, logit2)

    def count(self, objects): return torch.sum(torch.sigmoid(objects[:,0]))

    def red(self, objects):
        red_logits = self.red_mlp(objects[:,1:]).reshape([-1])
        return torch.cat([torch.min(red_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

    def green(self, objects):
        green_logits = self.green_mlp(objects[:,1:]).reshape([-1])
        return torch.cat([torch.min(green_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

    def blue(self, objects):
        blue_logits = self.blue_mlp(objects[:,1:]).reshape([-1])
        return torch.cat([torch.min(blue_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

fol_executor = FOLExecutor(fol_domain)
