import numpy as np
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

graph_domain_str = """
(domain Misc)
(:type
    loss - vector[float, 1]
)
(:predicate
    add_loss(x : loss) : float := by pass
    do_and (x y : Str) : boolean := by pass

    ;; action is a parameterized function that takes in several parameters and 
    ;; assign values to those parameters and a primitive sequence.

    execute (a : Action) : float := by pass
    execute (a : List[Action]) : float := by pass

    plan (s : State, g : Goal) : List[Action] := by pass

    ;; env is a function that takes a state and an action


    assert_rewrite (f : Str, x : Args, g : Str, y: Args) := by pass
)

"""

graph_domain = load_domain_string(graph_domain_str)

class MiscExecutor(CentralExecutor):
    def Id(self, x): return x

misc_executor = MiscExecutor(graph_domain)
