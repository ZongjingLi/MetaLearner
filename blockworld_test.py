# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 06:22:49
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-19 20:23:27

blockworld_domain_str = """
(domain Blockworld)
(:type
    state - vector[float,3] ;; encoding of position and is holding
    position - vector[float,2]
)
(:predicate
    block_position ?x-state -> position
    on ?x-state ?y-state -> boolean
    clear-above ?x-state -> boolean
    holding ?x-state -> boolean
)
(: constant
    hand-free - boolean
)
(:action
    (
        name: pick
        parameters: ?o1
        precondition: (and (clear ?o1) (hand-free) )
        effect:
        (and-do
            (and-do
                (assign (holding ?o1) true)
                (assign (clear ?o1) false)
            )
            (assign (hand-free) false)
        )
    )
    (
        name: place
        parameters: ?o1 ?o2
        precondition:
            (and (holding ?o1) (clear ?o2))
        effect :
            (and-do
            (and-do
                        (assign (hand-free) true)
                (and-do
                        (assign (holding ?o1) false)
                    (and-do
                        (assign (clear ?o2) false)
                        (assign (clear ?o1) true)
                    )
                )
                
            )
                (assign (on ?x ?y) true)
            )
    )
)
"""
import open3d as o3d
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string, Domain
from domains.utils import domain_parser

blockworld_domain = load_domain_string(blockworld_domain_str, domain_parser)

blockworld_domain.print_summary()

blockworld_executor = CentralExecutor(blockworld_domain, concept_dim = 128)


if __name__ == "__main__":
    import torch
    state = torch.randn([5,3])

    gt = torch.tensor([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
        ]).float()
    #gt = torch.tensor([1.0, 0.0, 1.0])

    context = {0:{"state": state, "end" : 1.0}, 1:{"state": state, "end" : 1.0}}
    context["hand-free"] = 1.0
    

    res = blockworld_executor.evaluate("(block_position $0)", context)
    #res = blockworld_executor.evaluate("(hand-free)", context)
    print(res["end"])

    #from env.blockworld.blockworld_env import *

