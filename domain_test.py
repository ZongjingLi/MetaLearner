# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 06:22:49
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-09 05:05:10
domain_str = """
(domain Contact)
(:type
    state - vector[float, 256]        ;; [x, y] coordinates
)
(:predicate
    ;; Basic position predicate
    ref ?x-state -> boolean
    get_position ?x-state -> vector[float, 2]
    
    ;; Qualitative distance predicates
    contact ?x-state ?y-state -> boolean
)
"""
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string, Domain
from domains.utils import domain_parser

contact_dom = load_domain_string(domain_str, domain_parser)


contact_dom.print_summary()

contact_executor = CentralExecutor(contact_dom, "cone", 256)

from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector
state_type = tvector(treal, 2)  # 2D position vector
position_type = tvector(treal, 2)  # 2D position vector
distance_type = treal  # scalar distance

#contact_executor.update_registry({
#    "ref": Primitive("ref",arrow(state_type, position_type), lambda x: {**x, "end": torch.tensor([1.0, 1.0, 0.0])})
#    })

import torch

state = torch.randn([3,256])

gt = torch.tensor([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
    ]).float()

gt = torch.tensor([1.0, 0.0, 1.0])

context = {0:{"state": state, "end" : 1.0}, 1:{"state": state, "end" : 1.0}}


optimizer = torch.optim.Adam(contact_executor.parameters(), lr=0.01)  # Optimizing "end" predictions
loss_fn = torch.nn.BCEWithLogitsLoss() 

num_epochs = 100  # Set optimization iterations
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients

    # Compute loss
    #res = contact_executor.evaluate("(contact $0 $1)", context)
    res = contact_executor.evaluate("(ref $0)", context)

    loss = loss_fn(res["end"], gt)  # Compare model predictions with ground truth

    # Backpropagation
    loss.backward()
    optimizer.step()  # Update model output

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# ----- Final Optimized Output -----
print("Optimized Predictions (res['end']):")
print(res["end"].detach()) 


res = contact_executor.evaluate("(get_position $0)", context)

print(res)