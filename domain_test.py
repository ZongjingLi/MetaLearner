# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 06:22:49
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-06 06:46:00
domain_str = """
(domain Contact)
(:type
    state - vector[float, 256]        ;; [x, y] coordinates
)
(:predicate
    ;; Basic position predicate
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

print(contact_executor)
import torch
state = torch.randn([3,256])

context = {0:{"state": state, "end" : 1.0}, 1:{"state": state, "end" : 1.0}}

contact_executor.evaluate("(contact $0 $1)", context)