# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-09 04:56:16
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-21 12:15:48


import torch
import torch.nn as nn

from core.model import EnsembleModel, curriculum_learning
from core.metaphors.diagram_legacy import MetaphorMorphism
from core.curriculum import load_curriculum
from domains.generic.generic_domain import generic_executor
from domains.distance.distance_domain import distance_executor
from domains.direction.direction_domain import direction_executor
from config import config
from core.model import save_ensemble_model, load_ensemble_model
from domains.utils import load_domain_string, domain_parser
from rinarak.knowledge.executor import CentralExecutor

model = EnsembleModel(config)

domain_string = """
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

number_string = 
"""
(:type
    state - vector[float,2]
)
(:predicate
    contain
)
"""



contact_domain = load_domain_string(domain_string, domain_parser)
contact_executor = CentralExecutor(contact_domain)

domains = {
    "Generic": generic_executor,
    "Distance": distance_executor,
    "Contact": contact_executor,
}

morphisms = [
    ("Generic", "Distance"),
    ("Distance", "Contact"),
    ]

model.concept_diagram.root_name = "Generic"
for domain_name in domains:
    model.concept_diagram.add_domain(domain_name,domains[domain_name])


for source, target in morphisms:
        model.concept_diagram.add_morphism(source, target, MetaphorMorphism(domains[source], domains[target]))

morph = model.concept_diagram.get_morphism("Distance", "Contact", 0)

morph.predicate_matrix.set_connection_weight("very_near", "contact", 1.0)

#print(model.concept_diagram.morphisms.keys())

save_ensemble_model(model, "checkpoints/contact1.ckpt")

model = load_ensemble_model(config, "checkpoints/contact1.ckpt")
