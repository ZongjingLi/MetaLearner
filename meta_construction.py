'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-11-10 12:01:37
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-12-28 18:23:31
 # @ Description: This file is distributed under the MIT license.
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import heapq

from rinarak.logger import get_logger, KFTLogFormatter
from rinarak.logger import set_logger_output_file

from rinarak.domain import load_domain_string
from rinarak.knowledge.executor import CentralExecutor

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

device = "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == "__main__":
    from core.metaphors.diagram import ConceptDiagram, MetaphorMorphism
    from domains.generic.generic_domain import generic_executor
    from domains.line.line_domain import line_executor
    from domains.rcc8.rcc8_domain import rcc8_executor
    from domains.curve.curve_domain import curve_executor
    from domains.distance.distance_domain import distance_executor
    from domains.direction.direction_domain import direction_executor
    from domains.pointcloud.pointcloud_domain import pointcloud_executor

    concept_diagram = ConceptDiagram()
    curve_executor.to(device)
    concept_diagram.add_domain("GenericDomain", generic_executor)
    concept_diagram.add_domain("LineDomain", line_executor)
    concept_diagram.add_domain("CurveDomain", curve_executor)
    concept_diagram.add_domain("RCC8Domain", rcc8_executor)
    concept_diagram.add_domain("DistanceDomain", distance_executor)
    concept_diagram.add_domain("DirectionDomain", direction_executor)
    concept_diagram.add_domain("PointcloudDomain", pointcloud_executor)


    concept_diagram.add_morphism("GenericDomain", "LineDomain", MetaphorMorphism(generic_executor, line_executor))
    concept_diagram.add_morphism("GenericDomain", "DistanceDomain", MetaphorMorphism(generic_executor, distance_executor))
    concept_diagram.add_morphism("GenericDomain", "DirectionDomain", MetaphorMorphism(generic_executor, direction_executor))

    concept_diagram.add_morphism("DistanceDomain", "DirectionDomain", MetaphorMorphism(distance_executor, direction_executor))

    concept_diagram.add_morphism("CurveDomain", "LineDomain", MetaphorMorphism(curve_executor, line_executor))
    concept_diagram.add_morphism("LineDomain", "RCC8Domain", MetaphorMorphism(line_executor, rcc8_executor))
    concept_diagram.add_morphism("LineDomain", "RCC8Domain", MetaphorMorphism(line_executor, rcc8_executor))
    concept_diagram.add_morphism("DistanceDomain", "RCC8Domain", MetaphorMorphism(distance_executor, rcc8_executor))

    concept_diagram.add_morphism("GenericDomain", "CurveDomain", MetaphorMorphism(generic_executor, curve_executor))
    concept_diagram.add_morphism("GenericDomain", "PointcloudDomain", MetaphorMorphism(generic_executor, pointcloud_executor))
    
    concept_diagram.to(device)

    """generic state space testing"""
    source_state = torch.randn([3, 256]).to(device)
    context = {
        0 : {"state" : source_state},
        1 : {"state" : source_state}
        
    }
    #print(concept_diagram.get_morphism('DistanceDomain', 'RCC8Domain', 0))


    result = concept_diagram.evaluate(source_state, "disconnected", "GenericDomain", "metaphor")
    apply_path = result["apply_path"][0] # [1.0, tensor([0.5170], device='mps:0', grad_fn=<MulBackward0>), tensor([0.2620], device='mps:0', grad_fn=<MulBackward0>)]
    state_path = result["state_path"][0]
    metas_path = result["metas_path"][0] # [('GenericDomain', 'DistanceDomain', 0), ('DistanceDomain', 'DirectionDomain', 0)]

    visualizations = concept_diagram.visualize_path(state_path, metas_path, result["results"][0].cpu().detach())

    
    for i in range(len(result["results"])):
        print(result["metas_path"][i])
        print(result["symbol_path"][i])
        print("Measure Size: ",result["results"][i].shape)
        print("Measure Conf: ",result["probs"][i])
        print("Apply Conf: ", result["apply_path"][i])
        print("Measure State:",result["states"][i].shape)
        print("\n")


    result = concept_diagram.evaluate(source_state, "disconnected", "GenericDomain", "literal")
    print("Done")
    for i in range(len(result["results"])):
        print(result["metas_path"][i])
        print(result["symbol_path"][i])
        print("Measure Size: ",result["results"][i].shape)
        print("Measure Conf: ",result["probs"][i])
        print("Measure State:",result["states"][i].shape)
        print("\n")
