'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-11-10 12:01:37
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-12-28 18:23:31
 # @ Description: This file is distributed under the MIT license.
'''

import torch
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

    domains = {
    "GenericDomain": generic_executor,
    "LineDomain": line_executor,
    "CurveDomain": curve_executor,
    "RCC8Domain": rcc8_executor,
    "DistanceDomain": distance_executor,
    "DirectionDomain": direction_executor,
    "PointcloudDomain": pointcloud_executor
    }

    for domain_name, executor in domains.items(): concept_diagram.add_domain(domain_name, executor)

    morphisms = [
    ("GenericDomain", "LineDomain"),
    ("GenericDomain", "DistanceDomain"),
    ("GenericDomain", "DirectionDomain"),
    ("DistanceDomain", "DirectionDomain"),
    ("CurveDomain", "LineDomain"),
    ("LineDomain", "RCC8Domain"),
    ("LineDomain", "RCC8Domain"),
    ("DistanceDomain", "RCC8Domain"),
    ("GenericDomain", "CurveDomain"),
    ("GenericDomain", "PointcloudDomain")
    ]

    for source, target in morphisms:
        concept_diagram.add_morphism(source, target, MetaphorMorphism(domains[source], domains[target]))


    
    from core.model import EnsembleModel
    from config import config
    model = EnsembleModel(config)
    model.concept_diagram = concept_diagram
    model.to(device)

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

    b, n, d = 4, 8, 256
    sample_dict = {
        "features" : torch.randn([b, n, d], device = device),
        "end" : torch.randn([b, n], device = device),
        "predicates" : ["disconnected", "disconnected", "disconnected", "disconnected"]
    }


    result = concept_diagram.batch_evaluation(sample_dict, "literal")

    print(len(concept_diagram.get_path("GenericDomain", "RCC8Domain")))
    print(concept_diagram.get_path("GenericDomain", "RCC8Domain"))