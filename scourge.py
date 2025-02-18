# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 05:45:31
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-16 17:49:15
from config import config
from core.model import save_ensemble_model, load_ensemble_model
from core.model import EnsembleModel
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_3d_point_cloud_with_relation(point_cloud, relation_matrix):
    """
    Visualizes a (n x 1024 x d) point cloud alongside its relation matrix.

    Args:
        point_cloud (torch.Tensor or np.ndarray): Shape (n, 1024, d), representing n objects in 3D space.
        relation_matrix (torch.Tensor or np.ndarray): Shape (n, n), representing relations between objects.
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    if isinstance(relation_matrix, torch.Tensor):
        relation_matrix = relation_matrix.cpu().numpy()

    n, num_points, dim = point_cloud.shape
    assert dim == 3, "Point cloud must have 3D coordinates (x, y, z)."

    fig = plt.figure(figsize=(12, 6))

    # Left: 3D Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(n):
        obj_points = point_cloud[i]
        ax1.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], s=5, label=f"Obj {i}")  # Plot 3D points
        centroid = np.mean(obj_points, axis=0)
        ax1.text(centroid[0], centroid[1], centroid[2], str(i), fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))  # Mark object ID
    
    ax1.set_title("3D Point Cloud Visualization")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    ax1.legend()

    # Right: Relation Matrix as Heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(relation_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax2)

    ax2.set_xticks(np.arange(n))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels([f"Obj {i}" for i in range(n)], rotation=90)
    ax2.set_yticklabels([f"Obj {i}" for i in range(n)])

    ax2.set_title("Relation Matrix")
    
    plt.tight_layout()
    plt.show()

device = "mps"
model = load_ensemble_model(config, "checkpoints/local_model.ckpt")

scourge_domain_str = """
(domain Scourge)
(:type
    state - vector[float,3]
    position - vector[float,2]
)
(:predicate
    block_position ?x-state -> position
    on ?x-state ?y-state -> boolean
    cleared ?x-state -> boolean
    holding ?x-state -> boolean
    hand-free -> boolean
)
"""

if 0:
        from core.model import EnsembleModel, curriculum_learning
        from core.curriculum import load_curriculum
        from domains.generic.generic_domain import generic_executor
        model = EnsembleModel(config)

        config.load_ckpt = "contact1.ckpt"
        if config.load_ckpt:
            #model = torch.load(f"{config.ckpt_dir}/{config.load_ckpt}")
            model = load_ensemble_model(config, f"{config.ckpt_dir}/{config.load_ckpt}")

        else:
            model.concept_diagram.add_domain("Generic", generic_executor)
            model.concept_diagram.root_name = "Generic"

        """load the core knowledge from defined domains"""
        core_knowledge = eval(config.core_knowledge)
        config.curriculum_file = "data/contact_curriculum_distance.txt"

        curriculum = load_curriculum(config.curriculum_file) # load the curriculum learning setup

        curriculum_learning(config, model, curriculum) # start the curriculum learning for each block

import torch.nn as nn
from domains.utils import domain_parser, load_domain_string
from rinarak.knowledge.executor import CentralExecutor
scourge_domain = load_domain_string(scourge_domain_str, domain_parser)
scourge_executor = CentralExecutor(scourge_domain)

#model.concept_diagram.add_domain("Scourge", scourge_executor)
#model.concept_diagram.add_morphism("Generic", "Scourge", nn.Linear(4,5))

for morph in model.concept_diagram.morphisms:print(morph)

from datasets.scene_dataset import SceneDataset, scene_collate
from torch.utils.data import DataLoader

dataset = SceneDataset("contact_experiment", "test")
loader = DataLoader(dataset, batch_size = 4, collate_fn = scene_collate, shuffle = True)

for sample in loader:
    raw_inputs = sample["input"]
    scene_predicates = sample["predicate"]
    break

# here we can combine a subgraph as a single domain as long as we assume the limit of the graph is always possible to find.
# this an be achieved by assuming we have some generic domain that all the inputs are in the generic domain.

model.to(device)
for b in range(len(raw_inputs)):
    inputs = torch.stack(raw_inputs[b]).to(device)
    result = model.evaluate(inputs, "contact", "pointcloud", eval_mode = "metaphor")
    print(inputs.shape)
    for i,res in enumerate(result["results"]):
        print("Pred:")
        print(np.array((res > 0).float().cpu().detach()) )
        print("Gt:")
        print( np.array( (scene_predicates["contact"][0][b].detach() > 0).float() ) )
        print("Conf:", result["probs"][i])
        print("Path:",result["metas_path"][i], result["symbol_path"][i])

        apply_path = result["apply_path"][i] # [1.0, tensor([0.517tensorboard --logdir=logs0], device='mps:0', grad_fn=<MulBackward0>), tensor([0.2620], device='mps:0', grad_fn=<MulBackward0>)]
        state_path = result["state_path"][i]
        metas_path = result["metas_path"][i] # [('GenericDomain', 'DistanceDomain', 0), ('DistanceDomain', 'DirectionDomain', 0)]
        print("Apply Path:", apply_path)

        result["metas_path"][i], 
        metaphor_path = [result["metas_path"][i][0][0]]
        for it in result["metas_path"][i]: metaphor_path.append(it[1])
        print(metaphor_path)
        print(result["symbol_path"][i])
        model.concept_diagram.visualize(metaphor_path, result["symbol_path"][i])

        #for state in state_path:
        #print(state.shape)

        #from domains.distance.distance_domain import DistanceDomain
        #distance_domain = DistanceDomain(0.2)

        context = {
        0:{"state":result["states"][i].cpu().detach()},
        1:{"state":result["states"][i].cpu().detach()},
        }
    
        #distance_domain.visualize(context,res.cpu().detach())
        #print(res)
        #model.concept_diagram.domains["Distance"].visualize(context, res.cpu().detach())

        visualizations = model.concept_diagram.visualize_path(state_path, metas_path, result["results"][0].cpu().detach())
        visualize_3d_point_cloud_with_relation(inputs, np.array( (scene_predicates["contact"][0][b].detach() > 0).float() ) )

