# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 05:56:03
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-06 06:14:34
import torch
import torch.nn as nn
from rinarak.dklearn.nn.pnn import PointNetfeat
model = PointNetfeat()

class PointCloudEncoder(nn.Module):
    """This is just a wrapper method of the PointNet (object wise)"""
    def __init__(self, generic_dim = 256):
        super().__init__()
        self.encoder = PointNetfeat()
        self.linear = nn.Linear(1024, generic_dim)

    def forward(self, x):
        return self.linear(self.encoder(x))

class PointCloudRelationEncoder(nn.Module):
    def __init__(self, generic_dim=1024, relation_dim=512, hidden_dim=256):
        super().__init__()

        # Point feature extractor (Per-object embedding)
        self.pointnet = PointNetfeat()  # Outputs [N, 1024]
        feature_dim = 1024
        # Relation Network: Processes (fi, fj) pairs
        self.relation_network = nn.Sequential(
            nn.Linear(2 * feature_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, relation_dim),
        )

        # Global Reasoning Network (Processes Aggregated Relations)
        self.global_reasoning = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, generic_dim),
        )

    def forward(self, x):
        """
        x: Tensor of shape [N, 2024, 3] (N objects, each with 2024 points in 3D)
        Returns: Object-level features [N, 1024]
        """
        N = x.shape[0]  # Number of objects

        # Step 1: Extract per-object features (PointNet)
        object_features = self.pointnet(x)  # [N, 1024]

        # Step 2: Compute pairwise relations (Batched)
        idx_i, idx_j = torch.triu_indices(N, N, offset=1)  # Get upper triangle indices (avoid duplicates)
        pairs = torch.cat([object_features[idx_i], object_features[idx_j]], dim=-1)  # [Pairs, 2048]

        relations = self.relation_network(pairs)  # [Pairs, relation_dim]

        # Step 3: Aggregate relation information (Sum per object)
        relation_matrix = torch.zeros(N, relations.shape[-1], device=x.device)
        relation_matrix.index_add_(0, idx_i, relations)  # Aggregate for object i
        relation_matrix.index_add_(0, idx_j, relations)  # Aggregate for object j

        # Step 4: Global reasoning (Refine object-level features)
        scene_features = self.global_reasoning(relation_matrix)  # [N, 1024]

        # Step 5: Combine with initial object features (Residual Connection)
        output_features = scene_features + object_features  # [N, 1024]

        return output_features  # [5, 1024]