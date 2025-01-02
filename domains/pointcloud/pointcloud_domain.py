#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pointcloud.py
# Author : AI Assistant
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Distributed under terms of the MIT license.

import os
import torch
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay
from typing import Dict, List, Tuple, Optional

from domains.utils import build_domain_dag, DifferentiableOps, load_domain_string, domain_parser
from .pointcloud_repr import PointCloudVAE
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean

__all__ = [
    'PointcloudDomain',
    'build_pointcloud_executor'
]

# Domain definition
POINTCLOUD_DOMAIN = """
(domain Pointcloud)
(:type
    state - vector[float,128]
    position - vector[float, 3]
    pointcloud - vector[float,1024,3]
)
(:predicate
    pointcloud-geometry ?x-state -> pointcloud
    pos ?x-state -> position
    big ?x-state -> boolean
    inside ?x-state ?y-state -> boolean
    outside ?x-state ?y-state -> boolean
    around ?x-state ?y-state -> boolean
    beyond ?x-state ?y-state -> boolean
)
"""

class PointcloudDomain:
    """Pointcloud domain operations with differentiable predicates."""

    def __init__(self, vae_path: str, num_points: int = 2048, 
                 latent_dim: int = 128):
        """Initialize pointcloud domain.
        
        Args:
            vae_path: Path to VAE model state
            num_points: Number of points in point cloud
            latent_dim: Latent dimension size
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize VAE
        self.pointcloud_vae = PointCloudVAE(num_points, latent_dim)
        self.pointcloud_vae.load_state_dict(
            torch.load(vae_path, map_location=self.device)
        )
        self.pointcloud_vae.to(self.device)
        
        # Initialize differentiable ops
        self.ops = DifferentiableOps()

    def decode_pointcloud(self, state: torch.Tensor) -> torch.Tensor:
        """Decode state to pointcloud."""
        return self.pointcloud_vae.decoder(state)

    def decode_position(self, state: torch.Tensor) -> torch.Tensor:
        """Decode state to position."""
        n = state.shape[0]
        return torch.randn([n, 3], device=self.device) * 1.2

    def pointcloud_inside(self, state_A: torch.Tensor, 
                         state_B: torch.Tensor) -> torch.Tensor:
        """Check if pointclouds in A are inside B."""
        A = self.decode_pointcloud(state_A)
        B = self.decode_pointcloud(state_B)

        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        
        a, n_A, dim = A.shape
        b, n_B, _ = B.shape

        assert n_A == n_B, "Point clouds must have same number of points"
        assert dim == 3, "Point clouds must be 3-dimensional"
        
        result_matrix = np.zeros((a, b), dtype=bool)
        
        for j in range(b):
            hull_B = ConvexHull(B[j])
            delaunay_B = Delaunay(B[j][hull_B.vertices])
            
            for i in range(a):
                is_inside = delaunay_B.find_simplex(A[i]) >= 0
                result_matrix[i, j] = np.all(is_inside)

        return torch.tensor(result_matrix, device=self.device).float()

    def setup_predicates(self, executor: 'CentralExecutor'):
        """Setup all pointcloud predicates."""
        # Add implementations to executor's registry
        executor.implement_registry.update({
            "big": Primitive(
                "big",
                arrow(boolean, boolean),
                lambda x: {
                    "end": self.ops.smooth_max(x["state"].norm(dim=-1) - 1.0),
                    **x
                }
            ),
            
            "pointcloud-geometry": Primitive(
                "pointcloud-geometry",
                arrow(boolean, boolean),
                lambda x: {
                    "end": self.decode_pointcloud(x["state"]),
                    **x
                }
            ),
            
            "pos": Primitive(
                "pos",
                arrow(boolean, boolean),
                lambda x: {
                    "end": self.decode_position(x["state"]),
                    **x
                }
            ),
            
            "inside": Primitive(
                "inside",
                arrow(boolean, boolean),
                lambda x: lambda y: {
                    "end": self.pointcloud_inside(x["state"], y["state"]),
                    **x
                }
            ),
            
            "outside": Primitive(
                "outside",
                arrow(boolean, boolean),
                lambda x: lambda y: {
                    "end": 1.0 - self.pointcloud_inside(x["state"], y["state"]),
                    **x
                }
            ),
            
            "around": Primitive(
                "around",
                arrow(boolean, boolean),
                lambda x: lambda y: {
                    "end": self.ops.gaussian_kernel(
                        self.pointcloud_inside(x["state"], y["state"])
                    ),
                    **x
                }
            ),
            
            "beyond": Primitive(
                "beyond",
                arrow(boolean, boolean),
                lambda x: lambda y: {
                    "end": 1.0 - self.ops.gaussian_kernel(
                        self.pointcloud_inside(x["state"], y["state"])
                    ),
                    **x
                }
            )
        })

def build_pointcloud_executor(vae_path: Optional[str] = None, 
                            num_points: int = 2048, 
                            latent_dim: int = 128) -> 'CentralExecutor':
    """Build pointcloud executor with domain.
    
    Args:
        vae_path: Optional path to VAE model state. If None, uses default path
        num_points: Number of points in point cloud
        latent_dim: Latent dimension size
        
    Returns:
        Initialized pointcloud executor
    """
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use default VAE path if none provided
    if vae_path is None:
        vae_path = os.path.join(current_dir, "pointcloud_vae_state.pth")
    
    # Load domain and create executor
    domain = load_domain_string(POINTCLOUD_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain with VAE
    pointcloud_domain = PointcloudDomain(vae_path, num_points, latent_dim)
    pointcloud_domain.setup_predicates(executor)
    
    return executor

# Create default executor instance
pointcloud_executor = build_pointcloud_executor()