import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.spatial import ConvexHull, Delaunay
from typing import Dict, List, Tuple, Optional, Any

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
)
"""

class PositionDecoder(nn.Module):
    """Small differentiable network to decode position from state"""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        position = self.fc3(x) * 100
        return position

class PointcloudDomain:
    """Pointcloud domain operations with differentiable predicates."""

    def __init__(self, vae_path: str, num_points: int = 2048, latent_dim: int = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        
        # Initialize VAE
        self.pointcloud_vae = PointCloudVAE(num_points, latent_dim)
        self.pointcloud_vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.pointcloud_vae.to(self.device)
        
        # Position decoder
        self.position_decoder = PositionDecoder(latent_dim).to(self.device)
        
        # Initialize differentiable ops
        self.ops = DifferentiableOps()

    def decode_pointcloud(self, state: torch.Tensor) -> torch.Tensor:
        return self.pointcloud_vae.decoder(state)

    def decode_position(self, state: torch.Tensor) -> torch.Tensor:
        return self.position_decoder(state)

    def pointcloud_inside(self, state_A: torch.Tensor, state_B: torch.Tensor) -> torch.Tensor:
        A = self.decode_pointcloud(state_A)
        B = self.decode_pointcloud(state_B)

        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        
        result_matrix = np.zeros((A.shape[0], B.shape[0]), dtype=bool)
        
        for j in range(B.shape[0]):
            hull_B = ConvexHull(B[j])
            delaunay_B = Delaunay(B[j][hull_B.vertices])
            
            for i in range(A.shape[0]):
                is_inside = delaunay_B.find_simplex(A[i]) >= 0
                result_matrix[i, j] = np.all(is_inside)

        return torch.tensor(result_matrix, device=self.device).float()

    def setup_predicates(self, executor: 'CentralExecutor'):
        executor.update_registry({
            "big": Primitive(
                "big",
                arrow(boolean, boolean),
                lambda x: {**x, "end": self.ops.smooth_max(x["state"].norm(dim=-1) - 1.0)}
            ),
            
            "pointcloud-geometry": Primitive(
                "pointcloud-geometry",
                arrow(boolean, boolean),
                lambda x: {**x, "end": self.decode_pointcloud(x["state"])}
            ),
            
            "pos": Primitive(
                "pos",
                arrow(boolean, boolean),
                lambda x: {**x, "end": self.decode_position(x["state"])}
            )
        })

    def visualize(self, states_dict: Dict[int, Any], relation_dict) -> plt.Figure:
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, first subplot (3D)
        ax2 = fig.add_subplot(122)  # 1 row, 2 columns, second subplot (2D)


        colors = [
    "#003f5c",  # Deep Navy Blue
    "#2a6f97",  # Muted Ocean Blue
    "#468faf",  # Soft Teal Blue
    "#70a9a1",  # Subtle Cyan Blue
    "#d62828",  # Rich Crimson Red
    "#ba181b",  # Deep Brick Red
    "#a4161a",  # Classic Blood Red
    "#800f2f"   # Dark Maroon Red
        ]



        for _, (key, value) in enumerate(states_dict.items()):
            pointclouds = self.decode_pointcloud(value["state"].to(self.device)).detach().cpu().numpy()
            positions = self.decode_position(value["state"].to(self.device)).detach().cpu().numpy()

            for i,(pc, pos) in enumerate(zip(pointclouds, positions)):
                ax1.scatter(pc[:, 0] + pos[0], pc[:, 1] + pos[1], pc[:, 2] +  pos[2], s=1, color=colors[i % len(colors)], alpha=0.7)
                ax1.scatter(pos[0], pos[1], pos[2], s=50, color='red', marker='x')
            break

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title("Point Cloud Visualization")
        return fig


def build_pointcloud_executor(vae_path: Optional[str] = None, num_points: int = 2048, latent_dim: int = 128) -> 'CentralExecutor':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if vae_path is None:
        vae_path = os.path.join(current_dir, "pointcloud_vae_state.pth")
    
    domain = load_domain_string(POINTCLOUD_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    pointcloud_domain = PointcloudDomain(vae_path, num_points, latent_dim)
    pointcloud_domain.setup_predicates(executor)
    executor.visualize = pointcloud_domain.visualize
    
    return executor

# Create default executor instance
pointcloud_executor = build_pointcloud_executor()
