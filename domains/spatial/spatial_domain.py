"""
# @ Author: [Your Name]
# @ Create Time: 2024-11-10
# @ Description: Fully differentiable spatial domain implementation
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from rinarak.dklearn.nn.mlp import FCBlock
from rinarak.utils.tensor import logit
from rinarak.knowledge.executor import CentralExecutor
from domains.utils import domain_parser, load_domain_string

"""define the spatial executor and build up"""
folder_path = os.path.dirname(__file__)
spatial_domain_str = ""
with open(f"{folder_path}/spatial_domain.txt","r") as domain:
        for line in domain: spatial_domain_str += line
executor_domain = load_domain_string(spatial_domain_str, domain_parser)
spatial_executor = CentralExecutor(executor_domain, "cone", 100)

"""load the pretrained curve-pointcloud executor"""
# Constants for fuzzy transitions
gc = 0.0  # Gap constant
tc = 0.15  # Temperature constant

checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .spatial_repr import PointCloudVAE, CurveVAE

pointcloud_vae = PointCloudVAE(num_points=2048, latent_dim=128)
pointcloud_checkpoint_path = os.path.join(checkpoint_dir, 'pointcloud_vae_state.pth')
pointcloud_vae.load_state_dict(torch.load(pointcloud_checkpoint_path, map_location="cpu"))


curve_vae = CurveVAE(num_points=320, latent_dim=64)
curve_checkpoint_path = os.path.join(checkpoint_dir, 'curve_vae_state.pth')
curve_vae.load_state_dict(torch.load(curve_checkpoint_path, map_location="cpu"))


def decode_curve(state):return curve_vae.decoder(state)

def decode_pointcloud(state):
    return pointcloud_vae.decoder(state) + torch.randn(3) * 2

def soft_min(x, dim, temperature=0.1):
    """Differentiable soft minimum"""
    return -(temperature * torch.logsumexp(-x/temperature, dim=dim))

def soft_threshold(x, threshold, scale=0.1):
    """Differentiable threshold function"""
    return torch.sigmoid((threshold - x) / scale)

def soft_and(x, y):
    """Differentiable AND operation"""
    return x * y

def soft_or(x, y):
    """Differentiable OR operation"""
    return x + y - x * y

class ShapeModule(nn.Module):
    def __init__(self, state_dim=128, output_dim=128):
        super().__init__()
        self.linear0 = FCBlock(256, 2, state_dim, output_dim)
    
    def forward(self, x): 
        return self.linear0(x)

def compute_pairwise_distances(pc1, pc2):
    """Compute differentiable pairwise distances between point clouds"""
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # [batch, n1, n2, 3]
    dist = torch.norm(diff + 1e-8, dim=-1)  # Add small epsilon for stability
    return dist

def compute_shape_distance(x, y):
    """Compute differentiable distance between shape point clouds"""
    pc1 = decode_pointcloud(x["state"])  # [batch, n1, 3]
    pc2 = decode_pointcloud(y["state"])  # [batch, n2, 3]
    
    # Compute pairwise distances
    dist = compute_pairwise_distances(pc1, pc2)  # [batch, n1, n2]
    
    # Soft minimum distances in both directions
    min_dist1 = soft_min(dist, dim=2)  # [batch, n1]
    min_dist2 = soft_min(dist, dim=1)  # [batch, n2]
    
    return torch.cat([min_dist1, min_dist2], dim=-1)

def compute_overlap_ratio(x, y, threshold=0.1):
    """Compute differentiable overlap ratio between shapes"""
    dist = compute_pairwise_distances(
        decode_pointcloud(x["state"]),
        decode_pointcloud(y["state"])
    )
    # Soft thresholding for overlap
    overlap_scores = soft_threshold(dist, threshold)
    return torch.mean(overlap_scores, dim=[-1, -2])

# RCC-8 predicates implementation

spatial_executor.redefine_predicate(
    "get_shape",
    lambda x: {**x,
                        "from": 1.0,
                        "set": x["end"],
                        "end": decode_pointcloud(x["state"])}
)

spatial_executor.redefine_predicate(
    "disconnected",
    lambda x: lambda y: {**x,
                        "from": "disconnected",
                        "set": x["end"],
                        "end": 1 - soft_threshold(
                            soft_min(compute_pairwise_distances(
                                decode_pointcloud(x["state"]),
                                decode_pointcloud(y["state"])
                            ), dim=[-1, -2]),
                            threshold=gc,
                            scale=tc
                        )}
)

spatial_executor.redefine_predicate(
    "externally_connected",
    lambda x: lambda y: {**x,
                        "from": "externally_connected",
                        "set": x["end"],
                        "end": soft_threshold(
                            torch.abs(soft_min(compute_pairwise_distances(
                                decode_pointcloud(x["state"]),
                                decode_pointcloud(y["state"])
                            ), dim=[-1, -2])),
                            threshold=tc,
                            scale=gc
                        )}
)

def compute_centroid(points):
    """Compute differentiable centroid of points"""
    return torch.mean(points, dim=1)

def gaussian_kernel(x, sigma=1.0):
    """Differentiable Gaussian kernel"""
    return torch.exp(-x**2 / (2 * sigma**2))

def compute_directional_relation(x, y, direction):
    """Compute differentiable directional relation"""
    pc1 = decode_pointcloud(x["state"])
    pc2 = decode_pointcloud(y["state"])
    
    # Compute relative positions
    c1 = compute_centroid(pc1)
    c2 = compute_centroid(pc2)
    rel_pos = c1 - c2
    
    # Project onto direction
    proj = torch.sum(rel_pos * direction, dim=-1)
    
    # Smooth activation
    return torch.sigmoid(proj / tc)

spatial_executor.redefine_predicate(
    "above",
    lambda x: lambda y: {**x,
                        "from": "above",
                        "set": x["end"],
                        "end": compute_directional_relation(
                            x, y, 
                            direction=torch.tensor([0., 0., 1.]).to(x["state"].device)
                        )}
)

def compute_distance_relation(x, y, near_threshold=0.5):
    """Compute differentiable distance relation"""
    pc1 = decode_pointcloud(x["state"])
    pc2 = decode_pointcloud(y["state"])
    
    # Compute minimum distances using soft min
    dist = soft_min(compute_pairwise_distances(pc1, pc2), dim=[-1, -2])
    
    # Smooth distance membership
    return torch.sigmoid((near_threshold - dist) / tc)

spatial_executor.redefine_predicate(
    "near",
    lambda x: lambda y: {**x,
                        "from": "near",
                        "set": x["end"],
                        "end": compute_distance_relation(x, y, near_threshold=0.5)}
)

spatial_executor.redefine_predicate(
    "far",
    lambda x: lambda y: {**x,
                        "from": "far",
                        "set": x["end"],
                        "end": 1 - compute_distance_relation(x, y, near_threshold=2.0)}
)

def compute_partial_overlap(x, y):
    """Compute differentiable partial overlap"""
    pc1 = decode_pointcloud(x["state"])
    pc2 = decode_pointcloud(y["state"])
    
    # Compute smooth overlap scores
    dist = compute_pairwise_distances(pc1, pc2)
    overlap_scores = gaussian_kernel(dist, sigma=tc)
    
    # Compute partial overlap conditions
    overlap_ratio = torch.mean(overlap_scores, dim=[-1, -2])
    not_complete = 1 - torch.min(
        torch.mean(torch.max(overlap_scores, dim=2)[0], dim=1)[0],
        torch.mean(torch.max(overlap_scores, dim=1)[0], dim=1)[0]
    )
    
    return soft_and(overlap_ratio, not_complete)

spatial_executor.redefine_predicate(
    "partial_overlap",
    lambda x: lambda y: {**x,
                        "from": "partial_overlap",
                        "set": x["end"],
                        "end": compute_partial_overlap(x, y)}
)

from .visualize import *
# Add visualization method to executor
spatial_executor.visualize = visualize