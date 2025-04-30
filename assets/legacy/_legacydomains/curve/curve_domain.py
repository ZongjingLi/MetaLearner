#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : curve.py
# Author : Zongjing Li
# Modified: [Assistant]
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of curve predicates
# Distributed under terms of the MIT license.

import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    load_domain_string,
    domain_parser,
    DifferentiableOps,
    smooth_and,
    smooth_or,
    gaussian_kernel
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow 
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

__all__ = [
    'CurveDomain',
    'build_curve_executor'
]

# Domain definition 
CURVE_DOMAIN = """
(domain Curve)
(:type
    state - vector[float,64]
    curve - vector[float,320,2] 
    point - vector[float,2]
    scalar - float
    angle - float
)
(:predicate
    get_curve ?x-state -> curve
    get_start ?x-state -> point
    get_end ?x-state -> point 
    get_length ?x-state -> scalar
    get_centroid ?x-state -> point
    get_curvature ?x-state -> vector[float,320]
    get_direction ?x-state -> vector[float,320]
    get_complexity ?x-state -> scalar
    get_speed ?x-state -> vector[float,320]
    is_closed ?x-state -> boolean
    is_straight ?x-state -> boolean
    is_circular ?x-state -> boolean
    is_uniform ?x-state -> boolean
    similar_shape ?x-state ?y-state -> boolean
    same_length ?x-state ?y-state -> boolean
    parallel_to ?x-state ?y-state -> boolean
)
"""

class CurveDomain:
    """Handler for curve predicates and geometric relations.
    
    Implements differentiable geometric predicates for curve analysis and comparison
    using a VAE-based curve representation. Supports operations like shape similarity,
    spatial relations, and geometric properties.
    """
    
    def __init__(self, num_points: int = 320, latent_dim: int = 64,
                 temperature: float = 0.1, epsilon: float = 1e-6):
        """Initialize curve domain with parameters.
        
        Args:
            num_points: Number of points sampled along each curve
            latent_dim: Dimension of VAE latent space
            temperature: Temperature for smooth operations
            epsilon: Small value for numerical stability
        """
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.temperature = temperature 
        self.epsilon = epsilon

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if not torch.backends.mps.is_available():
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load VAE model
        folder_path = os.path.dirname(os.path.abspath(__file__))
        from .curve_repr import PointCloudVAE
        self.curve_vae = PointCloudVAE(num_points=num_points, latent_dim=latent_dim)
        self.curve_vae.load_state_dict(
            torch.load(f"{folder_path}/curve_vae_state.pth", map_location=self.device, weights_only = True)
        )
        self.curve_vae.to(self.device)


    def _pairwise_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise distances between point sets.
        
        Args:
            x: [B1, N1, 2] tensor of first point set
            y: [B2, N2, 2] tensor of second point set
            
        Returns:
            [B1, B2, N1, N2] tensor of pairwise distances
        """
        x_exp = x.unsqueeze(1).unsqueeze(-2)  # [B1, 1, N1, 1, 2]
        y_exp = y.unsqueeze(0).unsqueeze(-3)  # [1, B2, 1, N2, 2]
        return torch.sqrt(torch.sum((x_exp - y_exp)**2, dim=-1) + self.epsilon)
    
    def _segment_lengths(self, points: torch.Tensor) -> torch.Tensor:
        """Calculate lengths of curve segments between consecutive points.
        
        Args:
            points: [B, N, 2] tensor of curve points
            
        Returns:
            [B, N-1] tensor of segment lengths
        """
        diff = points[:, 1:] - points[:, :-1]
        return torch.sqrt(torch.sum(diff**2, dim=-1) + self.epsilon)

    def decode_curve(self, state: torch.Tensor) -> torch.Tensor:
        """Decode state tensor to curve points using VAE.
        
        Args:
            state: [B, latent_dim] tensor of latent vectors
            
        Returns:
            [B, num_points, 2] tensor of curve points
        """
        return self.curve_vae.decoder(state)

    def get_curve(self, x_state: torch.Tensor) -> torch.Tensor:
        """Get points along curve from state.
        
        Args:
            x_state: [B, latent_dim] tensor of states
        
        Returns:
            [B, num_points, 2] tensor of curve points
        """
        return self.decode_curve(x_state)
    
    def get_start(self, x_state: torch.Tensor) -> torch.Tensor:
        """Get starting point of each curve.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B, 2] tensor of start points
        """
        return self.decode_curve(x_state)[:, 0]
    
    def get_end(self, x_state: torch.Tensor) -> torch.Tensor:
        """Get ending point of each curve.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B, 2] tensor of end points
        """
        return self.decode_curve(x_state)[:, -1]
    
    def get_length(self, x_state: torch.Tensor) -> torch.Tensor:
        """Calculate total length of each curve.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B] tensor of curve lengths
        """
        return self._segment_lengths(self.decode_curve(x_state)).sum(dim=-1)
    
    def get_centroid(self, x_state: torch.Tensor) -> torch.Tensor:
        """Calculate centroid (center of mass) of each curve.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B, 2] tensor of centroids
        """
        return self.decode_curve(x_state).mean(dim=1)

    def get_curvature(self, x_state: torch.Tensor) -> torch.Tensor:
        """Calculate discrete curvature at each point.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B, num_points] tensor of curvature values
        """
        points = self.decode_curve(x_state)
        tangent = points[:, 2:] - points[:, :-2]
        tangent_angles = torch.atan2(tangent[..., 1], tangent[..., 0])
        curvature = torch.diff(tangent_angles, dim=1)
        return F.pad(curvature, (1, 1), mode='replicate')

    def get_direction(self, x_state: torch.Tensor) -> torch.Tensor:
        """Calculate tangent direction at each point.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B, num_points] tensor of angles in radians
        """
        points = self.decode_curve(x_state)
        central = points[:, 2:] - points[:, :-2]  
        angles = torch.atan2(central[..., 1], central[..., 0])
        return F.pad(angles, (1, 1), mode='replicate')

    def get_complexity(self, x_state: torch.Tensor) -> torch.Tensor:
        """Calculate curve complexity based on total absolute curvature.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B] tensor of complexity scores
        """
        return torch.sum(torch.abs(self.get_curvature(x_state)), dim=-1)

    def get_speed(self, x_state: torch.Tensor) -> torch.Tensor:
        """Calculate point spacing along curve.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B, num_points] tensor of speeds
        """
        points = self.decode_curve(x_state)
        diffs = points[:, 1:] - points[:, :-1]
        speeds = torch.sqrt(torch.sum(diffs**2, dim=-1) + self.epsilon)
        return F.pad(speeds, (0, 1), mode='replicate')

    def is_closed(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if curves are closed (start ≈ end).
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B] tensor of closure scores
        """
        start = self.get_start(x_state)
        end = self.get_end(x_state)
        dist = torch.sqrt(torch.sum((end - start)**2, dim=-1) + self.epsilon)
        return torch.sigmoid(-dist / self.temperature + 3.0)

    def is_straight(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if curves approximate straight lines.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B] tensor of straightness scores
        """
        points = self.decode_curve(x_state)
        centered = points - points.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (points.size(1) - 1)
        eigenvals, _ = torch.linalg.eigh(cov)
        linearity_error = eigenvals[..., 0] / (eigenvals.sum(dim=-1) + self.epsilon)
        return (2 * torch.sigmoid(-linearity_error / self.temperature) - 1 + 1) / 2

    def is_circular(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if curves approximate circles using multiple criteria.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B] tensor of circularity scores
        """
        points = self.decode_curve(x_state)
        center = points.mean(dim=1, keepdim=True)
        
        # Radius consistency
        radii = torch.sqrt(torch.sum((points - center)**2, dim=-1) + self.epsilon)
        mean_radius = radii.mean(dim=-1, keepdim=True)
        radius_error = torch.abs(radii - mean_radius).mean(dim=-1)
        radius_prob = (2 * torch.sigmoid(-radius_error / self.temperature + 3) - 1 + 1) / 2

        # Angular distribution
        angles = torch.atan2(
            points[..., 1] - center[..., 1],
            points[..., 0] - center[..., 0]
        )
        sorted_angles, _ = torch.sort(angles, dim=-1)
        angle_diffs = torch.diff(sorted_angles, dim=-1)
        expected_diff = 2 * torch.pi / (points.size(1) - 1)
        angle_error = torch.abs(angle_diffs - expected_diff).mean(dim=-1) 
        angle_prob = (2 * torch.sigmoid(3 - angle_error / self.temperature) - 1 + 1) / 2

        # Curvature consistency
        v1 = points[:, 1:-1] - points[:, :-2]
        v2 = points[:, 2:] - points[:, 1:-1]
        dots = torch.sum(v1 * v2, dim=-1)
        norms = torch.sqrt(torch.sum(v1**2, dim=-1) + self.epsilon) * \
                torch.sqrt(torch.sum(v2**2, dim=-1) + self.epsilon)
        angles = torch.acos(torch.clamp(dots / norms, -1 + self.epsilon, 1 - self.epsilon))
        mean_angle = angles.mean(dim=-1, keepdim=True)
        curvature_error = torch.abs(angles - mean_angle).mean(dim=-1)
        curvature_prob = (2 * torch.sigmoid(-curvature_error / self.temperature + 3) - 1 + 1) / 2

        return (radius_prob * angle_prob * curvature_prob) ** (1/3)
    
    def is_uniform(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if points are uniformly spaced along curve.
        
        Args:
            x_state: [B, latent_dim] tensor of states
            
        Returns:
            [B] tensor of uniformity scores
        """
        speeds = self.get_speed(x_state)[:, :-1]
        mean_speed = speeds.mean(dim=-1, keepdim=True)
        speed_var = torch.mean((speeds - mean_speed)**2, dim=-1)
        return torch.sigmoid(-speed_var / self.temperature)

    def similar_shape(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if curves have similar shape after Procrustes alignment.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of similarity scores
        """
        x_points = self.decode_curve(x_state)
        y_points = self.decode_curve(y_state)
        
        x_centered = x_points - self.get_centroid(x_state).unsqueeze(1)
        y_centered = y_points - self.get_centroid(y_state).unsqueeze(1)
        
        x_scale = torch.sqrt(torch.sum(x_centered**2, dim=(-2,-1)) + self.epsilon)
        y_scale = torch.sqrt(torch.sum(y_centered**2, dim=(-2,-1)) + self.epsilon)
        
        x_normalized = x_centered / (x_scale.unsqueeze(-1).unsqueeze(-1) + self.epsilon)
        y_normalized = y_centered / (y_scale.unsqueeze(-1).unsqueeze(-1) + self.epsilon)
        
        dists = self._pairwise_distances(x_normalized, y_normalized)
        min_dists = torch.min(dists, dim=-1)[0].mean(dim=-1)
        
        return torch.sigmoid(-min_dists / self.temperature)

    def same_length(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if curves have same arc length.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of length similarity scores
        """
        x_length = self.get_length(x_state)
        y_length = self.get_length(y_state)
        diff = torch.abs(x_length.unsqueeze(1) - y_length.unsqueeze(0))
        return torch.sigmoid(-diff / self.temperature)

    def parallel_to(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if curves are locally parallel.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of parallelism scores
        """
        x_points = self.decode_curve(x_state)
        y_points = self.decode_curve(y_state)
        
        x_dir = x_points[:, 1:] - x_points[:, :-1]
        y_dir = y_points[:, 1:] - y_points[:, :-1]
        
        x_norm = torch.sqrt(torch.sum(x_dir**2, dim=-1, keepdim=True) + self.epsilon)
        y_norm = torch.sqrt(torch.sum(y_dir**2, dim=-1, keepdim=True) + self.epsilon)
        
        x_dir = x_dir / (x_norm + self.epsilon)
        y_dir = y_dir / (y_norm + self.epsilon)
        
        x_exp = x_dir.unsqueeze(1)
        y_exp = y_dir.unsqueeze(0)
        dot_prod = torch.sum(x_exp * y_exp, dim=-1)
        alignment = torch.abs(dot_prod).mean(dim=-1)
        
        return torch.sigmoid((alignment - 0.5) / self.temperature)

    def intersects(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if curves intersect.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of intersection scores
        """
        x_points = self.decode_curve(x_state)
        y_points = self.decode_curve(y_state)
        dists = self._pairwise_distances(x_points, y_points)
        min_dist = torch.min(torch.min(dists, dim=-1)[0], dim=-1)[0]
        return torch.exp(-min_dist / self.temperature)

    def contains(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if one curve contains another.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of containment scores
        """
        x_points = self.decode_curve(x_state)
        y_points = self.decode_curve(y_state)
        dists = self._pairwise_distances(x_points, y_points)
        min_dists = torch.min(dists, dim=-2)[0]
        max_dist = torch.max(min_dists, dim=-1)[0]
        return torch.sigmoid(-max_dist / self.temperature)

    def above(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if one curve is above another.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of above relation scores
        """
        x_center = self.get_centroid(x_state)
        y_center = self.get_centroid(y_state)
        x_exp = x_center.unsqueeze(1)
        y_exp = y_center.unsqueeze(0)
        diff_y = y_exp[..., 1] - x_exp[..., 1]
        return torch.sigmoid(-diff_y / self.temperature)

    def left_of(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if one curve is left of another.
        
        Args:
            x_state: [B1, latent_dim] tensor of first states
            y_state: [B2, latent_dim] tensor of second states
            
        Returns:
            [B1, B2] tensor of left relation scores
        """
        x_center = self.get_centroid(x_state)
        y_center = self.get_centroid(y_state)
        x_exp = x_center.unsqueeze(1)
        y_exp = y_center.unsqueeze(0)
        diff_x = x_exp[..., 0] - y_exp[..., 0]
        return torch.sigmoid(-diff_x / self.temperature)

    def visualize(self, states_dict: Dict[int, Any],
                 relation_matrix: Optional[torch.Tensor] = None,
                 program: Optional[str] = None) -> plt.Figure:
        """Visualize curves and their relationships.
        
        Args:
            states_dict: Dictionary mapping indices to state tensors
            relation_matrix: Optional relation scores between curves
            program: Optional program string to display
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        state_sizes = {}
        
        self._plot_curves(ax1, states_dict, colors, markers, state_sizes)
        
        if relation_matrix is not None:
            self._plot_relations(ax1, ax2, states_dict, state_sizes, relation_matrix)
        
        ax1.grid(True)
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.set_title("Curve Configuration")
        
        if program is not None:
            fig.suptitle(f"Program: {program}")
        
        plt.tight_layout()
        return fig

    def _plot_curves(self, ax: plt.Axes, states_dict: Dict,
                    colors: List[str], markers: List[str],
                    state_sizes: Dict):
        """Plot curves with start/end markers.
        
        Args:
            ax: Matplotlib axes for plotting
            states_dict: Dictionary of state tensors
            colors: List of colors for different curves
            markers: List of markers for different curves
            state_sizes: Dictionary to store number of curves per state
        """
        for i, (key, value) in enumerate(states_dict.items()):
            #print(self.device)
            curves = self.decode_curve(value["state"].to(self.device)).detach()
            state_sizes[key] = len(curves)
            
            for b in range(len(curves)):
                curve = curves[b]
                ax.plot(curve[:, 0].cpu().numpy(), curve[:, 1].cpu().numpy(),
                       f'-{markers[i%len(markers)]}',
                       color=colors[i % len(colors)],
                       label=f'Curve {key}_{b}' if b == 0 else "",
                       alpha=0.7,
                       markersize=4)
                
                # Mark start and end points
                ax.plot(curve[0, 0].cpu().numpy(), curve[0, 1].cpu().numpy(),
                       'g>', markersize=8, alpha=0.8)
                ax.plot(curve[-1, 0].cpu().numpy(), curve[-1, 1].cpu().numpy(),
                       'r<', markersize=8, alpha=0.8)

    def _plot_relations(self, ax1: plt.Axes, ax2: plt.Axes,
                       states_dict: Dict, state_sizes: Dict,
                       relation_matrix: torch.Tensor):
        """Plot relation lines and matrices.
        
        Args:
            ax1: First matplotlib axes for curve plot
            ax2: Second matplotlib axes for relation matrix
            states_dict: Dictionary of state tensors
            state_sizes: Dictionary of number of curves per state
            relation_matrix: Tensor of relation scores
        """
        relation_matrix = relation_matrix.squeeze(-1)
        if relation_matrix.dim() == 2:
            curves0 = self.decode_curve(states_dict[0]["state"])
            curves1 = self.decode_curve(states_dict[1]["state"])
            
            # Draw lines for strong relations
            for i in range(state_sizes[0]):
                for j in range(state_sizes[1]):
                    strength = relation_matrix[i, j].item()
                    if strength > 0.5:
                        c0 = curves0[i].mean(dim=0)
                        c1 = curves1[j].mean(dim=0)
                        ax1.plot([c0[0].item(), c1[0].item()],
                               [c0[1].item(), c1[1].item()],
                               'k--', alpha=strength * 0.5)
            
            # Plot relation matrix
            im = ax2.imshow(relation_matrix.detach().numpy(),
                          cmap='viridis',
                          aspect='equal',
                          interpolation='nearest')
            plt.colorbar(im, ax=ax2)
            ax2.set_title("Relation Matrix")
            ax2.set_xlabel("Curve 2 Index")
            ax2.set_ylabel("Curve 1 Index")
        
        elif relation_matrix.dim() == 3:
            print(relation_matrix.shape)
            # For per-point values (e.g., curvature)
            curves0 = self.decode_curve(states_dict[0]["state"])
            for i in range(min(4, state_sizes[0])):
                values = relation_matrix[i].detach()
                curve = curves0[i]
                print(curve.shape)

                scatter = ax1.scatter(curve[:, 0].cpu().detach().numpy(), curve[:, 1].cpu().detach().numpy(),
                                    c=values.numpy(),
                                    cmap='viridis',
                                    s=100,
                                    alpha=0.5)
                plt.colorbar(scatter, ax=ax2)
                ax2.set_title(f"Point Values (Curve {i})")

    def setup_predicates(self, executor: CentralExecutor):
        """Setup all curve predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        state_type = tvector(treal, 64)  # state vector
        curve_type = tvector(tvector(treal, 2), 320)  # curve points
        point_type = tvector(treal, 2)  # 2D point
        scalar_type = treal  # float scalar
        
        executor.update_registry({
            "get_curve": Primitive(
                "get_curve",
                arrow(state_type, curve_type),
                lambda x: {**x, "end": self.get_curve(x["state"])}
            ),
            
            "get_start": Primitive(
                "get_start",
                arrow(state_type, point_type),
                lambda x: {**x, "end": self.get_start(x["state"])}
            ),
            
            "get_end": Primitive(
                "get_end",
                arrow(state_type, point_type),
                lambda x: {**x, "end": self.get_end(x["state"])}
            ),
            
            "get_length": Primitive(
                "get_length",
                arrow(state_type, scalar_type),
                lambda x: {**x, "end": self.get_length(x["state"])}
            ),
            
            "get_centroid": Primitive(
                "get_centroid",
                arrow(state_type, point_type),
                lambda x: {**x, "end": self.get_centroid(x["state"])}
            ),
            
            "get_curvature": Primitive(
                "get_curvature",
                arrow(state_type, tvector(treal, 320)),
                lambda x: {**x, "end": self.get_curvature(x["state"])}
            ),
            
            "get_direction": Primitive(
                "get_direction",
                arrow(state_type, tvector(treal, 320)),
                lambda x: {**x, "end": self.get_direction(x["state"])}
            ),
            
            "get_complexity": Primitive(
                "get_complexity",
                arrow(state_type, scalar_type),
                lambda x: {**x, "end": self.get_complexity(x["state"])}
            ),
            
            "get_speed": Primitive(
                "get_speed",
                arrow(state_type, tvector(treal, 320)),
                lambda x: {**x, "end": self.get_speed(x["state"])}
            ),
            
            "is_closed": Primitive(
                "is_closed",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.is_closed(x["state"])}
            ),
            
            "is_straight": Primitive(
                "is_straight",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.is_straight(x["state"])}
            ),
            
            "is_circular": Primitive(
                "is_circular", 
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.is_circular(x["state"])}
            ),
            
            "is_uniform": Primitive(
                "is_uniform",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.is_uniform(x["state"])}
            ),
            
            "similar_shape": Primitive(
                "similar_shape",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.similar_shape(x["state"], y["state"])}
            ),
            
            "same_length": Primitive(
                "same_length",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.same_length(x["state"], y["state"])}
            ),
            
            "parallel_to": Primitive(
                "parallel_to",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.parallel_to(x["state"], y["state"])}
            ),
            
            "intersects": Primitive(
                "intersects",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.intersects(x["state"], y["state"])}
            ),
            
            "contains": Primitive(
                "contains",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.contains(x["state"], y["state"])}
            ),
            
            "left_of": Primitive(
                "left_of",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.left_of(x["state"], y["state"])}
            ),
            
            "above": Primitive(
                "above",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.above(x["state"], y["state"])}
            )
        })


def build_curve_executor(num_points: int = 320, 
                        latent_dim: int = 64,
                        temperature: float = 0.1) -> CentralExecutor:
    """Build curve executor with domain.
    
    Args:
        num_points: Number of points per curve
        latent_dim: Dimension of latent space
        temperature: Temperature for smooth operations
        
    Returns:
        Initialized curve executor
    """
    # Load domain and create executor
    domain = load_domain_string(CURVE_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup predicates
    curve_domain = CurveDomain(num_points, latent_dim, temperature)
    curve_domain.setup_predicates(executor)
    
    # Add visualization method to executor
    executor.visualize = curve_domain.visualize
    
    return executor

# Create default executor instance
curve_executor = build_curve_executor()