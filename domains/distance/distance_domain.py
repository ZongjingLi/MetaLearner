#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : distance.py
# Author : Zongjing Li
# Modified: [Assistant]
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of distance predicates
# Distributed under terms of the MIT license.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    load_domain_string,
    domain_parser,
    DifferentiableOps
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

__all__ = [
    'DistanceDomain',
    'build_distance_executor'
]

# Domain definition
DISTANCE_DOMAIN = """
(domain Distance)
(:type
    state - vector[float,2]
    position - vector[float,2]
    distance - float
)
(:predicate
    get_position ?x-state -> position
    very_near ?x-state ?y-state -> boolean
    near ?x-state ?y-state -> boolean
    moderately_far ?x-state ?y-state -> boolean
    far ?x-state ?y-state -> boolean
    very_far ?x-state ?y-state -> boolean
    euclidean_distance ?x-state ?y-state -> distance
    manhattan_distance ?x-state ?y-state -> distance
    closer_than ?x-state ?y-state ?z-state -> boolean
    further_than ?x-state ?y-state ?z-state -> boolean
)
"""

class DistanceDomain:
    """Handler for distance predicates and spatial relations.
    
    Implements differentiable predicates for qualitative distance reasoning
    between points in 2D space. Supports both metric distances (Euclidean, Manhattan)
    and qualitative relations (near, far, etc.) with smooth transitions.
    """
    
    def __init__(self, temperature: float = 0.1, epsilon: float = 1e-6):
        """Initialize distance domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls transition sharpness
            epsilon: Small value for numerical stability in distance calculations
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define distance thresholds for qualitative predicates
        self.distance_thresholds = {
            'very_near': 0.5,     # Under 0.5 units
            'near': 1.0,          # Under 1 unit
            'moderately_far': 2.0, # Around 2 units
            'far': 4.0,           # Around 4 units
            'very_far': 8.0       # Over 8 units
        }
        
        # Define sigmas for Gaussian kernels
        self.gaussian_sigmas = {
            'very_near': 0.2,
            'near': 0.4,
            'moderately_far': 0.8,
            'far': 1.6,
            'very_far': 3.2
        }
    
    def _gaussian_kernel(self, x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Compute Gaussian kernel for smooth transitions.
        
        Args:
            x: Input tensor to transform
            sigma: Standard deviation parameter
            
        Returns:
            Tensor of same shape as input with Gaussian kernel values
        """
        return torch.exp(-0.5 * (x / sigma) ** 2)
    
    def euclidean_distance(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise Euclidean distances between points.
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of Euclidean distances
        """
        x_exp = x_state.unsqueeze(1)  # [B1, 1, 2]
        y_exp = y_state.unsqueeze(0)  # [1, B2, 2]
        diff = x_exp - y_exp
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + self.epsilon)
    
    def manhattan_distance(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise Manhattan distances between points.
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of Manhattan distances
        """
        x_exp = x_state.unsqueeze(1)  # [B1, 1, 2]
        y_exp = y_state.unsqueeze(0)  # [1, B2, 2]
        diff = x_exp - y_exp
        return torch.sum(torch.abs(diff), dim=-1)

    def _qualitative_distance(self, x_state: torch.Tensor, y_state: torch.Tensor,
                            threshold: float, sigma: float) -> torch.Tensor:
        """Helper for qualitative distance predicates.
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            threshold: Distance threshold for predicate
            sigma: Sigma for Gaussian kernel smoothing
            
        Returns:
            [B1, B2] tensor of predicate values
        """
        distances = self.euclidean_distance(x_state, y_state)
        return self._gaussian_kernel(distances - threshold, sigma)

    def very_near(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate very near predicate (distance < 0.5).
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of very_near scores
        """
        return self._qualitative_distance(
            x_state, y_state,
            self.distance_thresholds['very_near'],
            self.gaussian_sigmas['very_near']
        )
    
    def near(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate near predicate (distance < 1.0).
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of near scores
        """
        return self._qualitative_distance(
            x_state, y_state,
            self.distance_thresholds['near'],
            self.gaussian_sigmas['near']
        )
    
    def moderately_far(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate moderately far predicate (distance ≈ 2.0).
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of moderately_far scores
        """
        return self._qualitative_distance(
            x_state, y_state,
            self.distance_thresholds['moderately_far'],
            self.gaussian_sigmas['moderately_far']
        )
    
    def far(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate far predicate (distance ≈ 4.0).
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of far scores
        """
        return self._qualitative_distance(
            x_state, y_state,
            self.distance_thresholds['far'],
            self.gaussian_sigmas['far']
        )
    
    def very_far(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate very far predicate (distance > 8.0).
        
        Args:
            x_state: [B1, 2] tensor of first point positions
            y_state: [B2, 2] tensor of second point positions
            
        Returns:
            [B1, B2] tensor of very_far scores
        """
        return self._qualitative_distance(
            x_state, y_state,
            self.distance_thresholds['very_far'],
            self.gaussian_sigmas['very_far']
        )

    def closer_than(self, x_state: torch.Tensor, y_state: torch.Tensor,
                   ref_state: torch.Tensor) -> torch.Tensor:
        """Calculate if points in x are closer to ref than points in y.
        
        Args:
            x_state: [B1, 2] tensor of test point positions
            y_state: [B2, 2] tensor of comparison point positions
            ref_state: [B3, 2] tensor of reference point positions
            
        Returns:
            [B1, B2, B3] tensor of closer_than scores
        """
        x_ref_dist = self.euclidean_distance(x_state, ref_state)  # [B1, B3]
        y_ref_dist = self.euclidean_distance(y_state, ref_state)  # [B2, B3]
        
        x_ref_dist = x_ref_dist.unsqueeze(1)  # [B1, 1, B3]
        y_ref_dist = y_ref_dist.unsqueeze(0)  # [1, B2, B3]
        
        return torch.sigmoid((y_ref_dist - x_ref_dist) / self.temperature)
    
    def further_than(self, x_state: torch.Tensor, y_state: torch.Tensor,
                    ref_state: torch.Tensor) -> torch.Tensor:
        """Calculate if points in x are further from ref than points in y.
        
        Args:
            x_state: [B1, 2] tensor of test point positions
            y_state: [B2, 2] tensor of comparison point positions
            ref_state: [B3, 2] tensor of reference point positions
            
        Returns:
            [B1, B2, B3] tensor of further_than scores
        """
        return self.closer_than(y_state, x_state, ref_state)

    def visualize(self, states_dict: Dict[int, Any],
                 relation_matrix: Optional[torch.Tensor] = None,
                 program: Optional[str] = None) -> plt.Figure:
        """Visualize points and their distance relationships.
        
        Args:
            states_dict: Dictionary mapping indices to state tensors
            relation_matrix: Optional tensor of relation scores
            program: Optional program string to display
            
        Returns:
            Matplotlib figure with visualization
        """
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        state_sizes = {}
        
        # Setup plot bounds
        self._setup_plot_bounds(ax1, states_dict)
        
        # Plot points and relations
        self._plot_points(ax1, states_dict, colors, markers, state_sizes)
        if relation_matrix is not None:
            self._plot_relations(ax1, ax2, states_dict, state_sizes, relation_matrix)
        
        # Finalize plot
        self._finalize_plot(fig, ax1, program)
        
        return fig

    def _setup_plot_bounds(self, ax: plt.Axes, states_dict: Dict):
        """Set plot bounds based on point positions.
        
        Args:
            ax: Matplotlib axes for plotting
            states_dict: Dictionary of state tensors
        """
        all_points = []
        for value in states_dict.values():
            all_points.extend(value["state"][:, :2].numpy())
        all_points = np.array(all_points)
        
        if len(all_points) > 0:
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            max_range = max(max_x - min_x, max_y - min_y) * 1.2
            ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
            ax.set_ylim(center_y - max_range/2, center_y + max_range/2)
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

    def _plot_points(self, ax: plt.Axes, states_dict: Dict,
                    colors: List[str], markers: List[str],
                    state_sizes: Dict):
        """Plot points with labels.
        
        Args:
            ax: Matplotlib axes for plotting
            states_dict: Dictionary of state tensors
            colors: List of colors for different states
            markers: List of markers for different states
            state_sizes: Dictionary to store number of points per state
        """
        for i, (key, value) in enumerate(states_dict.items()):
            state = value["state"]
            state_sizes[key] = len(state)
            
            ax.scatter(
                state[:, 0].numpy(),
                state[:, 1].numpy(),
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=f'State {key}',
                s=100,
                zorder=3
            )
            
            for j in range(len(state)):
                ax.annotate(
                    f'{key}_{j}',
                    (state[j, 0].item(), state[j, 1].item()),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )

    def _plot_relations(self, ax1: plt.Axes, ax2: plt.Axes,
                       states_dict: Dict, state_sizes: Dict,
                       relation_matrix: torch.Tensor):
        """Plot relation lines and matrix visualization.
        
        Args:
            ax1: First matplotlib axes for spatial plot
            ax2: Second matplotlib axes for relation matrix
            states_dict: Dictionary of state tensors
            state_sizes: Dictionary of numbers of points per state
            relation_matrix: Tensor of relation scores
        """
        if relation_matrix.dim() == 2:
            state0 = states_dict[0]["state"]
            state1 = states_dict[1]["state"]
            
            # Draw relation lines
            for i in range(state_sizes[0]):
                for j in range(state_sizes[1]):
                    strength = relation_matrix[i, j].item()
                    if strength > 0.5:
                        ax1.plot(
                            [state0[i, 0].item(), state1[j, 0].item()],
                            [state0[i, 1].item(), state1[j, 1].item()],
                            'k--', alpha=min(0.7, strength),
                            linewidth=1, zorder=2
                        )
            
            # Plot relation matrix
            im = ax2.imshow(
                relation_matrix.numpy(),
                cmap='viridis',
                aspect='equal',
                interpolation='nearest'
            )
            plt.colorbar(im, ax=ax2)
            
            # Add matrix labels
            ax2.set_xticks(np.arange(state_sizes[1]))
            ax2.set_yticks(np.arange(state_sizes[0]))
            ax2.set_xticklabels([f'1_{i}' for i in range(state_sizes[1])])
            ax2.set_yticklabels([f'0_{i}' for i in range(state_sizes[0])])
            ax2.set_title("Relation Matrix")

    def _finalize_plot(self, fig: plt.Figure, ax: plt.Axes, program: Optional[str]):
        """Add final touches to the visualization.
        
        Args:
            fig: Matplotlib figure
            ax: Main plotting axes
            program: Optional program string to display
        """
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax.set_axisbelow(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("Spatial Configuration")
        
        if program is not None:
            fig.suptitle(f"Program: {program}")
            
        plt.tight_layout()

    def setup_predicates(self, executor: CentralExecutor):
        """Setup all distance predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        state_type = tvector(treal, 2)  # 2D position vector
        position_type = tvector(treal, 2)  # 2D position vector
        distance_type = treal  # scalar distance
        
        executor.update_registry({
            "get_position": Primitive(
                "get_position",
                arrow(state_type, position_type),
                lambda x: {**x, "end": x["state"]}
            ),
            
            "very_near": Primitive(
                "very_near",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.very_near(x["state"], y["state"])}
            ),
            
            "near": Primitive(
                "near",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.near(x["state"], y["state"])}
            ),
            
            "moderately_far": Primitive(
                "moderately_far",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.moderately_far(x["state"], y["state"])}
            ),
            
            "far": Primitive(
                "far",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.far(x["state"], y["state"])}
            ),
            
            "very_far": Primitive(
                "very_far",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.very_far(x["state"], y["state"])}
            ),
            
            "euclidean_distance": Primitive(
                "euclidean_distance",
                arrow(state_type, arrow(state_type, distance_type)),
                lambda x: lambda y: {**x, "end": self.euclidean_distance(x["state"], y["state"])}
            ),
            
            "manhattan_distance": Primitive(
                "manhattan_distance",
                arrow(state_type, arrow(state_type, distance_type)),
                lambda x: lambda y: {**x, "end": self.manhattan_distance(x["state"], y["state"])}
            ),
            
            "closer_than": Primitive(
                "closer_than",
                arrow(state_type, arrow(state_type, arrow(state_type, boolean))),
                lambda x: lambda y: lambda z: {**x, "end": self.closer_than(x["state"], y["state"], z["state"])}
            ),
            
            "further_than": Primitive(
                "further_than",
                arrow(state_type, arrow(state_type, arrow(state_type, boolean))),
                lambda x: lambda y: lambda z: {**x, "end": self.further_than(x["state"], y["state"], z["state"])}
            )
        })


def build_distance_executor(temperature: float = 0.1) -> CentralExecutor:
    """Build distance executor with domain.
    
    Args:
        temperature: Temperature for smooth operations, controls transition sharpness
        
    Returns:
        Initialized distance executor instance
    """
    # Load domain and create executor
    domain = load_domain_string(DISTANCE_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup predicates
    distance_domain = DistanceDomain(temperature)
    distance_domain.setup_predicates(executor)
    
    # Add visualization method to executor
    executor.visualize = distance_domain.visualize
    
    return executor

# Create default executor instance
distance_executor = build_distance_executor()