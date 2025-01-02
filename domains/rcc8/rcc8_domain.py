#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rcc8.py
# Author : Zongjing Li
# Modified: [Assistant]
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of RCC8 (Region Connection Calculus)
# Distributed under terms of the MIT license.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    DifferentiableOps,
    load_domain_string,
    domain_parser,
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

__all__ = [
    'RCC8Domain',
    'build_rcc8_executor'
]

# Domain definition
RCC8_DOMAIN = """
(domain RCC8)
(:type
    state - vector[float,3]
    region - vector[float,3]
)
(:predicate
    get_region ?x-state -> region
    disconnected ?x-state ?y-state -> boolean
    externally_connected ?x-state ?y-state -> boolean
    partial_overlap ?x-state ?y-state -> boolean
    equal ?x-state ?y-state -> boolean
    tangential_proper_part ?x-state ?y-state -> boolean
    non_tangential_proper_part ?x-state ?y-state -> boolean
    tangential_proper_part_inverse ?x-state ?y-state -> boolean
    non_tangential_proper_part_inverse ?x-state ?y-state -> boolean
)
"""

class RCC8Domain:
    """Handler for RCC8 spatial relations between regions.
    
    Implements differentiable predicates for the Region Connection Calculus (RCC8) 
    qualitative spatial reasoning framework. Each region is represented by its center
    coordinates and radius. The predicates define topological relationships between
    regions using smooth, differentiable operations.
    """

    def __init__(self, temperature: float = 0.1, epsilon: float = 1e-6):
        """Initialize RCC8 domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls sharpness of transitions
            epsilon: Small value for numerical stability in distance calculations
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ops = DifferentiableOps()

    def _compute_distance(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise distances between region centers.
        
        Args:
            x_state: [B1, 3] tensor of region 1 parameters [x, y, radius]
            y_state: [B2, 3] tensor of region 2 parameters [x, y, radius]
            
        Returns:
            [B1, B2] tensor of center-to-center distances
        """
        x_centers = x_state[:, :2].unsqueeze(1)  # [B1, 1, 2]
        y_centers = y_state[:, :2].unsqueeze(0)  # [1, B2, 2]
        
        diff = x_centers - y_centers
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + self.epsilon)

    def disconnected(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate disconnected (DC) relation.
        
        Regions are disconnected if their distance is greater than the sum of their radii.
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of DC relation scores
        """
        d = self._compute_distance(x_state, y_state)
        x_r = x_state[:, 2].unsqueeze(1)  # [B1, 1]
        y_r = y_state[:, 2].unsqueeze(0)  # [1, B2]
        sum_radii = x_r + y_r
        
        return torch.relu(torch.tanh((d - sum_radii) / self.temperature))

    def externally_connected(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate externally connected (EC) relation.
        
        Regions are externally connected if they touch at their boundaries.
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of EC relation scores
        """
        d = self._compute_distance(x_state, y_state)
        x_r = x_state[:, 2].unsqueeze(1)
        y_r = y_state[:, 2].unsqueeze(0)
        sum_radii = x_r + y_r
        
        diff = torch.abs(d - sum_radii)
        return self.ops.gaussian_kernel(diff, self.temperature)

    def partial_overlap(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate partial overlap (PO) relation.
        
        Regions partially overlap if their distance is between |r1-r2| and r1+r2.
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of PO relation scores
        """
        d = self._compute_distance(x_state, y_state)
        x_r = x_state[:, 2].unsqueeze(1)
        y_r = y_state[:, 2].unsqueeze(0)
        sum_radii = x_r + y_r
        diff_radii = torch.abs(x_r - y_r)
        
        lower_bound = torch.sigmoid((d - diff_radii) / self.temperature)
        upper_bound = torch.sigmoid((sum_radii - d) / self.temperature)
        
        return lower_bound * upper_bound

    def equal(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate equal (EQ) relation.
        
        Regions are equal if their centers coincide and radii match.
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of EQ relation scores
        """
        d = self._compute_distance(x_state, y_state)
        x_r = x_state[:, 2].unsqueeze(1)
        y_r = y_state[:, 2].unsqueeze(0)
        
        centers_equal = self.ops.gaussian_kernel(d, self.temperature)
        radii_equal = self.ops.gaussian_kernel(x_r - y_r, self.temperature)
        
        return centers_equal * radii_equal

    def tangential_proper_part(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate tangential proper part (TPP) relation.
        
        Region x is a TPP of y if it's properly inside y and touches the boundary.
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of TPP relation scores
        """
        d = self._compute_distance(x_state, y_state)
        x_r = x_state[:, 2].unsqueeze(1)
        y_r = y_state[:, 2].unsqueeze(0)
        
        containment = torch.sigmoid((y_r - (x_r + d)) / self.temperature)
        boundary_touch = self.ops.gaussian_kernel(d - (y_r - x_r), self.temperature)
        
        return containment * boundary_touch * (1 - self.equal(x_state, y_state))

    def non_tangential_proper_part(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Calculate non-tangential proper part (NTPP) relation.
        
        Region x is an NTPP of y if it's strictly inside y without touching the boundary.
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of NTPP relation scores
        """
        d = self._compute_distance(x_state, y_state)
        x_r = x_state[:, 2].unsqueeze(1)
        y_r = y_state[:, 2].unsqueeze(0)
        
        containment = torch.sigmoid((y_r - (x_r + d)) / self.temperature)
        non_touching = torch.sigmoid((y_r - x_r - d) / self.temperature)
        
        return containment * non_touching * (1 - self.equal(x_state, y_state))

    def tangential_proper_part_inverse(self, x_state: torch.Tensor, 
                                     y_state: torch.Tensor) -> torch.Tensor:
        """Calculate inverse tangential proper part (TPPi) relation.
        
        TPPi(x,y) is equivalent to TPP(y,x).
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of TPPi relation scores
        """
        return self.tangential_proper_part(y_state, x_state)

    def non_tangential_proper_part_inverse(self, x_state: torch.Tensor, 
                                         y_state: torch.Tensor) -> torch.Tensor:
        """Calculate inverse non-tangential proper part (NTPPi) relation.
        
        NTPPi(x,y) is equivalent to NTPP(y,x).
        
        Args:
            x_state: [B1, 3] tensor of first regions
            y_state: [B2, 3] tensor of second regions
            
        Returns:
            [B1, B2] tensor of NTPPi relation scores
        """
        return self.non_tangential_proper_part(y_state, x_state)

    def visualize(self, states_dict: Dict[int, Any],
                 relation_matrix: Optional[torch.Tensor] = None,
                 program: Optional[str] = None) -> plt.Figure:
        """Visualize regions and their relationships.
        
        Args:
            states_dict: Dictionary mapping indices to state tensors
            relation_matrix: Optional tensor of relation scores between regions
            program: Optional program string to display
            
        Returns:
            Matplotlib figure with visualization
        """
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        state_sizes = {}
        
        self._setup_plot_bounds(ax1, states_dict)
        self._plot_regions(ax1, states_dict, colors, state_sizes)
        
        if relation_matrix is not None:
            self._plot_relations(ax1, ax2, states_dict, state_sizes, relation_matrix)
        
        self._finalize_plot(fig, ax1, program)
        
        return fig

    def _setup_plot_bounds(self, ax: plt.Axes, states_dict: Dict):
        """Setup plot bounds based on region positions and sizes.
        
        Args:
            ax: Matplotlib axes for plotting
            states_dict: Dictionary of state tensors
        """
        all_centers = []
        max_radius = 0
        for value in states_dict.values():
            state = value["state"]
            all_centers.extend(state[:, :2].numpy())
            max_radius = max(max_radius, torch.max(state[:, 2]).item())
        all_centers = np.array(all_centers)
        
        if len(all_centers) > 0:
            min_x, min_y = np.min(all_centers, axis=0)
            max_x, max_y = np.max(all_centers, axis=0)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            range_x = (max_x + max_radius) - (min_x - max_radius)
            range_y = (max_y + max_radius) - (min_y - max_radius)
            max_range = max(range_x, range_y) * 1.2
            
            ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
            ax.set_ylim(center_y - max_range/2, center_y + max_range/2)
        else:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)

    def _plot_regions(self, ax: plt.Axes, states_dict: Dict,
                     colors: List[str], state_sizes: Dict):
        """Plot circular regions with centers and labels.
        
        Args:
            ax: Matplotlib axes for plotting
            states_dict: Dictionary of state tensors
            colors: List of colors for different states
            state_sizes: Dictionary to store number of regions per state
        """
        for i, (key, value) in enumerate(states_dict.items()):
            state = value["state"]
            state_sizes[key] = len(state)
            
            for j in range(len(state)):
                circle = plt.Circle(
                    (state[j, 0].item(), state[j, 1].item()),
                    state[j, 2].item(),
                    fill=False,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.7,
                    label=f'State {key}' if j == 0 else "",
                    zorder=2
                )
                ax.add_artist(circle)
                
                ax.scatter(
                    state[j, 0].item(),
                    state[j, 1].item(),
                    color=colors[i % len(colors)],
                    s=50,
                    zorder=3
                )
                ax.annotate(
                    f'{key}_{j}',
                    (state[j, 0].item(), state[j, 1].item()),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    zorder=4
                )

    def _plot_relations(self, ax1: plt.Axes, ax2: plt.Axes,
                       states_dict: Dict, state_sizes: Dict,
                       relation_matrix: torch.Tensor):
        """Plot relation lines and relation matrix visualization.
        
        Args:
            ax1: First matplotlib axes for spatial plot
            ax2: Second matplotlib axes for relation matrix
            states_dict: Dictionary of state tensors
            state_sizes: Dictionary of numbers of regions per state
            relation_matrix: Tensor of relation scores
        """
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
                        linewidth=1, zorder=1
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
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("Spatial Configuration")
        
        if program is not None:
            fig.suptitle(f"Program: {program}")
        
        plt.tight_layout()

    def setup_predicates(self, executor: CentralExecutor):
        """Setup all RCC8 predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        state_type = tvector(treal, 3)  # [x, y, radius]
        region_type = tvector(treal, 3)  # same as state
        
        executor.update_registry({
            "get_region": Primitive(
                "get_region",
                arrow(state_type, region_type),
                lambda x: {**x, "end": x["state"]}
            ),
            
            "disconnected": Primitive(
                "disconnected",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.disconnected(x["state"], y["state"])}
            ),
            
            "externally_connected": Primitive(
                "externally_connected",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.externally_connected(x["state"], y["state"])}
            ),
            
            "partial_overlap": Primitive(
                "partial_overlap",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.partial_overlap(x["state"], y["state"])}
            ),
            
            "equal": Primitive(
                "equal",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.equal(x["state"], y["state"])}
            ),
            
            "tangential_proper_part": Primitive(
                "tangential_proper_part",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.tangential_proper_part(x["state"], y["state"])}
            ),
            
            "non_tangential_proper_part": Primitive(
                "non_tangential_proper_part",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.non_tangential_proper_part(x["state"], y["state"])}
            ),
            
            "tangential_proper_part_inverse": Primitive(
                "tangential_proper_part_inverse",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.tangential_proper_part_inverse(x["state"], y["state"])}
            ),
            
            "non_tangential_proper_part_inverse": Primitive(
                "non_tangential_proper_part_inverse",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.non_tangential_proper_part_inverse(x["state"], y["state"])}
            )
        })


def build_rcc8_executor(temperature: float = 0.1, 
                       epsilon: float = 1e-6) -> CentralExecutor:
    """Build RCC8 executor with domain.
    
    Args:
        temperature: Temperature for smooth operations
        epsilon: Small value for numerical stability
        
    Returns:
        Initialized RCC8 executor
    """
    # Load domain and create executor
    domain = load_domain_string(RCC8_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup predicates
    rcc8_domain = RCC8Domain(temperature, epsilon)
    rcc8_domain.setup_predicates(executor)
    
    # Add visualization method to executor
    executor.visualize = rcc8_domain.visualize
    
    return executor

# Create default executor instance
rcc8_executor = build_rcc8_executor()