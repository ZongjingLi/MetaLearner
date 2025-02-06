#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : direction_domain.py
# Author : Zongjing Li
# Modified: Yiqi Sun
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of directional predicates
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
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean

__all__ = [
    'DirectionalDomain',
    'build_direction_executor'
]

# Domain definition
DIRECTION_DOMAIN = """
(domain Direction)
(:type
    state - vector[float,2]
    position - vector[float,2]
    angle - float
)
(:predicate
    get_position ?x-state -> position
    north ?x-state ?y-state -> boolean
    south ?x-state ?y-state -> boolean
    east ?x-state ?y-state -> boolean
    west ?x-state ?y-state -> boolean
    northeast ?x-state ?y-state -> boolean
    northwest ?x-state ?y-state -> boolean
    southeast ?x-state ?y-state -> boolean
    southwest ?x-state ?y-state -> boolean
    angle_between ?x-state ?y-state -> angle
)
"""

class DirectionalDomain:
    """Handler for directional predicates and spatial relations."""

    def __init__(self, temperature: float = 0.2):
        """Initialize directional domain.
        
        Args:
            temperature: Smoothing factor for angle calculations
        """
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_angle(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable batch angle calculation for all pairs.
        
        Args:
            x_state: [B1, 2] tensor of positions
            y_state: [B2, 2] tensor of positions
            
        Returns:
            [B1, B2] tensor of angles in radians [-π, π]
        """
        x_exp = x_state.unsqueeze(1)
        y_exp = y_state.unsqueeze(0)
        diff = y_exp - x_exp
        return torch.atan2(diff[..., 1] + 1e-6, diff[..., 0] + 1e-6)

    def angle_membership(self, angles: torch.Tensor, center: float, 
                        width: float) -> torch.Tensor:
        """Compute smooth membership for an angle range.
        
        Args:
            angles: [B1, B2] tensor of angles in radians
            center: Center angle in radians
            width: Width of the range in radians
            
        Returns:
            Membership values tensor
        """
        diff = angles - center
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return torch.sigmoid((width/2 - torch.abs(diff)) / self.temperature)

    def north(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable north predicate using angles."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, torch.pi/2, torch.pi/2)
    
    def south(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable south predicate using angles."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, -torch.pi/2, torch.pi/2)
    
    def east(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable east predicate using angles."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, 0.0, torch.pi/2)
    
    def west(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable west predicate using angles."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, torch.pi, torch.pi/2)
    
    def northeast(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable northeast predicate."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, torch.pi/4, torch.pi/2)
    
    def northwest(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable northwest predicate."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, 3*torch.pi/4, torch.pi/2)
    
    def southeast(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable southeast predicate."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, -torch.pi/4, torch.pi/2)
    
    def southwest(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Differentiable southwest predicate."""
        angles = self.compute_angle(x_state, y_state)
        return self.angle_membership(angles, -3*torch.pi/4, torch.pi/2)

    def visualize(self, states_dict: Dict, relation_matrix: Optional[torch.Tensor] = None, 
                 program: Optional[str] = None) -> plt.Figure:
        """Visualize positions and their distance relationships."""
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        state_sizes = {}
        
        # Determine plot bounds
        all_points = []
        for value in states_dict.values():
            all_points.extend(value["state"][:, :2].cpu().detach().numpy())
        all_points = np.array(all_points)
        
        # Set plot bounds and style
        if len(all_points) > 0:
            self._set_plot_bounds(ax1, all_points)
        else:
            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-1, 1)
        
        # Plot points and relations
        self._plot_points(ax1, states_dict, colors, markers, state_sizes)
        if relation_matrix is not None:
            self._plot_relations(ax1, ax2, states_dict, state_sizes, relation_matrix.cpu().detach())
        
        # Finalize plot
        ax1.set_aspect('equal')
        ax1.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax1.set_axisbelow(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title("Spatial Configuration")

        
        if program is not None:
            fig.suptitle(f"Program: {program}")
            
        plt.tight_layout()
        return fig, ax1

    def _set_plot_bounds(self, ax: plt.Axes, points: np.ndarray):
        """Set plot bounds to make square box around points."""
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        max_range = max(max_x - min_x, max_y - min_y) * 1.2
        ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
        ax.set_ylim(center_y - max_range/2, center_y + max_range/2)

    def _plot_points(self, ax: plt.Axes, states_dict: Dict, 
                    colors: List[str], markers: List[str], state_sizes: Dict):
        """Plot points with labels."""
        for i, (key, value) in enumerate(states_dict.items()):
            state = value["state"]
            state_sizes[key] = len(state)
            
            ax.scatter(
                state[:, 0].cpu().detach().numpy(),
                state[:, 1].cpu().detach().numpy(),
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
                    fontsize=8
                )

    def _plot_relations(self, ax1: plt.Axes, ax2: plt.Axes, 
                       states_dict: Dict, state_sizes: Dict, 
                       relation_matrix: torch.Tensor):
        """Plot relation lines and matrix."""
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

    def setup_predicates(self, executor : 'CentralExecutor'):
        """Setup all direction predicates with correct type signatures."""
        from rinarak.program import Primitive
        from rinarak.types import treal, tvector

    
        # Define base types
        state_type = tvector(treal, 2)  # vector[float,2]
        position_type = tvector(treal, 2)  # vector[float,2]
        angle_type = treal  # float
    
        executor.update_registry({
        "get_position": Primitive(
            "get_position",
            arrow(state_type, position_type),
            lambda x: {**x, "end": x["state"]}
        ),
        
        "north": Primitive(
            "north", 
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.north(x["state"], y["state"])}
        ),
        
        "south": Primitive(
            "south",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.south(x["state"], y["state"])}
        ),
        
        "east": Primitive(
            "east",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.east(x["state"], y["state"])}
        ),
        
        "west": Primitive(
            "west",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.west(x["state"], y["state"])}
        ),
        
        "northeast": Primitive(
            "northeast",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.northeast(x["state"], y["state"])}
        ),
        
        "northwest": Primitive(
            "northwest",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.northwest(x["state"], y["state"])}
        ),
        
        "southeast": Primitive(
            "southeast",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.southeast(x["state"], y["state"])}
        ),
        
        "southwest": Primitive(
            "southwest",
            arrow(state_type, arrow(state_type, boolean)),
            lambda x: lambda y: {**x, "end": self.southwest(x["state"], y["state"])}
        ),
        
        "angle_between": Primitive(
            "angle_between",
            arrow(state_type, arrow(state_type, angle_type)),
            lambda x: lambda y: {**x, "end": self.compute_angle(x["state"], y["state"])}
        )
        })
    
def build_direction_executor(temperature: float = 0.2) -> CentralExecutor:
    """Build direction executor with domain.
    
    Args:
        temperature: Temperature for smooth operations
        
    Returns:
        Initialized direction executor
    """
    domain = load_domain_string(DIRECTION_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    direction_domain = DirectionalDomain(temperature)
    direction_domain.setup_predicates(executor)
    
    # Add visualization method to executor

    executor.visualize = direction_domain.visualize

    constraints = {
    "north": 2,
    "south": 2,
    "east": 2,
    "west": 2,
    "northeast": 2,
    "northwest": 2,
    "southeast": 2,
    "southwest": 2
    }
    executor.costraints = constraints
    
    return executor

# Create default executor instance
direction_executor = build_direction_executor()