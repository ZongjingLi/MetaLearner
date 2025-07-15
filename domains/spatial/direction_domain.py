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
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string


__all__ = [
    'DirectionalDomain',
    'direction_executor'
]


direction_domain_str = """
(domain Direction)
(:type
    position - vector[float,2]
    angle - float
)
(:predicate
    north ?x-position ?y-position -> boolean
    south ?x-position ?y-position -> boolean
    east ?x-position ?y-position -> boolean
    west ?x-position ?y-position -> boolean
    northeast ?x-position ?y-position -> boolean
    northwest ?x-position ?y-position -> boolean
    southeast ?x-position ?y-position -> boolean
    southwest ?x-position ?y-position -> boolean
    angle_between ?x-position ?y-position -> angle
)
"""
direction_domain = load_domain_string(direction_domain_str)


def angle_matrix(x, y):
    angles_x = torch.atan2(x[:, 1], x[:, 0])  # Shape: [n]
    angles_y = torch.atan2(y[:, 1], y[:, 0])  # Shape: [m]
    angles_x = angles_x.unsqueeze(1)  # Shape: [n, 1]

    angle_matrix = angles_x - angles_y  # Shape: [n, m]
    angle_matrix = (angle_matrix + torch.pi) % (2 * torch.pi) - torch.pi
    return angle_matrix

def angle_proximity(angle_matrix, target_angle, tolerance=0.1):
    """
    Calculate how close each angle in the matrix is to the target angle,
    considering the circular nature of angles (0 and 2π are the same).
    
    Parameters:
        angle_matrix: Tensor of shape [n, m] containing angles
        target_angle: Target angle in radians
        tolerance: Controls the sharpness of the proximity measure
                   (smaller values create a narrower peak)
    
    Returns:
        proximity_matrix: Tensor of shape [n, m] with values between 0 and 1,
                          where 1 means the angle is exactly the target angle
    """

    target_angle = target_angle % (2 * torch.pi)

    diff = torch.abs((angle_matrix - target_angle) % (2 * torch.pi))
    
    circular_diff = torch.min(diff, 2 * torch.pi - diff)
    
    # Convert difference to a proximity score (1 means exact match, 0 means furthest away)
    # Using a differentiable Gaussian-like function
    proximity_matrix = torch.exp(-(circular_diff ** 2) / (2 * tolerance ** 2))
    
    return proximity_matrix

class DirectionExecutor(CentralExecutor):
    def __init__(self, domain, temperature=0.2):
       super().__init__(domain)
       self.temperature = temperature
       self.margin = 0.3
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_angle(self, x_state, y_state):
       if len(x_state.shape) == 1: x_state = x_state.unsqueeze(0)
       if len(y_state.shape) == 1: y_state = y_state.unsqueeze(0)
       x_exp = x_state.unsqueeze(1)
       y_exp = y_state.unsqueeze(0)

       diff = y_exp - x_exp

       return torch.atan2(diff[..., 1] + 1e-6, diff[..., 0] + 1e-6)

    def angle_membership(self, angles, center, width):
       diff = angles - center
       
       logits =  (self.margin - torch.abs(diff)) / self.temperature
       if logits.numel() == 1: return logits.flatten()[0]
       return logits

    def north(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       #print(self.angle_membership(angles, torch.pi/2, torch.pi/2).shape)
       return self.angle_membership(angles, torch.pi/2, torch.pi/2)
   
    def south(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, -torch.pi/2, torch.pi/2)
   
    def east(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, 0.0, torch.pi/2)
   
    def west(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, torch.pi, torch.pi/2)
   
    def northeast(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, torch.pi/4, torch.pi/2)
   
    def northwest(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, 3*torch.pi/4, torch.pi/2)
   
    def southeast(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, -torch.pi/4, torch.pi/2)
   
    def southwest(self, x_state, y_state):
       angles = self.compute_angle(x_state, y_state)
       return self.angle_membership(angles, -3*torch.pi/4, torch.pi/2)
    
    def visualize(self, x, file_name):
        relations = {
        "North": self.north,
        "South": self.south,
        "East": self.east,
        "West": self.west,
        # Add or remove relations as needed
        # "Northeast": executor.northeast,
        # "Northwest": executor.northwest,
        # "Southeast": executor.southeast,
        # "Southwest": executor.southwest
        }
        visualize(x, relations, file_name)
def visualize(x, relations, filename, threshold=0.5, figsize=(12, 10)):
    """
    Visualize directional relationships using a serene teal/mint color palette.
    
    Args:
        x: Tensor of shape [n, 2] representing n points
        relations: Dictionary mapping relation names to relation functions
        filename: Path to save the visualization
        threshold: Minimum probability to draw a line
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Define serene teal/mint color palette
    relation_colors = {
    'north': (25/255, 95/255, 165/255),      # Deep blue
    'south': (185/255, 65/255, 45/255),      # Deep red
    'east': (45/255, 145/255, 75/255),       # Deep green
    'west': (15/255, 65/255, 125/255),       # Dark navy blue
    'northeast': (65/255, 125/255, 155/255), # Steel blue (blue-green mix)
    'northwest': (35/255, 85/255, 145/255),  # Medium blue
    'southeast': (85/255, 165/255, 115/255), # Teal blue (blue-green mix)
    'southwest': (135/255, 95/255, 75/255)   # Muted brown-blue
}
    
    # White background
    bg_color = '#ffffff'
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # Add very subtle gradient overlay
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
    zz = (np.sin(xx*3 + yy*4) + np.cos(xx*2 - yy*3)) * 0.02
    
    # Create custom subtle gradient colormap
    gradient_colors = [(240/255, 255/255, 240/255), (235/255, 255/255, 245/255)]
    gradient_cmap = LinearSegmentedColormap.from_list("gradient", gradient_colors, N=100)
    
    # Add subtle texture (almost invisible)
    c = ax.pcolormesh(xx*2, yy*2, zz, cmap=gradient_cmap, alpha=0.1, shading='auto')
    
    # Determine radius based on points
    radius = torch.norm(x, dim=1).max().item() * 0.8
    
    # Coordinate axes with subtle styling
    ax.axhline(y=0, color='#e0e0e0', linestyle='-', alpha=0.7, linewidth=0.8)
    ax.axvline(x=0, color='#e0e0e0', linestyle='-', alpha=0.7, linewidth=0.8)
    
    # Reference circle - teal style
    circle = plt.Circle((0, 0), radius, fill=False, 
                       color='#a9cece', alpha=0.5, linewidth=1.0)
    ax.add_artist(circle)
    
    # Plot points
    ax.scatter(x[:, 0], x[:, 1], color='#3a7c7c', s=70, zorder=10, 
              alpha=0.9, edgecolor='#ffffff', linewidth=1.0)
    
    # Add point labels
    for i, (px, py) in enumerate(x):
        ax.text(px.item() + 0.1, py.item() + 0.1, str(i), fontsize=9,
              color='#2c5d5d', fontweight='bold')
    
    # Create legend handles
    legend_handles = []
    
    # Process each relation
    n = x.shape[0]
    for rel_name, rel_func in relations.items():
        relationship_matrix = rel_func(x, x)
        base_color = relation_colors.get(rel_name.lower(), (0.2, 0.5, 0.5))
        
        lines = []
        colors = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob = relationship_matrix[i, j].item()
                    if prob > threshold:
                        # Create line
                        start = np.array([x[i, 0].item(), x[i, 1].item()])
                        end = np.array([x[j, 0].item(), x[j, 1].item()])
                        
                        line = [start, end]
                        lines.append(line)
                        
                        # Color based on probability
                        r, g, b = base_color
                        color = (r, g, b, prob)
                        colors.append(color)
        
        if lines:
            # Create subtle shadow effect
            line_collection = LineCollection(lines, 
                                           colors=[(0.9, 0.9, 0.9, 0.3) for c in colors], 
                                           linewidths=2.5,
                                           zorder=5)
            ax.add_collection(line_collection)
            
            # Main lines
            line_collection = LineCollection(lines, colors=colors, linewidths=1.5, zorder=6)
            ax.add_collection(line_collection)
            
            # Add to legend
            legend_handles.append(mpatches.Patch(
                color=base_color, 
                label=f"{rel_name.capitalize()}", 
                alpha=0.9
            ))
    
    # Set title and labels with teal styling
    title_props = {'fontsize': 16, 'fontweight': 'bold', 'color': '#1d6363'}
    ax.set_title(f"directional relations", **title_props)
    
    label_props = {'fontsize': 11, 'color': '#3a7c7c'}
    ax.set_xlabel("X", **label_props)
    ax.set_ylabel("Y", **label_props)
    
    # Add legend with teal styling
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles, 
            loc='upper right', 
            framealpha=0.7,
            prop={'size': 10},
            facecolor='#f5fffa',
            edgecolor='#a9cece'
        )
    
    # Add threshold info
    info_text = f"Threshold: {threshold}"
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, 
        verticalalignment='top',
        color='#3a7c7c',
        fontweight='bold',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#f5fffa', alpha=0.7, 
                edgecolor='#a9cece', linewidth=1)
    )
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    max_val = torch.abs(x).max().item() * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    # Set grid
    ax.grid(True, linestyle='-', color='#e9f2f2', alpha=0.6)
    
    # Clean up ticks
    ax.tick_params(axis='both', colors='#3a7c7c')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    
    return fig, ax

direction_executor = DirectionExecutor(direction_domain)

if __name__ == "__main__":
    points = torch.randn([10,2])
    executor = DirectionExecutor(direction_domain)
    executor.visualize(points, "directional_relationships.png", )