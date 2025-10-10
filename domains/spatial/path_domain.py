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
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline


__all__ = [
    'PathDomain',
    'path_executor'
]

path_domain_str = """
(domain Path)
(def type
    path - vector[float, 4] ;; start point and end point
    point - vector[float,2] ;; a point
)
(def function
    sample_path -> path                   ;; random choose a pair of points and create spline path
    start_point ?x-path -> point          ;; the start point of the path (first element)
    end_point ?x-path -> point            ;; the end point of the path (last element)
    point_on_body ?x-path -> point        ;; random select a point on the body of the path (not end or start)

    intersect ?x-path ?y-path -> boolean  ;; if two path intersect with each other, it means if at least one pair of points from each path is close
    on_path ?x-point ?y-path -> boolean   ;; if a point is on the path ,basically means if the start point is close to at least one point on the path
    on_body ?x-point ?y-path -> boolean   ;; same for on path except not on the start or end point

    path_from ?x-path ?y-point -> boolean ;; assert the start point of a path is close to the given point
    path_to ?x-path ?y-point -> boolean   ;; assert the  end  point of a path is close to the given point

    attach ?x-path ?y-point -> path       ;; move a path so it starts at a point
)
(def constraint
)
"""
path_domain = load_domain_string(path_domain_str)

class PathExecutor(CentralExecutor):
    """Domain executor for path operations and relationships."""
    
    def __init__(self, domain, tolerance=0.5):
        super().__init__(domain)
        self.tolerance = tolerance  # Tolerance for 'close' points
    
    def _close(self, point1, point2):
        """Calculate if two points are close in a differentiable way."""
        dist = torch.norm(point1 - point2, dim=-1)
        return torch.exp(-(dist ** 2) / (2 * self.tolerance ** 2))
    
    def sample_path(self, batch_size=1):    
        n_steps = 10
        n_control_points = 3
    
        paths = torch.zeros(batch_size, n_steps, 2)
    
        for b in range(batch_size):
            # Generate random start and end points
            start_point = torch.rand(2) * 10 - 5
            end_point = torch.rand(2) * 10 - 5
        
            # Create time points for control points
            t_control = torch.linspace(0, 1, n_control_points)
        
            # Initialize control points
            control_points = torch.zeros(n_control_points, 2)
            control_points[0] = start_point
            control_points[-1] = end_point
        
            # Calculate direct vector from start to end
            direct_vector = end_point - start_point
            direct_dist = torch.norm(direct_vector)
        
            # Create perpendicular vector for adding variation
            perp_vector = torch.tensor([-direct_vector[1].item(), direct_vector[0].item()])
            perp_vector = perp_vector / torch.norm(perp_vector) * direct_dist * 0.5
        
            # Set intermediate control points
            for i in range(1, n_control_points - 1):
                # Base position along direct path
                base_pos = start_point + direct_vector * t_control[i]
            
                # Add random offset perpendicular to the direct path
                random_offset = (torch.rand(1).item() * 2 - 1) * perp_vector
            
                # Set the control point
                control_points[i] = base_pos + random_offset
            
            # Convert to numpy for cubic spline
            control_points_np = control_points.numpy()
            t_control_np = t_control.numpy()
        
            # Create splines for x and y coordinates
            from scipy.interpolate import CubicSpline
            cs_x = CubicSpline(t_control_np, control_points_np[:, 0])
            cs_y = CubicSpline(t_control_np, control_points_np[:, 1])
        
            # Evaluate the spline at n_steps points
            t_eval = np.linspace(0, 1, n_steps)
            x_spline = cs_x(t_eval)
            y_spline = cs_y(t_eval)
        
            # Combine x and y coordinates
            path_np = np.column_stack((x_spline, y_spline))
        
            paths[b] = torch.tensor(path_np, dtype=torch.float32)
    
        return paths
    
    def start_point(self, paths):
        """Return the start points of the paths."""
        return paths[:, 0]
    
    def end_point(self, paths):
        """Return the end points of the paths."""
        return paths[:, -1]
    
    def point_on_body(self, paths):
        """Return random points on the body of the paths (not endpoints)."""
        batch_size = paths.shape[0]
        # Randomly select indices (not first or last)
        indices = torch.randint(1, paths.shape[1]-1, (batch_size,))
        
        # Select points
        return torch.stack([paths[i, idx] for i, idx in enumerate(indices)])
    
    def intersect(self, paths1, paths2):
        """Check if two sets of paths intersect."""
        batch_size1 = paths1.shape[0]
        batch_size2 = paths2.shape[0]
        
        # Expand to compare all points from both paths
        expanded_paths1 = paths1.unsqueeze(1).repeat(1, batch_size2, 1, 1)
        expanded_paths2 = paths2.unsqueeze(0).repeat(batch_size1, 1, 1, 1)
        
        # Reshape for point-wise comparison
        points1 = expanded_paths1.reshape(batch_size1, batch_size2, -1, 2)
        points2 = expanded_paths2.reshape(batch_size1, batch_size2, -1, 2)
        
        # Compare all points from path1 with all points from path2
        # Calculate distances between all pairs of points
        points1_expanded = points1.unsqueeze(3)
        points2_expanded = points2.unsqueeze(2)
        
        dists = torch.norm(points1_expanded - points2_expanded, dim=-1)
        
        # Calculate closeness for each pair
        closeness = torch.exp(-(dists ** 2) / (2 * self.tolerance ** 2))
        
        # If any point pair is close, the paths intersect
        return torch.max(closeness, dim=-1).values.max(dim=-1).values
    
    def on_path(self, points, paths):
        """Check if points are on paths."""
        batch_size_points = points.shape[0]
        batch_size_paths = paths.shape[0]
        
        # Expand to compare each point with each path
        expanded_points = points.unsqueeze(1).repeat(1, batch_size_paths, 1)
        expanded_paths = paths.unsqueeze(0).repeat(batch_size_points, 1, 1, 1)
        
        # Calculate distances from points to all points on paths
        expanded_points_reshaped = expanded_points.unsqueeze(2)
        dists = torch.norm(expanded_points_reshaped - expanded_paths, dim=-1)
        
        # Find minimum distance for each point-path pair
        min_dists = torch.min(dists, dim=-1).values
        
        # Convert to closeness metric
        return torch.exp(-(min_dists ** 2) / (2 * self.tolerance ** 2))
    
    def on_body(self, points, paths):
        """Check if points are on the body of paths (not endpoints)."""
        batch_size_points = points.shape[0]
        batch_size_paths = paths.shape[0]
        
        # Expand to compare each point with each path
    
        expanded_points = points.unsqueeze(1).repeat(1, batch_size_paths, 1)
        expanded_paths = paths.unsqueeze(0).repeat(batch_size_points, 1, 1, 1)
        
        # Extract path bodies (excluding endpoints)
        path_bodies = expanded_paths[:, :, 1:-1, :]
        
        # Calculate distances from points to all points on path bodies
        expanded_points_reshaped = expanded_points.unsqueeze(2)
        dists = torch.norm(expanded_points_reshaped - path_bodies, dim=-1)
        
        # Find minimum distance for each point-path pair
        min_dists = torch.min(dists, dim=-1).values
        
        # Convert to closeness metric
        return torch.exp(-(min_dists ** 2) / (2 * self.tolerance ** 2))
    
    def path_from(self, paths, points):
        """Check if paths start from points."""
        start_points = self.start_point(paths)
        return self._close(start_points.unsqueeze(1), points.unsqueeze(0))
    
    def path_to(self, paths, points):
        """Check if paths end at points."""
        end_points = self.end_point(paths)
        return self._close(end_points.unsqueeze(1), points.unsqueeze(0))
    
    def attach(self, paths, points):
        """Move paths to start at given points."""
        batch_size_paths = paths.shape[0]
        batch_size_points = points.shape[0]
        
        # Expand to handle batch dimensions
        expanded_paths = paths.unsqueeze(1).repeat(1, batch_size_points, 1, 1)
        expanded_points = points.unsqueeze(0).unsqueeze(2).repeat(batch_size_paths, 1, 1, 1)
        
        # Get current start points
        current_starts = expanded_paths[:, :, 0:1, :]
        
        # Calculate offset for each path-point pair
        offsets = expanded_points - current_starts
        
        # Apply offset to move paths
        moved_paths = expanded_paths + offsets
        
        return moved_paths

def visualize_paths(paths, points=None, relationships=None, filename="path_visualization.png", figsize=(10, 8), highlight_indices=None):
    """
    Visualize paths and their relationships with improved contrast.
    
    Args:
        paths: Tensor of shape [batch_size, n_steps, 2] representing paths
        points: Optional tensor of shape [batch_size, 2] representing points
        relationships: Optional dictionary of relationship matrices
        filename: Path to save the visualization
        figsize: Size of the figure
        highlight_indices: Optional list of path indices to highlight
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches
    import numpy as np
    
    # Define updated teal color palette with better contrast
    path_colors = [
        (35/255, 120/255, 120/255),    # Darker Teal
        (79/255, 175/255, 182/255),    # Brighter Teal
        (15/255, 76/255, 92/255),      # Deep Teal
        (102/255, 204/255, 204/255),   # Light Teal
        (53/255, 138/255, 128/255),    # Forest Teal
        (18/255, 95/255, 98/255),      # Dark Teal
        (88/255, 184/255, 166/255),    # Mint Teal
        (31/255, 78/255, 85/255)       # Slate Teal
    ]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot paths with improved contrast
    for i, path in enumerate(paths):
        color_idx = i % len(path_colors)
        path_color = path_colors[color_idx]
        
        # Convert to numpy for plotting
        path_np = path.detach().cpu().numpy()
        
        # Determine line style and width
        line_style = '-'
        line_width = 2.5
        alpha_val = 0.9
        zorder_val = 5
        
        # Highlight specific paths if requested
        if highlight_indices is not None and i in highlight_indices:
            line_width = 3.5
            alpha_val = 1.0
            zorder_val = 20
        
        # Plot the path
        ax.plot(path_np[:, 0], path_np[:, 1], line_style, color=path_color, 
                linewidth=line_width, alpha=alpha_val, zorder=zorder_val, label=f"Path {i}")
        
        # Mark start and end points with improved visibility
        ax.scatter(path_np[0, 0], path_np[0, 1], color=path_color, s=100, 
                   marker='o', edgecolor='white', linewidth=1.5, zorder=zorder_val+1, label="_nolegend_")
        ax.scatter(path_np[-1, 0], path_np[-1, 1], color=path_color, s=100, 
                   marker='s', edgecolor='white', linewidth=1.5, zorder=zorder_val+1, label="_nolegend_")
    
    # Plot points if provided
    if points is not None:
        points_np = points.detach().cpu().numpy()
        ax.scatter(points_np[:, 0], points_np[:, 1], color='#2c5d5d', s=120, 
                   marker='*', edgecolor='white', linewidth=1.8, zorder=25, label="Points")
    
    # Plot relationships if provided
    if relationships is not None:
        for rel_name, rel_matrix in relationships.items():
            # Skip non-matrix relationships
            if not isinstance(rel_matrix, torch.Tensor) or rel_matrix.dim() < 2:
                continue
                
            rel_np = rel_matrix.detach().cpu().numpy()
            
            # Draw lines for relationships between paths/points
            for i in range(rel_np.shape[0]):
                for j in range(rel_np.shape[1]):
                    # Only draw significant relationships
                    strength = rel_np[i, j]
                    if strength > 0.4:  # Lower threshold to show more relationships
                        relationship_color = '#3a7c7c'
                        line_style = '--'
                        
                        # Different line styles for different relationships
                        if rel_name == "intersect":
                            x1, y1 = paths[i, 0].detach().cpu().numpy()
                            x2, y2 = paths[j, 0].detach().cpu().numpy()
                            relationship_color = '#3a7c7c'
                            line_style = '--'
                        elif rel_name == "path_from":
                            if points is not None:
                                x1, y1 = paths[i, 0].detach().cpu().numpy()
                                x2, y2 = points[j].detach().cpu().numpy()
                                relationship_color = '#c06060'  # Reddish for "from"
                                line_style = '-.'
                        elif rel_name == "path_to":
                            if points is not None:
                                x1, y1 = paths[i, -1].detach().cpu().numpy()
                                x2, y2 = points[j].detach().cpu().numpy()
                                relationship_color = '#6060c0'  # Bluish for "to"
                                line_style = ':'
                        elif rel_name in ["on_path", "on_body"]:
                            if points is not None:
                                x1, y1 = points[i].detach().cpu().numpy()
                                x2, y2 = paths[j, 0].detach().cpu().numpy()
                                relationship_color = '#60a060'  # Greenish
                                line_style = ':'
                        
                        # Draw the relationship line
                        ax.plot([x1, x2], [y1, y2], line_style, color=relationship_color, 
                               alpha=min(strength+0.2, 1.0), linewidth=1.8, zorder=3)
    
    # Add legend with improved styling
    handles, labels = ax.get_legend_handles_labels()
    path_legend = [mpatches.Patch(color=path_colors[i % len(path_colors)], 
                                 label=f"Path {i}") for i in range(len(paths))]
    
    # Add legend entries for start/end markers with improved visibility
    start_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=path_colors[0], 
                             markersize=10, label="Start Point")
    end_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=path_colors[0], 
                           markersize=10, label="End Point")
    
    # Create legend
    legend_items = path_legend + [start_marker, end_marker]
    if points is not None:
        point_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#2c5d5d', 
                                 markersize=12, label="Points")
        legend_items.append(point_marker)
    
    # Style and add the legend
    ax.legend(handles=legend_items, loc='upper right', 
              framealpha=0.9, facecolor='#f9f9f9', edgecolor='#cccccc')
    
    # Set title and labels with improved styling
    title_props = {'fontsize': 18, 'fontweight': 'bold', 'color': '#1d6363'}
    ax.set_title("Path Domain Visualization", **title_props)
    
    label_props = {'fontsize': 12, 'color': '#3a7c7c', 'fontweight': 'bold'}
    ax.set_xlabel("X", **label_props)
    ax.set_ylabel("Y", **label_props)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Set grid with improved styling
    ax.grid(True, linestyle='-', color='#e9e9e9', alpha=0.7)
    
    # Clean up ticks
    ax.tick_params(axis='both', colors='#3a7c7c', labelsize=10)
    
    # Ensure proper plot dimensions
    plt.tight_layout()
    
    # Save figure with improved quality
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, ax

class PathDataset(Dataset):
    def __init__(self, num_samples=10000, n_steps=10, n_control_points=2):
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.n_control_points = n_control_points  # Number of control points for the spline
        self.paths = []
        self.start_points = []
        self.end_points = []
        
        for _ in range(num_samples):
            # Generate random start and end points
            start_point = np.random.uniform(-5, 5, size=(2,))
            end_point = np.random.uniform(-5, 5, size=(2,))
            
            # Generate a smooth path using cubic spline interpolation
            # First, create some random control points between start and end
            t_control = np.linspace(0, 1, self.n_control_points)
            
            # First and last control points are the start and end points
            control_points = np.zeros((self.n_control_points, 2))
            control_points[0] = start_point
            control_points[-1] = end_point
            
            # Generate intermediate control points with some randomness
            # Calculate the direct vector from start to end
            direct_vector = end_point - start_point
            direct_dist = np.linalg.norm(direct_vector)
            
            # Create a perpendicular vector for adding variation
            perp_vector = np.array([-direct_vector[1], direct_vector[0]])
            perp_vector = perp_vector / np.linalg.norm(perp_vector) * direct_dist * 0.5
            
            # Set intermediate control points
            for i in range(1, self.n_control_points - 1):
                # Base position along direct path
                base_pos = start_point + direct_vector * t_control[i]
                
                # Add random offset perpendicular to the direct path
                random_offset = np.random.uniform(-1, 1) * perp_vector
                
                # Set the control point
                control_points[i] = base_pos + random_offset
            
            # Create the spline
            x_control = control_points[:, 0]
            y_control = control_points[:, 1]
            
            # Create splines for x and y coordinates
            cs_x = CubicSpline(t_control, x_control)
            cs_y = CubicSpline(t_control, y_control)
            
            # Evaluate the spline at n_steps points
            t_eval = np.linspace(0, 1, n_steps)
            x_spline = cs_x(t_eval)
            y_spline = cs_y(t_eval)
            
            # Combine x and y coordinates
            path = np.column_stack((x_spline, y_spline))
            
            self.paths.append(path.astype(np.float32))
            self.start_points.append(start_point.astype(np.float32))
            self.end_points.append(end_point.astype(np.float32))
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'path': self.paths[idx],
            'start': self.start_points[idx],
            'end': self.end_points[idx]
        }

def demonstrate_path_attachment():
    """Demonstrate attaching a path to a point on another path's body."""
    # Create the executor
    executor = PathExecutor(path_domain)
    
    # Generate two paths for demonstration
    source_path = executor.sample_path(1)[0]  # Path to be moved
    target_path = executor.sample_path(1)[0]  # Path to attach to
    
    # Get a point from the body of the target path
    # We'll select a specific point rather than random for better visualization
    body_point_index = target_path.shape[0] // 2  # Middle of path
    attachment_point = target_path[body_point_index].unsqueeze(0)
    
    # Attach the source path to the attachment point
    attached_path = executor.attach(source_path.unsqueeze(0), attachment_point)[0, 0]
    
    # Combine paths for visualization
    all_paths = torch.stack([source_path, target_path, attached_path])
    
    # Create a list of points for visualization
    points = attachment_point
    
    # Highlight the source path and attached result
    highlight_indices = [0, 2]
    
    # Create visualization showing attachment process
    visualize_paths(
        all_paths, 
        points, 
        filename="outputs/path_attachment.png",
        highlight_indices=highlight_indices
    )
    
    # Create a second visualization with annotations
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot paths with custom styling for clarity
    path_colors = [
        (35/255, 120/255, 120/255),    # Original path
        (15/255, 76/255, 92/255),      # Target path
        (102/255, 204/255, 204/255),   # Attached path
    ]
    
    # Plot source path
    source_np = source_path.detach().cpu().numpy()
    ax.plot(source_np[:, 0], source_np[:, 1], '-', color=path_colors[0], 
            linewidth=2.5, alpha=0.9, label="Source Path")
    ax.scatter(source_np[0, 0], source_np[0, 1], color=path_colors[0], s=100, 
               marker='o', edgecolor='white', linewidth=1.5, zorder=10)
    ax.scatter(source_np[-1, 0], source_np[-1, 1], color=path_colors[0], s=100, 
               marker='s', edgecolor='white', linewidth=1.5, zorder=10)
    
    # Plot target path
    target_np = target_path.detach().cpu().numpy()
    ax.plot(target_np[:, 0], target_np[:, 1], '-', color=path_colors[1], 
            linewidth=2.5, alpha=0.9, label="Target Path")
    ax.scatter(target_np[0, 0], target_np[0, 1], color=path_colors[1], s=100, 
               marker='o', edgecolor='white', linewidth=1.5, zorder=10)
    ax.scatter(target_np[-1, 0], target_np[-1, 1], color=path_colors[1], s=100, 
               marker='s', edgecolor='white', linewidth=1.5, zorder=10)
    
    # Plot attached path
    attached_np = attached_path.detach().cpu().numpy()
    ax.plot(attached_np[:, 0], attached_np[:, 1], '-', color=path_colors[2], 
            linewidth=3.5, alpha=1.0, label="Attached Path")
    ax.scatter(attached_np[0, 0], attached_np[0, 1], color=path_colors[2], s=120, 
               marker='o', edgecolor='white', linewidth=2.0, zorder=20)
    ax.scatter(attached_np[-1, 0], attached_np[-1, 1], color=path_colors[2], s=120, 
               marker='s', edgecolor='white', linewidth=2.0, zorder=20)
    
    # Plot attachment point
    point_np = attachment_point.detach().cpu().numpy()
    ax.scatter(point_np[:, 0], point_np[:, 1], color='#d63031', s=150, 
               marker='*', edgecolor='white', linewidth=2.0, zorder=30, label="Attachment Point")
    
    # Add arrows and annotations
    attachment_pt = point_np[0]
    source_start = source_np[0]
    attached_start = attached_np[0]
    
    # Arrow from source to attachment point 
    ax.annotate("", xy=attachment_pt, xytext=source_start,
                arrowprops=dict(arrowstyle="->", lw=2, color="#666666"))
    
    # Add explanatory text
    ax.text(attachment_pt[0] + 0.5, attachment_pt[1] + 0.5, 
            "Attachment\nPoint", color='#d63031', fontsize=12, fontweight='bold',
            ha='left', va='bottom')
    
    # Position for the explanation text
    x_pos = min(ax.get_xlim()[0] + 1, min(point_np[:, 0]) - 4)
    y_pos = max(ax.get_ylim()[1] - 1, max(point_np[:, 1]) + 2)
    
    explanation = "The source path (dark teal) is\nattached to start from the\nselected point (red star)\non the target path (deep teal).\nThe resulting path is shown\nin light teal."
    
    ax.text(x_pos, y_pos, explanation, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.5'),
            fontsize=12, color='#333333')
    
    # Style the plot
    ax.legend(loc='upper right', framealpha=0.9, facecolor='#f9f9f9', 
              edgecolor='#cccccc', fontsize=12)
    
    title_props = {'fontsize': 18, 'fontweight': 'bold', 'color': '#1d6363'}
    ax.set_title("Path Attachment Demonstration", **title_props)
    
    label_props = {'fontsize': 12, 'color': '#3a7c7c', 'fontweight': 'bold'}
    ax.set_xlabel("X", **label_props)
    ax.set_ylabel("Y", **label_props)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='-', color='#e9e9e9', alpha=0.7)
    ax.tick_params(axis='both', colors='#3a7c7c', labelsize=10)
    
    plt.tight_layout()
    plt.savefig("outputs/path_attachment_annotated.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Path attachment demonstration completed!")
def path_domain_demo():
    """Demonstrate the path domain functionality with improved visualizations."""
    # Create the executor
    executor = PathExecutor(path_domain)
    
    # Generate some random paths with increased diversity
    n_paths = 4
    paths = executor.sample_path(n_paths)
    
    # Basic path visualization with improved contrast
    visualize_paths(paths, filename="outputs/basic_paths.png")
    
    # Generate some points
    n_points = 5
    points = torch.rand(n_points, 2) * 10 - 5
    
    # Check which paths intersect
    intersect_matrix = executor.intersect(paths, paths)
    
    # Check which points are on paths
    on_path_matrix = executor.on_path(points, paths)
    
    # Visualize paths with intersections and points
    relationships = {
        "intersect": intersect_matrix,
        "on_path": on_path_matrix
    }
    visualize_paths(paths, points, relationships, filename="outputs/path_relationships.png")
    
    # Demonstrate attaching paths to points
    body_points = executor.point_on_body(paths)
    attached_paths = executor.attach(paths[:2], body_points[:3])
    
    # Reshape to handle the batch dimensions
    attached_paths_reshaped = attached_paths.reshape(-1, paths.shape[1], 2)
    
    # Visualize attached paths with highlighting
    highlight_indices = list(range(attached_paths_reshaped.shape[0]))
    visualize_paths(attached_paths_reshaped, body_points[:3], 
                   filename="outputs/attached_paths.png",
                   highlight_indices=highlight_indices[:2])
    
    # Demonstrate path_from and path_to with improved visualization
    path_from_matrix = executor.path_from(paths, points)
    path_to_matrix = executor.path_to(paths, points)
    
    relationships = {
        "path_from": path_from_matrix,
        "path_to": path_to_matrix
    }
    visualize_paths(paths, points, relationships, filename="outputs/path_endpoints_relationships.png")
    
    # Demonstrate attaching a path to a point on another path's body
    demonstrate_path_attachment()
    
    print("Path domain demonstration completed! Check the visualization files in the outputs directory.")

# Run the demo
if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Run the demos
    path_domain_demo()


path_executor = PathExecutor(path_domain)

