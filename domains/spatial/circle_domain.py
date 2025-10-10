import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from domains.utils import BatchVisualizer

circle_domain_str = """
(domain :: Circle)
(def type
    state - Embedding[circle,3]
    region - float
)
(def function
    unit_circle : state := by pass
    area (x : state) : region := by pass
    disconnected (x y : state) : boolean := by pass
    externally_connected (x y : state) : boolean := by pass
    partial_overlap (x y : state) : boolean := by pass
    equal (x y : state) : boolean := by pass
    tangential_proper_part (x y : state) : boolean := by pass
    non_tangential_proper_part (x y : state) : boolean := by pass
    tangential_proper_part_inverse (x y : state) : boolean := by pass
    non_tangential_proper_part_inverse (x y : state) : boolean := by pass
)
"""

circle_domain = load_domain_string(circle_domain_str)

class CircleExecutor(CentralExecutor, BatchVisualizer):
    """
    A domain for spatial relationships between circles
    state vector[x, y, r] represents a circle with center (x,y) and radius r
    """
    def __init__(self, domain, temperature=10.0):
        super().__init__(domain)
        self.temperature = temperature  # Controls sharpness of transitions
    
    def area(self, state):
        # Simply return a representation of the circle (could be area or another metric)
        # Here we'll use the area of the circle
        return torch.pi * state[:, 2]**2
    

    def unit_circle(self): return torch.tensor([0.0, 0.0, 1.0]).reshape([1,3])
    
    def preprocess(self, state1, state2):
        state1 = state1.reshape([-1,3])
        state2 = state2.reshape([-1,3])
        return state1, state2

    def distance_between_centers(self, state1, state2):
        # Calculate Euclidean distance between circle centers
        state1 = state1.reshape([-1,3])
        state2 = state2.reshape([-1,3])
        state1, state2 = self.preprocess(state1, state2)
        return torch.sqrt((state1[:, 0] - state2[:, 0])**2 + 
                          (state1[:, 1] - state2[:, 1])**2)
    
    def disconnected(self, state1, state2):
        # Two circles are disconnected if the distance between their centers
        # is greater than the sum of their radii
        state1, state2 = self.preprocess(state1, state2)
        dist = self.distance_between_centers(state1, state2)
        radii_sum = state1[:, 2] + state2[:, 2]
        return self.temperature * (dist - radii_sum)

    def externally_connected(self, state1, state2):
        state1, state2 = self.preprocess(state1, state2)
        # Two circles are externally connected if they touch at exactly one point
        # This happens when the distance between centers equals the sum of radii
        dist = self.distance_between_centers(state1, state2)
        radii_sum = state1[:, 2] + state2[:, 2]
        # Use a Gaussian-like function centered at zero difference for smooth transition
        diff = dist - radii_sum
        return torch.logit(torch.exp(-(diff**2) * self.temperature))
  
    def partial_overlap(self, state1, state2):
        state1, state2 = self.preprocess(state1, state2)
        # Circles partially overlap when distance between centers is less than
        # the sum of radii but greater than the absolute difference of radii
        dist = self.distance_between_centers(state1, state2)
        radii_sum = state1[:, 2] + state2[:, 2]
        radii_diff = torch.abs(state1[:, 2] - state2[:, 2])
        upper_bound = self.temperature * (radii_sum - dist)
        lower_bound = self.temperature * (dist - radii_diff)
 
        return upper_bound + lower_bound
    
    def equal(self, state1, state2):
        state1, state2 = self.preprocess(state1, state2)
        # Circles are equal if their centers and radii are the same
        center_dist = self.distance_between_centers(state1, state2)
        radii_diff = torch.abs(state1[:, 2] - state2[:, 2])

        return -self.temperature * (center_dist + radii_diff)
 
    def tangential_proper_part(self, state1, state2):
        # Circle 1 is a tangential proper part of circle 2 if:
        # 1. Circle 1 is inside circle 2
        # 2. The inner circle touches the boundary of the outer circle
        state1, state2 = self.preprocess(state1, state2)
        dist = self.distance_between_centers(state1, state2)
        # Distance between centers plus radius of inner circle should equal radius of outer circle
        diff = torch.abs(dist + state1[:, 2] - state2[:, 2])
        # Also ensure the inner circle is smaller
        size_check = torch.sigmoid(self.temperature * (state2[:, 2] - state1[:, 2]))
        
        return torch.logit(torch.exp(-(diff**2) * self.temperature) * size_check)
    
    def non_tangential_proper_part(self, state1, state2):
        # Circle 1 is a non-tangential proper part of circle 2 if:
        # 1. Circle 1 is completely inside circle 2
        # 2. There's a gap between the boundary of circle 1 and circle 2
        state1, state2 = self.preprocess(state1, state2)
        dist = self.distance_between_centers(state1, state2)
        # Distance between centers plus radius of inner circle should be less than radius of outer circle
        contained = self.temperature * (state2[:, 2] - (dist + state1[:, 2]))
        # Also ensure the inner circle is smaller
        size_check = self.temperature * (state2[:, 2] - state1[:, 2])
        
        return contained + size_check
    
    def tangential_proper_part_inverse(self, state1, state2):
        # This is just the inverse of tangential_proper_part
        
        return self.tangential_proper_part(state2, state1)
    
    def non_tangential_proper_part_inverse(self, state1, state2):
        # This is just the inverse of non_tangential_proper_part
        return self.non_tangential_proper_part(state2, state1)
    
    def visualize(self, batched_data, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Circle
        
        # Convert batched data to numpy for visualization
        if isinstance(batched_data, torch.Tensor):
            circles = batched_data.detach().cpu().numpy()
        else:
            circles = np.array(batched_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # First subplot: Visualize circles
        ax1.set_aspect('equal')
        ax1.set_title('Circle Visualization')
        
        # Calculate plot boundaries
        max_radius = np.max(circles[:, 2])
        all_x = circles[:, 0]
        all_y = circles[:, 1]
        x_min, x_max = np.min(all_x) - max_radius, np.max(all_x) + max_radius
        y_min, y_max = np.min(all_y) - max_radius, np.max(all_y) + max_radius
        
        # Add some padding
        width = x_max - x_min
        height = y_max - y_min
        padding = max(width, height) * 0.1
        ax1.set_xlim(x_min - padding, x_max + padding)
        ax1.set_ylim(y_min - padding, y_max + padding)
        
        # Draw each circle
        for i, (x, y, r) in enumerate(circles):
            circle = Circle((x, y), r, fill=False, edgecolor=f'C{i}', linewidth=2)
            ax1.add_patch(circle)
            ax1.text(x, y, f"{i}", ha='center', va='center')
        
        # Second subplot: Relationship matrix
        n = len(circles)
        
        # Create tensors for evaluation
        circle_tensors = [torch.tensor(circle).unsqueeze(0) for circle in circles]
        
        # Define predicates to visualize
        predicates = [
            ("DC", self.disconnected),
            ("EC", self.externally_connected),
            ("PO", self.partial_overlap),
            ("EQ", self.equal),
            ("TPP", self.tangential_proper_part),
            ("NTPP", self.non_tangential_proper_part),
            ("TPPi", self.tangential_proper_part_inverse),
            ("NTPPi", self.non_tangential_proper_part_inverse)
        ]
        
        # Build relationship matrix
        relationship_matrix = np.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                if i == j:
                    relationship_matrix[i, j] = "EQ"
                    continue
                    
                # Find the strongest relationship
                max_val = -1
                max_rel = ""
                for rel_name, pred_func in predicates:
                    val = pred_func(circle_tensors[i], circle_tensors[j]).item()
                    if val > max_val:
                        max_val = val
                        max_rel = rel_name
                
                relationship_matrix[i, j] = max_rel
        
        # Create a table for the matrix
        ax2.set_title('RCC8 Spatial Relationships')
        ax2.axis('off')
        table = ax2.table(
            cellText=relationship_matrix,
            cellLoc='center',
            loc='center',
            rowLabels=[f"Circle {i}" for i in range(n)],
            colLabels=[f"Circle {i}" for i in range(n)]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)
        
        # Add a legend for the relationships
        ax2.text(0.5, -0.1, 
                 "DC: Disconnected  EC: Externally Connected  PO: Partial Overlap  EQ: Equal\n" +
                 "TPP: Tangential Proper Part  NTPP: Non-Tangential Proper Part\n" +
                 "TPPi: Tangential Proper Part Inverse  NTPPi: Non-Tangential Proper Part Inverse",
                 ha='center', va='center', transform=ax2.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'),
                 fontsize=9)
        
        # Adjust layout and save if needed
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

circle_executor = CircleExecutor(circle_domain)