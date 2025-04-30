'''
 # @ Author: Assistant
 # @ Create Time: 2024-07-21
 # @ Description: Cycle domain implementation where states are points on unit circle.
 # @ This file is distributed under the MIT license.
'''
import os
import torch
import torch.nn as nn
import numpy as np
from rinarak.dklearn.nn.mlp import FCBlock
from rinarak.utils.tensor import logit
#from domains.utils import build_domain_executor
import matplotlib.pyplot as plt

from rinarak.domain import load_domain_string, Domain
from rinarak.knowledge.executor import type_dim, CentralExecutor
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it

domain_parser = Domain("domains/base.grammar")

def build_domain_executor(domain : str, embedding_dim : int = 128):
    """domain is string of the domain name file under the domain/ folder."""
    if isinstance(domain, str):
        meta_domain_str = ""
        with open(domain,"r") as domain:
            for line in domain: meta_domain_str += line
    executor_domain = load_domain_string(meta_domain_str, domain_parser)

    return CentralExecutor(executor_domain, "cone", type_dim(executor_domain.types["state"])[0][0])

# Constants for comparison thresholds
gc = 0.0  # gap constant
tc = 0.15  # temperature constant

# Setup domain
domain_name = "cycle_domain.txt"
domain_file = os.path.dirname(__file__) + "/" + domain_name
cycle_executor = build_domain_executor(domain_file)

class CircleVectorModule(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.linear0 = FCBlock(128, 2, state_dim, output_dim)
        
    def forward(self, x):
        # Project output to unit circle
        raw = self.linear0(x)
        norm = torch.norm(raw, dim=-1, keepdim=True)
        return raw / (norm + 1e-8)  # normalize to unit circle

# Redefine position predicate to handle circular positions
cycle_executor.redefine_predicate(
    "pos",
    lambda x: {**x,
               "from": "pos",
               "set": x["end"],
               "end": x["pos"] if "pos" in x else x["state"]}
)

def circular_distance(x, y):
    """Compute shortest circular distance between angles"""
    diff = torch.atan2(torch.sin(x - y), torch.cos(x - y))
    return diff

def angle_from_vector(v):
    """Convert 2D vector to angle"""
    return torch.atan2(v[..., 1], v[..., 0])

def clockwise_logits(x, y):
    """Compute logits for clockwise relationship"""
    # Convert vectors to angles
    x_angle = angle_from_vector(x["state"].view(-1, 2))  # [n]
    y_angle = angle_from_vector(y["state"].view(-1, 2))  # [m]
    
    # Reshape for broadcasting
    x_angle = x_angle.view(-1, 1)  # [n, 1]
    y_angle = y_angle.view(1, -1)  # [1, m]
    
    # Compute circular distance in clockwise direction
    diff = circular_distance(y_angle, x_angle)
    return (diff - gc) / tc

def counterclockwise_logits(x, y):
    """Compute logits for counterclockwise relationship"""
    # Convert vectors to angles
    x_angle = angle_from_vector(x["state"].view(-1, 2))  # [n]
    y_angle = angle_from_vector(y["state"].view(-1, 2))  # [m]
    
    # Reshape for broadcasting
    x_angle = x_angle.view(-1, 1)  # [n, 1]
    y_angle = y_angle.view(1, -1)  # [1, m]
    
    # Compute circular distance in counterclockwise direction
    diff = circular_distance(x_angle, y_angle)
    return (diff - gc) / tc

# Define clockwise relationship
cycle_executor.redefine_predicate(
    "clockwise",
    lambda x: lambda y: {**x,
                        "from": "counterclockwise",
                        "set": x["end"],
                        "end": clockwise_logits(x, y)}
)

# Define counterclockwise relationship
cycle_executor.redefine_predicate(
    "counterclockwise",
    lambda x: lambda y: {**x,
                        "from": "clockwise",
                        "set": x["end"],
                        "end": counterclockwise_logits(x, y)}
)

# Define opposite relationship (180 degrees apart)
def opposite_logits(x, y):
    x_angle = angle_from_vector(x["state"].view(-1, 2))
    y_angle = angle_from_vector(y["state"].view(-1, 2))
    
    # Reshape for broadcasting
    x_angle = x_angle.view(-1, 1)
    y_angle = y_angle.view(1, -1)
    
    # Check if points are approximately opposite (pi radians apart)
    diff = torch.abs(circular_distance(x_angle, y_angle))
    return -(torch.abs(diff - np.pi) - gc) / tc

cycle_executor.redefine_predicate(
    "opposite",
    lambda x: lambda y: {**x,
                        "from": "opposite",
                        "set": x["end"],
                        "end": opposite_logits(x, y)}
)

def visualize_cycle(context, save_name="cycle_repr"):
    """Visualize points on unit circle with their relationships"""
    plt.figure("cycle representation", figsize=(8, 8))
    plt.cla()
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    plt.gca().add_artist(circle)
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    
    # Plot center point
    plt.scatter(0, 0, c="blue", label="Center")
    
    # Plot points and their labels
    for i in context:
        cont = {0: context[i]}
        pos = cycle_executor.evaluate("(pos $0)", context=cont)["end"].detach()
        
        # Plot point
        plt.scatter(pos[0], pos[1], c="red", marker="*")
        
        # Add label with angle
        angle = np.degrees(angle_from_vector(pos).item())
        if angle < 0:
            angle += 360
        label_x = pos[0] * 1.2
        label_y = pos[1] * 1.2
        plt.text(label_x, label_y, f"[{i}]:{angle:.1f}°")
        
        # Draw line from center
        plt.plot([0, pos[0]], [0, pos[1]], 'k:', alpha=0.3)
    
    # Add axis lines and labels
    plt.axhline(y=0, color='k', linestyle=':')
    plt.axvline(x=0, color='k', linestyle=':')
    plt.text(1.2, 0, '0°')
    plt.text(0, 1.2, '90°')
    plt.text(-1.2, 0, '180°')
    plt.text(0, -1.2, '270°')
    
    plt.title("Cycle Domain Representation")
    plt.savefig(f"outputs/{save_name}.png")
    plt.close()

# Example usage
def test_cycle_domain():
    # Create some test points on the unit circle
    context = {
        "A": {"state": torch.tensor([1.0, 0.0]), "end" : torch.tensor([13.])},  # 0 degrees
        "B": {"state": torch.tensor([0.0, 1.0]), "end" : torch.tensor([13.])},  # 90 degrees
        "C": {"state": torch.tensor([-1.0, 0.0]), "end" : torch.tensor([13.])}, # 180 degrees
        "D": {"state": torch.tensor([0.0, -1.0]), "end" : torch.tensor([13.])}  # 270 degrees
    }
    
    # Visualize the points
    visualize_cycle(context)
    
    # Test relationships
    print("Testing clockwise relationship:")
    result = cycle_executor.evaluate("(clockwise $0 $1)", 
                                   context={0: context["A"], 1: context["D"]})
    print(f"A clockwise to D: {logit(result['end'])}")
    
    print("\nTesting opposite relationship:")
    result = cycle_executor.evaluate("(opposite $0 $1)", 
                                   context={0: context["A"], 1: context["C"]})
    print(f"A opposite to C: {logit(result['end'])}")

if __name__ == "__main__":
    test_cycle_domain()