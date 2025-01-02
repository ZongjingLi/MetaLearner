'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-11-03 04:19:24
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-11-05 08:23:58
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch
import numpy as np
from rinarak.knowledge.executor import CentralExecutor
from domains.utils import domain_parser, load_domain_string, build_domain_dag
from scipy.spatial import ConvexHull, Delaunay
from rinarak.utils.tensor import logit

folder_path = os.path.dirname(__file__)
curve_domain_str = ""
with open(f"{folder_path}/curve_domain.txt","r") as domain:
        for line in domain: curve_domain_str += line
executor_domain = load_domain_string(curve_domain_str, domain_parser)

"""load the state decode the latent representation of curves and lines"""
from .curve_repr import PointCloudVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
curve_vae = PointCloudVAE(num_points=320, latent_dim=64)
curve_vae.load_state_dict(torch.load(f"{folder_path}/curve_vae_state.pth", map_location = device))

num_points = 320

def decode_curve(state):return curve_vae.decoder(state)

def start_mask(state): return torch.zeros([num_points, 1])

def end_mask(state) : return torch.zeros([num_points, 1])

def at_curve(state1, state2): return 1.0

def curve_length(state): return 1.0

def curve_intersect(state1, state2): return 1.0


def is_line(state1): return torch.ones([state1.shape[0]])


"""start the construction of simple geometric pointcloud encoder"""

# make a vae to decode state explicilty
concept_dim = 100
curve_executor = CentralExecutor(executor_domain, "cone", concept_dim)


"""write the construction matrix using the given locator"""

# decode the [bxd] state vector to [bxnum_pointsx2]
curve_executor.redefine_predicate(
    "curve-geometry", lambda x : {**x , "end": decode_curve(x["state"])})

curve_executor.redefine_predicate(
    "start", lambda x : {**x , "end": start_mask(x["state"])})

curve_executor.redefine_predicate(
    "end", lambda x : {**x , "end": end_mask(x["state"])})

curve_executor.redefine_predicate(
    "at-curve", lambda x : lambda y : {**x , "end": at_curve(x["state"], y["state"])})

curve_executor.redefine_predicate(
    "is-line", lambda x : {**x , "end": is_line(x["state"])})

"""visualize the give state representation and save the figure under outputs"""

import colorsys
import random

def generate_bright_colors(n=1, method='pastel'):
    """
    Generate bright, visually appealing colors using different methods.
    
    Args:
        n (int): Number of colors to generate
        method (str): Color generation method:
            'pastel': Bright pastel colors
            'golden': Golden ratio-based colors
            'neon': Vibrant neon colors
            'random': Randomly select from above methods
    
    Returns:
        list: List of RGB colors in hex format
    """
    
    def pastel_color():
        # Generate bright pastel by mixing with white
        hue = random.random()
        saturation = random.uniform(0.4, 0.8)
        value = random.uniform(0.9, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    
    def golden_ratio_color(i):
        # Use golden ratio to generate well-distributed colors
        golden_ratio = (1 + 5 ** 0.5) / 2
        hue = (i * golden_ratio) % 1
        saturation = random.uniform(0.6, 0.9)
        value = random.uniform(0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    
    def neon_color():
        # Generate vibrant neon colors
        hue = random.random()
        saturation = random.uniform(0.8, 1.0)
        value = 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    
    # Color palette presets for better visual harmony
    preset_palettes = [
        ['#FF61E6', '#7CFFCB', '#4EA8FF', '#FFB86B', '#FF6B6B'],  # Retro neon
        ['#FF9ECD', '#FFD300', '#4BC0C0', '#36A2EB', '#9966FF'],   # Pastel pop
        ['#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33']    # Vibrant
    ]
    
    colors = []
    for i in range(n):
        if method == 'pastel':
            colors.append(pastel_color())
        elif method == 'golden':
            colors.append(golden_ratio_color(i))
        elif method == 'neon':
            colors.append(neon_color())
        elif method == 'preset':
            palette = random.choice(preset_palettes)
            colors.append(palette[i % len(palette)])
        elif method == 'random':
            color_method = random.choice(['pastel', 'golden', 'neon', 'preset'])
            colors.append(generate_bright_colors(1, color_method)[0])
    
    return colors

def visualize_curve_states(states, executor, filename = "curve_states"):
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    max_arity = 3

    context = {
                i : {"state": states, "end": 1.0} for i in range(max_arity)
        }
    results = {p : executor.evaluate(f"({p} $0)", context) for p in ["curve-geometry", "start", "end"]}
    # Create figure with transparent background
    plt.figure(figsize=(8, 8))  # Square figure
    ax = plt.gca()
    #print(results)
    
    # Set transparent background
    ax.set_facecolor('none')
    plt.gcf().patch.set_alpha(0.0)
    
    # Plot curves
    points = results['curve-geometry']["end"].detach()  # [num_points x 2] array
    num_curves = points.shape[0]
    colors = generate_bright_colors(num_curves)

    for b in range(num_curves):
        plt.scatter(points[b, :, 0], points[b, :, 1], color = colors[b], linewidth=1, alpha=0.7)
    
    # Plot start points
    points = results['start']["end"].detach()  # [num_points x 2] array
    #plt.scatter(points[:, 0], points[:, 1], color='green', marker='o', s=100, label='Start', zorder=3)
    
    # Plot end points
    points = results['end']["end"].detach()  # [num_points x 2] array
    #plt.scatter(points[:, 0], points[:, 1], color='red', marker='x', s=100, label='End', zorder=3)
    
    plt.axis('equal')
    plt.axis('off')
    
    Path("outputs").mkdir(exist_ok=True)

    plt.savefig(os.path.join('outputs', filename), 
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=True,
                dpi=300)
    
    plt.close()

if __name__ == "__main__":
        visualize_curve_states(torch.randn([5,64]), curve_executor)