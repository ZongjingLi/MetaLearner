import torch
import torch.nn as nn
import torch.nn.functional as F
from helchriss.domain import load_domain_string
from helchriss.knowledge.executor import CentralExecutor
from domains.utils import BatchVisualizer

color_domain_str = """
(domain :: Color)
(def type
    color - Embedding[color_wheel, 1] ;; unnormalized distribution over a list of objects
)
(def function
    color   (x : color) : color := by pass ;; explicit base change to color
    red     (x : color) : boolean := by pass
    green   (x : color) : boolean := by pass
    blue    (x : color) : boolean := by pass
)
"""
color_domain = load_domain_string(color_domain_str)

class ColorDomain(CentralExecutor, BatchVisualizer):
    """here we consider color as some element on the color wheel"""
    def color(self, color_obj):
        return color_obj

    def red(self, color):
        #print("color:",color)
        # Red is centered at 0/1 on the color wheel
        redness = 0.5 * (torch.cos(2 * torch.pi * color) + 1)
        return torch.logit(redness, eps = 1e-6)

    def green(self, color):
        #print("green:",color)
        # Green is centered at 1/3 on the color wheel
        # Shift the cosine function by 1/3
        greenness = 0.5 * (torch.cos(2 * torch.pi * (color - 1/3)) + 1)
        return torch.logit(greenness, eps = 1e-6)

    def blue(self, color):
        #print("blue:",color)
        # Blue is centered at 2/3 on the color wheel
        # Shift the cosine function by 2/3
        blueness = 0.5 * (torch.cos(2 * torch.pi * (color - 2/3)) + 1)
        return torch.logit(blueness, eps = 1e-6)

    def visualize(self, batched_data, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert batched data to numpy for visualization
        if isinstance(batched_data, torch.Tensor):
            colors = batched_data.detach().cpu().numpy().flatten()
        else:
            colors = np.array(batched_data).flatten()
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First subplot: Color wheel with points
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k-')  # Draw the color wheel
        
        # Plot each point on the color wheel
        for idx, color_val in enumerate(colors):
            # Convert color position to angle (in radians)
            angle = 2 * np.pi * color_val
            x = np.cos(angle)
            y = np.sin(angle)
            
            # Get RGB values for this color
            r = float(torch.sigmoid(self.red(torch.tensor(color_val))))
            g = float(torch.sigmoid(self.green(torch.tensor(color_val))))
            b = float(torch.sigmoid(self.blue(torch.tensor(color_val))))
            
            # Plot the point with its index
            ax1.scatter(x, y, color=(r, g, b), s=100, edgecolor='black')
            ax1.text(x*1.1, y*1.1, str(idx), fontsize=10)
        
        # Set the first subplot properties
        ax1.set_aspect('equal')
        ax1.set_title('Color Wheel Positions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        
        # Second subplot: Bar chart of color values
        x = np.arange(len(colors))
        width = 0.25
        
        # Compute RGB values for each color
        r_values = np.array([float(self.red(torch.tensor(c)).item()) for c in colors])
        g_values = np.array([float(self.green(torch.tensor(c)).item()) for c in colors])
        b_values = np.array([float(self.blue(torch.tensor(c)).item()) for c in colors])
        
        # Plot the bar chart
        ax2.bar(x - width, r_values, width, label='Red', color='red')
        ax2.bar(x, g_values, width, label='Green', color='green')
        ax2.bar(x + width, b_values, width, label='Blue', color='blue')
        
        # Set the second subplot properties
        ax2.set_title('RGB Values by Index')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # Adjust layout and save if needed
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

color_executor = ColorDomain(color_domain)



