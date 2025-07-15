import torch
import torch.nn as nn
import math
from helchriss.domain import load_domain_string
from helchriss.knowledge.executor import CentralExecutor
from domains.utils import BatchVisualizer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.cm as cm

cycle_domain_str = """
(domain Cycle)
(:type
    theta - vector[float,1]
    pos - vector[float,2]
)
(:predicate
    pos ?x-theta -> vector[float,2] 
    pos_to_theta ?x-pos -> theta

    clockwise ?x-pos ?y-pos -> boolean
    counterclockwise ?x-pos ?y-pos -> boolean
    opposite ?x-pos ?y-pos -> boolean
)
"""

cycle_domain = load_domain_string(cycle_domain_str)

class CycleExecutor(CentralExecutor, BatchVisualizer):
    """Executor for operations on a unit circle, handling angular positions
    and relationships between positions."""
    
    def __init__(self, domain, temperature=7.58, color_style='viridis'):
        super().__init__(domain)
        self.temperature = temperature
        self.color_style = color_style  # New parameter for color style
    
    def pos(self, theta):
        """Convert angle to position on unit circle."""
        theta_normalized = theta % (2 * math.pi)
        x = torch.cos(theta_normalized)
        y = torch.sin(theta_normalized)
        return torch.stack([x, y], dim=-1)
    
    def pos_to_theta(self, position):
        """Convert position on unit circle to angle."""
        x, y = position[..., 0], position[..., 1]
        theta = torch.atan2(y, x)
        theta = theta % (2 * math.pi)
        return theta.unsqueeze(-1)
    
    def _angle_between(self, pos1, pos2):
        """Calculate the smallest angle between two positions."""
        theta1 = self.pos_to_theta(pos1)
        theta2 = self.pos_to_theta(pos2)
        diff = (theta2 - theta1) % (2 * math.pi)
        smaller_angle = torch.min(diff, 2 * math.pi - diff)
        return smaller_angle
    
    def _is_clockwise(self, pos1, pos2):
        """Determine if moving from pos1 to pos2 is clockwise."""
        theta1 = self.pos_to_theta(pos1)
        theta2 = self.pos_to_theta(pos2)
        diff = (theta1 - theta2) % (2 * math.pi)
        return diff < math.pi
    
    def clockwise(self, pos1, pos2):
        """Determine if pos1 to pos2 is clockwise and output logits."""
        theta1 = self.pos_to_theta(pos1)
        theta2 = self.pos_to_theta(pos2)
        diff = (theta1 - theta2) % (2 * math.pi)
        clockwise_score = self.temperature * (math.pi - diff)
        return clockwise_score
    
    def counterclockwise(self, pos1, pos2):
        """Determine if pos1 to pos2 is counterclockwise and output logits."""
        theta1 = self.pos_to_theta(pos1)
        theta2 = self.pos_to_theta(pos2)
        diff = (theta2 - theta1) % (2 * math.pi)
        counterclockwise_score = self.temperature * (math.pi - diff)
        return counterclockwise_score
    
    def opposite(self, pos1, pos2):
        """Determine if pos1 and pos2 are on opposite sides of the circle."""
        theta1 = self.pos_to_theta(pos1)
        theta2 = self.pos_to_theta(pos2)
        diff = torch.abs((theta1 - theta2) % (2 * math.pi))
        opposite_score = -self.temperature * torch.abs(diff - math.pi) + 5
        return opposite_score
    
    def visualize(self, batched_data, save_path=None, map_to_circle=True):
        """Visualize the cycle positions and relationships with custom color style.
        
        Args:
            batched_data: Batched theta values or positions to visualize
            save_path: Optional path to save the visualization
            map_to_circle: Whether to map points to the unit circle with connecting lines
        """
        # Convert batched data to positions on unit circle
        if isinstance(batched_data, torch.Tensor):
            if batched_data.shape[-1] == 2:
                positions = batched_data.detach().cpu().numpy()
            else:
                thetas = batched_data.detach().cpu()
                positions = self.pos(thetas).numpy()
        else:
            thetas = torch.tensor(batched_data)
            positions = self.pos(thetas).numpy()
        
        # Reshape to ensure we have a batch of positions
        if len(positions.shape) == 1:
            positions = positions.reshape(1, -1)
        
        # Extract x and y coordinates
        n = len(positions)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Create figure with subplots - add a third subplot if mapping to circle
        if map_to_circle:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First subplot: Unit circle with points
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='-', alpha=0.3)
        ax1.add_artist(circle)
        
        # Choose colormap for points based on color_style
        point_cmap = plt.get_cmap(self.color_style)
        
        # Plot each point on the unit circle with custom colors
        for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
            point_color = point_cmap(idx / max(1, n-1))
            ax1.scatter(x, y, color=point_color, s=100, edgecolor='black')
            ax1.text(x * 1.1, y * 1.1, str(idx), fontsize=10)
        
        # Set the first subplot properties
        ax1.set_title('Positions on Unit Circle')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Second subplot: Relationship matrix
        clockwise_matrix = np.zeros((n, n))
        counterclockwise_matrix = np.zeros((n, n))
        opposite_matrix = np.zeros((n, n))
        
        # Compute relationship values for each pair
        for i in range(n):
            for j in range(n):
                pos1 = torch.tensor(positions[i])
                pos2 = torch.tensor(positions[j])
                
                clockwise_val = torch.sigmoid(self.clockwise(pos1, pos2)).item()
                counterclockwise_val = torch.sigmoid(self.counterclockwise(pos1, pos2)).item()
                opposite_val = torch.sigmoid(self.opposite(pos1, pos2)).item()
                
                # Store in matrices
                clockwise_matrix[i, j] = clockwise_val
                counterclockwise_matrix[i, j] = counterclockwise_val
                opposite_matrix[i, j] = opposite_val
        
        # Set up custom relationship matrix coloring based on color_style
        if self.color_style == 'pastel':
            # Pastel color scheme
            relationship_matrix = np.zeros((n, n, 3))
            relationship_matrix[:, :, 0] = 0.7 + 0.3 * clockwise_matrix  # Pastel red for clockwise
            relationship_matrix[:, :, 1] = 0.7 + 0.3 * opposite_matrix   # Pastel green for opposite
            relationship_matrix[:, :, 2] = 0.7 + 0.3 * counterclockwise_matrix  # Pastel blue for counterclockwise
            legend_colors = ['lightcoral', 'lightgreen', 'lightblue']
            
        elif self.color_style == 'dark':
            # Dark color scheme
            relationship_matrix = np.zeros((n, n, 3))
            relationship_matrix[:, :, 0] = 0.3 * clockwise_matrix  # Dark red for clockwise
            relationship_matrix[:, :, 1] = 0.3 * opposite_matrix   # Dark green for opposite
            relationship_matrix[:, :, 2] = 0.3 * counterclockwise_matrix  # Dark blue for counterclockwise
            legend_colors = ['darkred', 'darkgreen', 'darkblue']
            
        elif self.color_style == 'monochrome':
            # Monochrome scheme with different shades for different relationships
            relationship_matrix = np.zeros((n, n, 3))
            
            # Use different grayscale intensities for different relationships
            for i in range(n):
                for j in range(n):
                    if i == j:  # Diagonal elements 
                        value = 0.5  # Mid-gray for diagonal
                    else:
                        # Determine which relationship is strongest
                        rel_values = [clockwise_matrix[i, j], 
                                    opposite_matrix[i, j], 
                                    counterclockwise_matrix[i, j]]
                        max_index = np.argmax(rel_values)
                        max_value = rel_values[max_index]
                        
                        # Assign different grayscale ranges based on relationship type
                        if max_index == 0:  # Clockwise - darker
                            value = 0.2 + (0.2 * max_value)
                        elif max_index == 1:  # Opposite - medium
                            value = 0.4 + (0.2 * max_value)
                        else:  # Counterclockwise - lighter
                            value = 0.6 + (0.3 * max_value)
                    
                    # Set all channels to same value for grayscale
                    relationship_matrix[i, j, 0] = value
                    relationship_matrix[i, j, 1] = value
                    relationship_matrix[i, j, 2] = value
            
            # Set distinct grayscale values for legend
            legend_colors = [0.3, 0.5, 0.8]
            
        elif self.color_style == 'warm':
            # Warm colors scheme
            relationship_matrix = np.zeros((n, n, 3))
            relationship_matrix[:, :, 0] = clockwise_matrix  # Red
            relationship_matrix[:, :, 1] = 0.5 * opposite_matrix  # Reduce green to make warmer
            relationship_matrix[:, :, 2] = 0.2 * counterclockwise_matrix  # Minimal blue
            legend_colors = ['red', 'orange', 'yellow']
            
        elif self.color_style == 'cool':
            # Cool colors scheme
            relationship_matrix = np.zeros((n, n, 3))
            relationship_matrix[:, :, 0] = 0.2 * clockwise_matrix  # Minimal red
            relationship_matrix[:, :, 1] = 0.5 * opposite_matrix  # Some green
            relationship_matrix[:, :, 2] = counterclockwise_matrix  # Full blue
            legend_colors = ['purple', 'teal', 'blue']
            
        elif self.color_style == 'viridis' or self.color_style == 'plasma' or self.color_style == 'inferno':
            # Use 3 distinct colors from the colormap for the different relationships
            cmap = plt.get_cmap(self.color_style)
            
            # Create a color-coded matrix where each relationship has a distinct color range
            relationship_matrix = np.zeros((n, n, 3))
            
            # Create separate colormaps for each relationship type
            clockwise_colors = cmap(np.linspace(0, 0.3, 100))[:, :3]  # Lower range of colormap
            opposite_colors = cmap(np.linspace(0.35, 0.65, 100))[:, :3]  # Middle range
            counterclockwise_colors = cmap(np.linspace(0.7, 1.0, 100))[:, :3]  # Upper range
            
            # For each cell, determine which relationship is strongest
            for i in range(n):
                for j in range(n):
                    if i == j:  # Diagonal elements
                        relationship_matrix[i, j] = cmap(0.5)[:3]  # Neutral color
                    else:
                        # Find the dominant relationship
                        rel_values = [clockwise_matrix[i, j], 
                                    opposite_matrix[i, j], 
                                    counterclockwise_matrix[i, j]]
                        max_index = np.argmax(rel_values)
                        max_value = rel_values[max_index]
                        
                        # Scale the value to an index in the appropriate color range
                        color_idx = min(int(max_value * 99), 99)  # Scale to 0-99 index
                        
                        if max_index == 0:  # Clockwise is dominant
                            relationship_matrix[i, j] = clockwise_colors[color_idx]
                        elif max_index == 1:  # Opposite is dominant
                            relationship_matrix[i, j] = opposite_colors[color_idx]
                        else:  # Counterclockwise is dominant
                            relationship_matrix[i, j] = counterclockwise_colors[color_idx]
            
            # Set legend colors to the middle intensity of each range
            legend_colors = [
                clockwise_colors[50],     # Middle of clockwise range
                opposite_colors[50],      # Middle of opposite range
                counterclockwise_colors[50]  # Middle of counterclockwise range
            ]
            legend_labels = ['Clockwise', 'Opposite', 'Counterclockwise']
        else:
            # Default RGB scheme from original code
            relationship_matrix = np.zeros((n, n, 3))
            relationship_matrix[:, :, 0] = clockwise_matrix  # Red channel: clockwise
            relationship_matrix[:, :, 1] = opposite_matrix   # Green channel: opposite
            relationship_matrix[:, :, 2] = counterclockwise_matrix  # Blue channel: counterclockwise
            legend_colors = ['red', 'green', 'blue']
            legend_labels = ['Clockwise', 'Opposite', 'Counterclockwise']
        
        # Plot the relationship matrix
        im = ax2.imshow(relationship_matrix)
        
        # Set the second subplot properties
        ax2.set_title(f'Relationship Matrix ({self.color_style} color style)')
        
        ax2.set_xlabel('Element j')
        ax2.set_ylabel('Element i')
        ax2.set_xticks(np.arange(n))
        ax2.set_yticks(np.arange(n))
        
        # Add a custom legend for the relationship colors
        legend_elements = [
            Patch(facecolor=legend_colors[0], edgecolor='w', label='Clockwise'),
            Patch(facecolor=legend_colors[1], edgecolor='w', label='Opposite'),
            Patch(facecolor=legend_colors[2], edgecolor='w', label='Counterclockwise')
        ]
        
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout and save if needed
        plt.tight_layout()
        
        # Third subplot: Map points to their positions on the unit circle
        if map_to_circle:
            # Create unit circle
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='-', alpha=0.3)
            ax3.add_artist(circle)
            
            # Original data points (could be anywhere, not just on the circle)
            # For this example, let's create some arbitrary points near but not on the circle
            if isinstance(batched_data, torch.Tensor):
                if batched_data.shape[-1] == 1:  # If they are angles
                    # Create some offset points for demonstration
                    original_positions = self.pos(batched_data).detach().cpu().numpy()
                    # Add some arbitrary offset to move points off the circle
                    offset_positions = original_positions * (0.5 + 0.3 * np.random.rand(len(original_positions), 1))
                    positions_on_circle = original_positions
                elif batched_data.shape[-1] == 2:  # If they are already positions
                    positions = batched_data.detach().cpu().numpy()
                    # Calculate distances from origin
                    distances = np.sqrt(np.sum(positions**2, axis=1))
                    # If points are already on the circle (or very close), add some offsets
                    if np.all(np.abs(distances - 1.0) < 0.1):
                        offset_positions = positions * (0.5 + 0.3 * np.random.rand(len(positions), 1))
                        positions_on_circle = positions
                    else:
                        # Points are not on circle, so project them
                        offset_positions = positions
                        # Normalize to get positions on circle
                        distances = np.sqrt(np.sum(positions**2, axis=1))
                        positions_on_circle = positions / distances[:, np.newaxis]
            else:
                # Handle non-tensor data
                if np.array(batched_data).ndim == 1:  # Single batch
                    thetas = torch.tensor([batched_data])
                else:
                    thetas = torch.tensor(batched_data)
                
                original_positions = self.pos(thetas).detach().cpu().numpy()
                offset_positions = original_positions * (0.5 + 0.3 * np.random.rand(len(original_positions), 1))
                positions_on_circle = original_positions
            
            # Draw the original positions
            point_cmap = plt.get_cmap(self.color_style)
            for idx, pos in enumerate(offset_positions):
                point_color = point_cmap(idx / max(1, len(offset_positions)-1))
                ax3.scatter(pos[0], pos[1], color=point_color, s=100, edgecolor='black')
                ax3.text(pos[0] * 1.1, pos[1] * 1.1, str(idx), fontsize=10)
            
            # Draw the corresponding positions on the unit circle
            for idx, (off_pos, circ_pos) in enumerate(zip(offset_positions, positions_on_circle)):
                point_color = point_cmap(idx / max(1, len(offset_positions)-1))
                # Draw the position on the circle
                ax3.scatter(circ_pos[0], circ_pos[1], color=point_color, s=50, edgecolor='black', marker='x')
                
                # Draw a line connecting the original position to its circle position
                ax3.plot([off_pos[0], circ_pos[0]], [off_pos[1], circ_pos[1]], 
                        color=point_color, linestyle='--', alpha=0.7)
                
                # Add a small angle marker
                theta = np.arctan2(circ_pos[1], circ_pos[0])
                r_marker = 0.2  # Radius for the angle marker
                ax3.plot([0, r_marker * np.cos(theta)], [0, r_marker * np.sin(theta)], 
                        color=point_color, alpha=0.7)
            
            # Set the third subplot properties
            ax3.set_title('Mapping to Unit Circle')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_ylim(-1.5, 1.5)
            ax3.grid(True)
            ax3.set_aspect('equal')
            
            # Add a legend
            ax3.legend(['Point positions', 'Circle projections', 'Connecting lines'],
                      loc='upper right')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


cycle_executor = CycleExecutor(cycle_domain, color_style='viridis')