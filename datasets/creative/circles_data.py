#from domains.spatial.circle_domain import *
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class SpatialConstraint:
    """Represents a spatial constraint between two objects"""
    obj_i: int
    obj_j: int
    relation: str


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
    
    def gaussian_kernel(self, x: torch.Tensor, temperature) -> torch.Tensor:
        """
        Compute Gaussian kernel for soft relations.
        
        This is the core method used for creating differentiable spatial relations.
        Returns high values when x is close to 0, following Gaussian distribution.
        
        Args:
            x: Input tensor (typically absolute differences or distances)
            temperature: Temperature parameter (uses self.temperature if None)
            
        Returns:
            Gaussian kernel values in range (0, 1]
        """
        if temperature is None:
            temperature = self.temperature
            
        return torch.exp(-(x ** 2) / (2 * temperature ** 2))

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
        return self.gaussian_kernel(diff, self.temperature)

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
        
        centers_equal = self.gaussian_kernel(d, self.temperature)
        radii_equal = self.gaussian_kernel(x_r - y_r, self.temperature)
        
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
        boundary_touch = self.gaussian_kernel(d - (y_r - x_r), self.temperature)
        
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
                #print(state[j, 2].item())
                circle = plt.Circle(
                    (state[j, 0].item(), state[j, 1].item()),
                    state[j, 2].item() if state[j, 2].item() > 0 else 0.0,
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
    
class RCC8Dataset(Dataset):
    """Dataset for RCC8 spatial configurations with calculated relations"""
    def __init__(self, 
                 num_samples: int = 1000,
                 min_objects: int = 2,
                 max_objects: int = 5,
                 min_radius: float = 0.2,
                 max_radius: float = 1.0,
                 space_bounds: float = 10.0,
                 temperature: float = 0.1):
        """
        Args:
            num_samples: Number of configurations to generate
            min_objects: Minimum number of objects per configuration
            max_objects: Maximum number of objects per configuration
            min_radius: Minimum radius of objects
            max_radius: Maximum radius of objects
            space_bounds: Bounds of the space ([-space_bounds, space_bounds])
            temperature: Temperature for RCC8 calculations
        """
        self.num_samples = num_samples
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.space_bounds = space_bounds
        
        # Initialize RCC8 domain for relation calculations
        self.rcc8_domain = RCC8Domain(temperature=temperature)
        
        # List of all RCC8 relation calculation functions
        self.relation_functions = [
            (self.rcc8_domain.disconnected, "disconnected"),
            (self.rcc8_domain.externally_connected, "externally_connected"),
            (self.rcc8_domain.partial_overlap, "partial_overlap"),
            (self.rcc8_domain.equal, "equal"),
            (self.rcc8_domain.tangential_proper_part, "tangential_proper_part"),
            (self.rcc8_domain.non_tangential_proper_part, "non_tangential_proper_part"),
            (self.rcc8_domain.tangential_proper_part_inverse, "tangential_proper_part_inverse"),
            (self.rcc8_domain.non_tangential_proper_part_inverse, "non_tangential_proper_part_inverse")
        ]
        
        # Generate dataset
        self.data = self._generate_dataset()
    
    def _determine_relations(self, state: torch.Tensor):
        """Determine actual RCC8 relations between objects using the domain calculations"""
        n_objects = state.shape[0]
        constraints = []
        
        # Compare each pair of objects
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                x_state = state[i:i+1]
                y_state = state[j:j+1]
                
                # Calculate all relation scores
                relation_scores = []
                for func, name in self.relation_functions:
                    score = func(x_state, y_state).item()
                    relation_scores.append((score, name))
                
                # Get the most confident relation(s)
                max_score = max(score for score, _ in relation_scores)
                threshold = 0.8  # High confidence threshold
                
                if max_score > threshold:
                    # Add relations that are close to the maximum score
                    for score, name in relation_scores:
                        if score > threshold:
                            constraints.append([i, j, name])
        
        return constraints
    
    def _generate_pair_with_relation(self, relation: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a pair of circles with a specific RCC8 relation"""
        if relation == "disconnected":
            # Generate two circles far apart
            x1, y1 = np.random.uniform(-self.space_bounds/2, -1, 2)
            r1 = np.random.uniform(self.min_radius, self.max_radius)
            x2, y2 = np.random.uniform(1, self.space_bounds/2, 2)
            r2 = np.random.uniform(self.min_radius, self.max_radius)
            
        elif relation == "externally_connected":
            # Generate two circles that touch
            r1 = np.random.uniform(self.min_radius, self.max_radius)
            r2 = np.random.uniform(self.min_radius, self.max_radius)
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = r1 + r2
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
            
        elif relation == "partial_overlap":
            # Generate two circles that overlap partially
            r1 = np.random.uniform(self.min_radius, self.max_radius)
            r2 = np.random.uniform(self.min_radius, self.max_radius)
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0.5*(r1+r2), 0.9*(r1+r2))  # Partial overlap
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
            
        elif relation == "equal":
            # Generate two identical circles
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            r1 = np.random.uniform(self.min_radius, self.max_radius)
            x2, y2 = x1, y1
            r2 = r1
            
        elif relation in ["tangential_proper_part", "tangential_proper_part_inverse"]:
            # Generate two circles where one is inside and touches the other
            if relation == "tangential_proper_part":
                r1 = np.random.uniform(self.min_radius, self.max_radius/2)  # Smaller circle
                r2 = np.random.uniform(r1*2, self.max_radius)  # Larger circle
            else:
                r2 = np.random.uniform(self.min_radius, self.max_radius/2)  # Smaller circle
                r1 = np.random.uniform(r2*2, self.max_radius)  # Larger circle
                
            x2, y2 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = abs(r2 - r1)  # Distance for touching internally
            x1 = x2 + dist * np.cos(angle)
            y1 = y2 + dist * np.sin(angle)
            
        elif relation in ["non_tangential_proper_part", "non_tangential_proper_part_inverse"]:
            # Generate two circles where one is strictly inside the other
            if relation == "non_tangential_proper_part":
                r1 = np.random.uniform(self.min_radius, self.max_radius/2)  # Smaller circle
                r2 = np.random.uniform(r1*3, self.max_radius)  # Much larger circle
            else:
                r2 = np.random.uniform(self.min_radius, self.max_radius/2)  # Smaller circle
                r1 = np.random.uniform(r2*3, self.max_radius)  # Much larger circle
                
            x2, y2 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = (abs(r2 - r1) - r1) * np.random.uniform(0.2, 0.8)  # Random position inside
            x1 = x2 + dist * np.cos(angle)
            y1 = y2 + dist * np.sin(angle)
            
        return (torch.tensor([x1, y1, r1]), torch.tensor([x2, y2, r2]))

    def _generate_configuration(self) -> Tuple[torch.Tensor, List[SpatialConstraint]]:
        """Generate a single spatial configuration with diverse relations"""
        # Choose subset of relations to include
        n_relations = np.random.randint(2, min(5, len(self.relation_functions)))
        selected_relations = np.random.choice([name for _, name in self.relation_functions], 
                                           size=n_relations, replace=False)
        
        # Generate pairs for each selected relation
        pairs = []
        for relation in selected_relations:
            pair = self._generate_pair_with_relation(relation)
            pairs.extend([pair[0], pair[1]])
        
        # Add some additional random circles if needed to meet minimum objects requirement
        n_current = len(pairs)
        if n_current < self.min_objects:
            n_additional = self.min_objects - n_current
            for _ in range(n_additional):
                x, y = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
                r = np.random.uniform(self.min_radius, self.max_radius)
                pairs.append(torch.tensor([x, y, r]))
        
        # Combine all circles into state tensor
        state = torch.stack(pairs)
        
        # Determine all relations between objects
        constraints = self._determine_relations(state)
        
        return state, constraints
    
    def _generate_dataset(self) -> List[Tuple[torch.Tensor, List[SpatialConstraint]]]:
        """Generate the complete dataset"""

        return [self._generate_configuration() for _ in range(self.num_samples)]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[SpatialConstraint]]:
        data = self.data[idx]
        #print(data[0].float())
        #print(len(data[1]))
        return {"data":data[0].float(), "cond":{"edges" : data[1]}}
    

    def visualize_configuration(self, state: torch.Tensor, constraints: List[SpatialConstraint],
                              relation_name: str = None) -> None:
        """Visualize a single spatial configuration using RCC8Domain's visualizer"""
        # Split objects into two groups for visualization
        n_objects = state.shape[0]
        mid_point = n_objects // 2
        
        # Prepare states dictionary for RCC8Domain visualizer
        states_dict = {
            0: {"state": state[:mid_point]},
            1: {"state": state[mid_point:]}
        }
        
        # Calculate relation matrix for the specific relation if provided
        relation_matrix = None
        if relation_name:
            relation_func = next(func for func, name in self.relation_functions 
                               if name == relation_name)
            relation_matrix = relation_func(states_dict[0]["state"], 
                                         states_dict[1]["state"])
        
        # Use RCC8Domain's visualizer
        fig = self.rcc8_domain.visualize(states_dict, relation_matrix)
        plt.show()
        return fig

def process_batch_for_training(batch):
    """Convert batch data to format expected by training loop"""
    states, constraints = batch
    
    # Create edges list from constraints
    edges = []
    for batch_idx, sample_constraints in enumerate(constraints):
        for c in sample_constraints:
            # Map constraint to edge format
            edges.append((c.obj_i, c.obj_j, c.relation))
    
    return {
        "data": states,
        "cond": {"edges": edges}
    }

def collate_rcc8_batch(batch: List[Tuple[torch.Tensor, List[SpatialConstraint]]]):
    """Custom collate function for RCC8 batches that concatenates states and adjusts edge indices
    
    Args:
        batch: List of (state, constraints) tuples where:
            - state is a (n_i x 3) tensor for each batch item i
            - constraints is a list of SpatialConstraint objects
            
    Returns:
        Dict with:
            - data: Combined tensor of shape (sum(n_i) x 3)
            - cond: Dict with adjusted edges list for the combined tensor
    """
    # Separate states and constraints
    states, constraints_list = zip(*batch)
    
    # Calculate offsets for adjusting indices
    cumulative_sizes = [0]  # Start with 0 offset
    for state in states[:-1]:  # Don't need offset for last state
        cumulative_sizes.append(cumulative_sizes[-1] + state.shape[0])
    
    # Concatenate all states
    combined_states = torch.cat(states, dim=0)  # Shape: (sum(n_i) x 3)
    
    # Adjust edge indices and combine all constraints
    adjusted_edges = []
    for batch_idx, (offset, constraints) in enumerate(zip(cumulative_sizes, constraints_list)):
        for constraint in constraints:
            # Create new edge with adjusted indices
            adjusted_edge = (
                constraint[0] + offset,
                constraint[1] + offset,
                constraint[2]
            )
            adjusted_edges.append(adjusted_edge)
    
    return {
        "data": combined_states,
        "cond": {"edges": adjusted_edges}
    }

# Example usage:
if __name__ == "__main__":
    # Create dataset with actual RCC8 relations
    dataset = RCC8Dataset(
        num_samples=100, 
        min_objects=2,  # Using even number for better visualization
        max_objects=6,
        min_radius=0.3,
        max_radius=1.0,
        space_bounds=5.0,
        temperature=0.1
    )
    
    # Visualize a few examples
    for i in range(3):
        data = dataset[i]
        state, constraints = data.values()
        print(f"\nExample {i+1}:")
        print("State shape:", state.shape)
        print("Number of constraints:", len(constraints))
        for c in constraints["edges"]:
            print(f"  {c[0]} -> {c[1]}: {c[-1]}")
        dataset.visualize_configuration(state, constraints, c[-1])
    
    # Create dataloader for batched processing
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_rcc8_batch
    )

def get_constraint_dataset(): return RCC8Dataset()