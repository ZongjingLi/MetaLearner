from dataclasses import dataclass
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .direction_domain import DirectionalDomain, build_direction_executor

@dataclass
class DirectionalConstraint:
    """Represents a directional constraint between two objects"""
    obj_i: int
    obj_j: int
    relation: str

class DirectionDataset(Dataset):
    """Dataset for directional spatial configurations with calculated relations"""
    def __init__(self,
                 num_samples: int = 1000,
                 min_objects: int = 2,
                 max_objects: int = 5,
                 space_bounds: float = 10.0,
                 temperature: float = 0.1):
        """
        Args:
            num_samples: Number of configurations to generate
            min_objects: Minimum number of objects per configuration
            max_objects: Maximum number of objects per configuration
            space_bounds: Bounds of the space ([-space_bounds, space_bounds])
            temperature: Temperature for direction calculations
        """
        self.num_samples = num_samples
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.space_bounds = space_bounds
        
        # Initialize Direction domain for relation calculations
        self.direction_domain = DirectionalDomain(temperature=temperature)
        
        # List of all directional relation calculation functions
        self.relation_functions = [
            (self.direction_domain.north, "north"),
            (self.direction_domain.south, "south"),
            (self.direction_domain.east, "east"),
            (self.direction_domain.west, "west"),
            (self.direction_domain.northeast, "northeast"),
            (self.direction_domain.northwest, "northwest"),
            (self.direction_domain.southeast, "southeast"),
            (self.direction_domain.southwest, "southwest")
        ]
        
        # Generate dataset
        self.data = self._generate_dataset()

    def _determine_relations(self, state: torch.Tensor):
        """Determine actual directional relations between objects"""
        n_objects = state.shape[0]
        constraints = []
        
        # Compare each pair of objects
        for i in range(n_objects):
            for j in range(n_objects):
                if i != j:  # Don't compare object with itself
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
        """Generate a pair of points with a specific directional relation"""
        x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
        
        # Generate second point based on relation
        angle_map = {
            "north": np.pi/2,
            "south": -np.pi/2,
            "east": 0,
            "west": np.pi,
            "northeast": np.pi/4,
            "northwest": 3*np.pi/4,
            "southeast": -np.pi/4,
            "southwest": -3*np.pi/4
        }
        
        base_angle = angle_map[relation]
        # Add some random variation while maintaining the general direction
        angle = base_angle + np.random.uniform(-np.pi/8, np.pi/8)
        distance = np.random.uniform(1, self.space_bounds/4)
        
        x2 = x1 + distance * np.cos(angle)
        y2 = y1 + distance * np.sin(angle)
        
        return (torch.tensor([x1, y1]), torch.tensor([x2, y2]))

    def _generate_configuration(self) -> Tuple[torch.Tensor, List[DirectionalConstraint]]:
        """Generate a single spatial configuration with diverse relations"""
        # Choose number of objects
        n_objects = np.random.randint(self.min_objects, self.max_objects + 1)
        
        # Generate initial points
        points = []
        for _ in range(n_objects):
            x = np.random.uniform(-self.space_bounds/2, self.space_bounds/2)
            y = np.random.uniform(-self.space_bounds/2, self.space_bounds/2)
            points.append(torch.tensor([x, y]))
        
        # Combine all points into state tensor
        state = torch.stack(points)
        
        # Determine all relations between objects
        constraints = self._determine_relations(state)
        
        return state, constraints
    
    def _generate_dataset(self) -> List[Tuple[torch.Tensor, List[DirectionalConstraint]]]:
        """Generate the complete dataset"""
        return [self._generate_configuration() for _ in range(self.num_samples)]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        data = self.data[idx]
        return {"data": data[0].float(), "cond": {"edges": data[1]}}
    
    def visualize_configuration(self, state: torch.Tensor, constraints: List[DirectionalConstraint],
                              relation_name: str = None) -> None:
        """Visualize a single spatial configuration"""
        # Split objects into two groups for visualization
        n_objects = state.shape[0]
        mid_point = n_objects // 2
        
        # Prepare states dictionary for DirectionalDomain visualizer
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
        
        # Use DirectionalDomain's visualizer
        fig = self.direction_domain.visualize(states_dict, relation_matrix)
        plt.show()
        return fig

def process_batch_for_training(batch):
    """Convert batch data to format expected by training loop"""
    states, constraints = batch
    
    # Create edges list from constraints
    edges = []
    for batch_idx, sample_constraints in enumerate(constraints):
        for c in sample_constraints:
            edges.append((c.obj_i, c.obj_j, c.relation))
    
    return {
        "data": states,
        "cond": {"edges": edges}
    }

