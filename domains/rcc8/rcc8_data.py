from .rcc8_domain import *
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

@dataclass
class SpatialConstraint:
    """Represents a spatial constraint between two objects"""
    obj_i: int
    obj_j: int
    relation: str

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
        min_objects=4,  # Using even number for better visualization
        max_objects=6,
        min_radius=0.3,
        max_radius=1.0,
        space_bounds=5.0,
        temperature=0.1
    )
    
    # Visualize a few examples
    for i in range(3):
        state, constraints = dataset[i]
        print(f"\nExample {i+1}:")
        print("State shape:", state.shape)
        print("Number of constraints:", len(constraints))
        for c in constraints:
            print(f"  {c.obj_i} -> {c.obj_j}: {c.relation}")
            # Visualize each relation type found
        dataset.visualize_configuration(state, constraints, c.relation)
    
    # Create dataloader for batched processing
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_rcc8_batch
    )