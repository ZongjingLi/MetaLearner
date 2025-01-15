import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from domains.distance.distance_domain import DistanceDomain

@dataclass
class SpatialConstraint:
    """Represents a spatial constraint between two objects"""
    obj_i: int
    obj_j: int
    relation: str

class DistanceDataset(Dataset):
    """Dataset for distance spatial configurations"""
    def __init__(self, 
                 num_samples: int = 1000,
                 min_objects: int = 2,
                 max_objects: int = 5,
                 space_bounds: float = 10.0,
                 temperature: float = 0.1):
        self.num_samples = num_samples
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.space_bounds = space_bounds
        
        # Initialize distance domain for relation calculations
        self.distance_domain = DistanceDomain(temperature=temperature)
        
        # List of all distance relation functions
        self.relation_functions = [
            (self.distance_domain.very_near, "very_near"),
            (self.distance_domain.near, "near"),
            (self.distance_domain.moderately_far, "moderately_far"),
            (self.distance_domain.far, "far"),
            (self.distance_domain.very_far, "very_far")
        ]
        
        # Generate dataset
        self.data = self._generate_dataset()
    
    def _determine_relations(self, state: torch.Tensor) -> List[SpatialConstraint]:
        """Determine actual distance relations between objects"""
        n_objects = state.shape[0]
        constraints = []
        
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
                threshold = 0.8
                
                if max_score > threshold:
                    for score, name in relation_scores:
                        if score > threshold:
                            constraints.append(SpatialConstraint(i, j, name))
        
        return constraints
    
    def _generate_pair_with_relation(self, relation: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a pair of points with a specific distance relation"""
        if relation == "very_near":
            # Generate very close points
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0, 0.5)
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
            
        elif relation == "near":
            # Generate nearby points
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0.5, 1.0)
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
            
        elif relation == "moderately_far":
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(1.5, 2.5)
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
            
        elif relation == "far":
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(3.5, 4.5)
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
            
        elif relation == "very_far":
            x1, y1 = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(7.5, 8.5)
            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)
        
        return (torch.tensor([x1, y1]), torch.tensor([x2, y2]))
    
    def _generate_configuration(self) -> Tuple[torch.Tensor, List[SpatialConstraint]]:
        """Generate a single spatial configuration"""
        # Choose subset of relations to include
        n_relations = np.random.randint(2, min(4, len(self.relation_functions)))
        selected_relations = np.random.choice([name for _, name in self.relation_functions], 
                                           size=n_relations, replace=False)
        
        # Generate pairs for each selected relation
        pairs = []
        for relation in selected_relations:
            pair = self._generate_pair_with_relation(relation)
            pairs.extend([pair[0], pair[1]])
        
        # Add additional random points if needed
        n_current = len(pairs)
        if n_current < self.min_objects:
            n_additional = self.min_objects - n_current
            for _ in range(n_additional):
                x, y = np.random.uniform(-self.space_bounds/2, self.space_bounds/2, 2)
                pairs.append(torch.tensor([x, y]))
        
        # Combine all points into state tensor
        state = torch.stack(pairs)
        
        # Determine actual relations between objects
        constraints = self._determine_relations(state)
        
        return state, constraints
    
    def _generate_dataset(self):
        """Generate the complete dataset"""
        return [self._generate_configuration() for _ in range(self.num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def visualize_configuration(self, state: torch.Tensor, constraints: List[SpatialConstraint],
                              relation_name: str = None):
        """Visualize using distance domain visualizer"""
        states_dict = {0: {"state": state}, 1: {"state": state}}

        # Calculate relation matrix if specific relation provided
        relation_matrix = None
        if relation_name:
            relation_func = next(func for func, name in self.relation_functions 
                               if name == relation_name)
            relation_matrix = relation_func(state, state)
        
        fig = self.distance_domain.visualize(states_dict, relation_matrix)
        plt.show()
        return fig

def collate_distance_batch(batch):
    """Collate function that concatenates states and adjusts edge indices"""
    states, constraints_list = zip(*batch)
    
    # Calculate offset for each batch item
    offsets = [0]
    for state in states[:-1]:
        offsets.append(offsets[-1] + state.shape[0])
    
    # Concatenate all states
    combined_states = torch.cat(states, dim=0)
    
    # Adjust edge indices
    adjusted_edges = []
    for batch_idx, (offset, constraints) in enumerate(zip(offsets, constraints_list)):
        for constraint in constraints:
            adjusted_edge = (
                constraint.obj_i + offset,
                constraint.obj_j + offset,
                constraint.relation
            )
            adjusted_edges.append(adjusted_edge)
    
    return {
        "data": combined_states,
        "cond": {"edges": adjusted_edges}
    }