import torch
import torch.nn as nn
from typing import Dict
from spatial.energy_graph import TimeInputEnergyMLP, ModelMixin

class DistanceEnergyModel(nn.Module, ModelMixin):
    """Energy-based model for distance relations"""
    def __init__(self, constraints: Dict[str, int], dim: int = 2):
        super().__init__()
        self.energies = nn.ModuleDict({})
        for name in constraints:
            arity = constraints[name]  # Should be 2 for distance relations
            self.energies[name] = TimeInputEnergyMLP(arity * dim)
        self.input_dims = (dim,)
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: Dict = None) -> Dict:
        """Forward pass computing energy and gradient
        
        Args:
            x: States tensor containing point coordinates
            sigma: Noise level tensor
            cond: Dictionary with edges specifying relations
        """
        assert cond is not None, "cond requires edge parameters"
        total_energy = 0.0
        x.requires_grad = True

        for edge in cond["edges"]:
            obj_idx = edge[:-1]
            type_name = edge[-1]
            
            # Get inputs for this relation
            x_inputs = x[obj_idx, :].reshape(1, -1)
            
            # Handle sigma based on its dimension
            if sigma.dim() == 0:
                sigma_inputs = sigma.repeat(len(obj_idx))
            else:
                sigma_inputs = sigma[obj_idx[0]]
            
            # Calculate energy contribution
            comp = self.energies[type_name](x_inputs.float(), sigma_inputs.float())
            total_energy = total_energy + comp["energy"]

        # Calculate gradient
        grad = torch.autograd.grad(
            total_energy.flatten().sum(), x,
            retain_graph=True, create_graph=True
        )[0]
        
        return {"energy": total_energy, "gradient": grad}

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from spatial.diffusion import training_loop, samples, ScheduleLogLinear

from domains.distance.distance_data import DistanceDataset, collate_distance_batch

from domains.distance.distance_data import DistanceDataset
import matplotlib.pyplot as plt

def visualize_examples():
    # Create dataset with a few samples
    dataset = DistanceDataset(
        num_samples=5,
        min_objects=3,
        max_objects=5,
        space_bounds=10.0,
        temperature=0.1
    )
    
    # Show each example
    for i in range(len(dataset)):
        state, constraints = dataset[i]
        
        print(f"\nExample {i+1}:")
        print("State shape:", state.shape)
        print("Number of points:", len(state))
        print("Relations:")
        for c in constraints:
            print(f"  Point {c.obj_i} -> Point {c.obj_j}: {c.relation}")
        
        # Visualize the configuration
        dataset.visualize_configuration(state, constraints)
        plt.show()
        
        # Also show each relation type separately
        relation_types = set(c.relation for c in constraints)
        for relation in relation_types:
            print(f"\nVisualizating relation: {relation}")
        dataset.visualize_configuration(state, constraints, relation)
        plt.show()


def main():
    # Define distance relations to model
    constraints = {
        "very_near": 2,
        "near": 2,
        "moderately_far": 2,
        "far": 2,
        "very_far": 2
    }

    # Create model and dataset
    model = DistanceEnergyModel(constraints, dim=2)  # 2D points
    dataset = DistanceDataset(
        num_samples=1000,
        min_objects=3,
        max_objects=6,
        space_bounds=10.0,
        temperature=0.1
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_distance_batch
    )

    # Setup schedule
    schedule = ScheduleLogLinear(N=500, sigma_min=0.01, sigma_max=10)

    # Training loop
    for stats in training_loop(
        dataloader,
        model,
        schedule,
        epochs=0,
        conditional=True
    ):
        if 1: pass

    # Save trained model
    model.load_state_dict(torch.load("distance_model_state.pth"))
    #torch.save(model.state_dict(), "distance_model_state.pth")


    # Generate samples with specific relations
    cond = {
        "edges": [
            (0, 1, "near"),
            (1, 2, "far"),
            (2, 3, "very_near")
        ]
    }

    # Sample and visualize
    sigmas = schedule.sample_sigmas(30)
    for i, xt in enumerate(samples(model, sigmas, gam=2, cond=cond, batchsize=4)):
        print(f"Step {i+1}")
        dataset.visualize_configuration(xt, cond["edges"])
        plt.pause(0.01)

if __name__ == "__main__":
    main()