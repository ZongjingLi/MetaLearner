
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from domains.utils import BatchVisualizer

order_domain_str = """
(domain Order)
(:type
    order - vector[float, 1]
    boolean - vector[float, 1]
)
(:predicate
    greater ?x-order ?y-order -> boolean
    lesser ?x-order ?y-order -> boolean
    equal ?x-order ?y-order -> boolean
    inf -> order
    sup -> order
)
"""

order_domain = load_domain_string(order_domain_str)


class OrderExecutor(CentralExecutor, BatchVisualizer):
    """batched o1 and o2 and output the logit of greater"""
    def __init__(self, domain, temperature=(1/0.132)):
        super().__init__(domain)
        self.temperature = temperature
    
    def inf(self): return torch.tensor(1.0)

    def sup(self): return torch.tensor(-1.0)
    
    def greater(self, o1, o2):
        # Output logits directly without sigmoid for greater
        return self.temperature * (o1 - o2)
    
    def lesser(self, o1, o2):
        # Output logits directly without sigmoid for lesser
        return self.temperature * (o2 - o1)
    
    def equal(self, o1, o2):
        # Output logits directly without sigmoid for equal
        # Negative because smaller difference → more equal
        return -self.temperature * (torch.abs(o1 - o2) - 0.2)
    
    def visualize(self, batched_data, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert batched data to numpy for visualization
        if isinstance(batched_data, torch.Tensor):
            orders = batched_data.detach().cpu().numpy().flatten()
        else:
            orders = np.array(batched_data).flatten()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First subplot: Number line with points
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Draw the number line
        
        # Plot each point on the number line
        for idx, order_val in enumerate(orders):
            # Plot the point with its index
            ax1.scatter(order_val, 0, color='blue', s=100, edgecolor='black')
            ax1.text(order_val, 0.05, str(idx), fontsize=10)
        
        # Set the first subplot properties
        ax1.set_title('Order Values on Number Line')
        ax1.set_xlabel('Value')
        ax1.set_ylim(-0.5, 0.5)
        
        # Second subplot: Relationship matrix
        n = len(orders)
        greater_matrix = np.zeros((n, n))
        lesser_matrix = np.zeros((n, n))
        equal_matrix = np.zeros((n, n))
        
        # Compute relationship values for each pair
        for i in range(n):
            for j in range(n):
                o1 = torch.tensor(orders[i])
                o2 = torch.tensor(orders[j])
                greater_val = torch.sigmoid(self.greater(o1, o2)).item()
                lesser_val = torch.sigmoid(self.lesser(o1, o2)).item()
                equal_val = torch.sigmoid(self.equal(o1, o2)).item()
                
                # Store in matrices
                greater_matrix[i, j] = greater_val
                lesser_matrix[i, j] = lesser_val
                equal_matrix[i, j] = equal_val
        
        # Create a combined relationship matrix
        # RGB channels: Red=greater, Green=equal, Blue=lesser
        relationship_matrix = np.zeros((n, n, 3))
        relationship_matrix[:, :, 0] = greater_matrix  # Red channel: greater
        relationship_matrix[:, :, 1] = equal_matrix    # Green channel: equal
        relationship_matrix[:, :, 2] = lesser_matrix   # Blue channel: lesser
        
        # Plot the relationship matrix
        im = ax2.imshow(relationship_matrix)
        
        # Set the second subplot properties
        ax2.set_title('Relationship Matrix (R=greater, G=equal, B=lesser)')
        ax2.set_xlabel('Element j')
        ax2.set_ylabel('Element i')
        ax2.set_xticks(np.arange(n))
        ax2.set_yticks(np.arange(n))
        
        # Add a custom legend for the relationship colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='w', label='Greater'),
            Patch(facecolor='green', edgecolor='w', label='Equal'),
            Patch(facecolor='blue', edgecolor='w', label='Lesser')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout and save if needed
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

order_executor = OrderExecutor(order_domain)