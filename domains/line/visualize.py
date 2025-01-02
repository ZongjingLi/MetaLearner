'''
 # @ Description: Visualization utilities for line domain
'''
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def visualize_line_predicates(states_dict: Dict[int, Dict[str, Any]], 
                            relation_matrix: Optional[torch.Tensor] = None,
                            program: Optional[str] = None):
    """
    Visualize line domain states and relations
    
    Args:
        states_dict: Dictionary of states {id: {"state": tensor, "end": value}}
        relation_matrix: Optional relation values tensor
        program: Optional program string for title
    """
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['s', 's', 's', 's', 's']  # Using squares for all points
    state_sizes = {}
    
    # Create square subplot for line visualization
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_aspect('equal')
    
    # Draw unit line segment
    ax1.axhline(y=0, xmin=0, xmax=1, color='black', linestyle='-', alpha=0.3)
    
    # Draw points
    for i, (key, value) in enumerate(states_dict.items()):
        state = value["state"].sigmoid()
        state_sizes[key] = len(state)
        
        scatter = ax1.scatter(
            state.detach().numpy(),
            torch.zeros_like(state).numpy(),
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f'State {key}',
            s=100,
            zorder=3
        )
        
        # Add labels
        for j in range(len(state)):
            ax1.annotate(f'{key}_{j}', 
                        (state[j].item(), 0),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8)
    
    # Draw endpoint markers
    ax1.scatter([0, 1], [0, 0], color='black', marker='|', s=100, zorder=2)
    ax1.text(-0.05, -0.1, '0', ha='center')
    ax1.text(1.05, -0.1, '1', ha='center')
    
    # Show relations if provided
    if relation_matrix is not None:
        if relation_matrix.dim() == 2:
            im = ax2.imshow(relation_matrix.numpy(),
                          cmap='viridis',
                          aspect='equal',
                          interpolation='nearest')
            plt.colorbar(im, ax=ax2)
            
            ax2.set_xticks(range(state_sizes[1]))
            ax2.set_yticks(range(state_sizes[0]))
            ax2.set_xticklabels([f'1_{i}' for i in range(state_sizes[1])])
            ax2.set_yticklabels([f'0_{i}' for i in range(state_sizes[0])])
            ax2.set_title("Relation Matrix")
            
        elif relation_matrix.dim() == 3:
            im = ax2.imshow(relation_matrix[:, :, 0].numpy(),
                          cmap='viridis',
                          aspect='equal',
                          interpolation='nearest')
            plt.colorbar(im, ax=ax2)
            ax2.set_title("Relation Matrix (First Slice)")
    
    ax1.set_title("Line Configuration")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if program is not None:
        fig.suptitle(f"Program: {program}")
        
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test visualization
    batch_size1, batch_size2 = 4, 3
    context = {
        0: {"state": torch.rand(batch_size1), "end": 1.0},
        1: {"state": torch.rand(batch_size2), "end": 1.0}
    }
    
    # Example relation matrix
    relation_matrix = torch.rand(batch_size1, batch_size2)
    
    # Visualize
    fig = visualize_line_predicates(
        context,
        relation_matrix,
        "Test Line Visualization"
    )
    plt.show()