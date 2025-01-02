import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from domains.utils import smooth_and, smooth_or, gaussian_kernel, build_domain_executor

class BlockWorldPredicates:
    def __init__(self, temperature=0.1, epsilon=1e-6):
        self.temperature = temperature
        self.epsilon = epsilon
        self.table_height = 0.0
        self.block_height = 1.0
        
    def get_position(self, state):
        """Return block position"""
        return state
    
    def clear(self, state, context):
        """Check if no block is on top of this block"""
        all_states = torch.cat([v["state"] for k, v in context.items()])
        heights = all_states[:, 1]
        positions = all_states[:, 0]
        
        # Check if any block is directly above
        state_heights = state[:, 1]
        state_positions = state[:, 0]
        
        height_diffs = heights.unsqueeze(1) - state_heights
        position_diffs = torch.abs(positions.unsqueeze(1) - state_positions)
        
        is_above = torch.sigmoid((height_diffs - self.block_height) / self.temperature)
        is_aligned = torch.sigmoid(-(position_diffs - 0.5) / self.temperature)
        
        has_block_above = torch.max(is_above * is_aligned, dim=0)[0]
        return 1 - has_block_above
    
    def holding(self, state, context):
        """Check if block is being held"""
        return context.get("holding", torch.zeros(len(state)))
    
    def on_table(self, state, context):
        """Check if block is on the table"""
        heights = state[:, 1]
        return torch.sigmoid(-(heights - self.table_height - 0.5) / self.temperature)
    
    def on(self, state1, state2):
        """Check if state1 is on state2"""
        heights1 = state1[:, 1].unsqueeze(1)
        heights2 = state2[:, 1]
        positions1 = state1[:, 0].unsqueeze(1)
        positions2 = state2[:, 0]
        
        height_diff = heights1 - heights2
        position_diff = torch.abs(positions1 - positions2)
        
        correct_height = torch.sigmoid((height_diff - self.block_height) / self.temperature)
        aligned = torch.sigmoid(-(position_diff - 0.5) / self.temperature)
        
        return correct_height * aligned
    
    def free(self, context):
        """Check if gripper is free"""
        return context.get("free", torch.ones(1))
    
    def exists(self, state):
        """Check if block exists"""
        return torch.ones(len(state))
    
    def above(self, state1, state2):
        """Check if state1 is above state2 (not necessarily directly)"""
        heights1 = state1[:, 1].unsqueeze(1)
        heights2 = state2[:, 1]
        positions1 = state1[:, 0].unsqueeze(1)
        positions2 = state2[:, 0]
        
        is_higher = torch.sigmoid((heights1 - heights2) / self.temperature)
        aligned = torch.sigmoid(-(torch.abs(positions1 - positions2) - 0.5) / self.temperature)
        
        return is_higher * aligned
    
    def visualize(self, states_dict, relation_matrix=None, program=None):
        """Visualize block configurations"""
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Draw table
        table_x = [-1, 4]
        ax1.plot(table_x, [self.table_height, self.table_height], 'k-', linewidth=2)
        
        # Plot blocks
        for i, (key, value) in enumerate(states_dict.items()):
            if isinstance(key, (int, str)):  # Skip non-block entries like 'free'
                state = value["state"]
                ax1.scatter(state[:, 0], state[:, 1], c=colors[i % len(colors)],
                          s=100, label=f'Block {key}')
                
                # Add block labels
                for j in range(len(state)):
                    ax1.annotate(f'{key}_{j}', (state[j, 0], state[j, 1]),
                               xytext=(5, 5), textcoords='offset points')
        
        # Show relations if provided
        if relation_matrix is not None:
            im = ax2.imshow(relation_matrix.numpy(), cmap='viridis',
                          aspect='equal', interpolation='nearest')
            plt.colorbar(im, ax=ax2)
            ax2.set_title("Relation Matrix")
        
        ax1.grid(True)
        ax1.set_aspect('equal')
        ax1.set_ylim(-0.5, 4)
        ax1.set_xlim(-1.5, 4.5)
        ax1.legend()
        ax1.set_title("Block Configuration")
        
        if program is not None:
            fig.suptitle(f"Program: {program}")
        
        plt.tight_layout()
        return fig

# Build domain executor
folder_path = os.path.dirname(__file__)
blockworld_executor = build_domain_executor(f"{folder_path}/blockworld_domain.txt")

# Initialize predicates
blockworld_predicates = BlockWorldPredicates()

# Register predicates with executor
for predicate_name in ['get_position', 'clear', 'holding', 'on_table', 
                      'on', 'free', 'exists', 'above']:
    predicate_func = getattr(blockworld_predicates, predicate_name)
    blockworld_executor.redefine_predicate(predicate_name, predicate_func)

blockworld_executor.visualize = blockworld_predicates.visualize