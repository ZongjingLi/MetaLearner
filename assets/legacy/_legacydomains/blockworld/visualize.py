'''
 # @ Author: Your Name
 # @ Create Time: 2024-11-14
 # @ Description: Visualization utilities for BlockWorld domain
'''
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BlockWorldPredicates:
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.table_height = 0.0
        self.block_height = 1.0
        self.block_width = 1.0
    
    def block_position(self, x_state: torch.Tensor) -> torch.Tensor:
        return x_state
    
    def on_table(self, x_state: torch.Tensor) -> torch.Tensor:
        heights = x_state[:, 1]
        # Block should be exactly block_height/2 above table
        return torch.sigmoid(-(torch.abs(heights - (self.table_height + self.block_height/2)) - 0.1) / self.temperature)
    
    def on(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        heights1 = x_state[:, 1].unsqueeze(1)
        heights2 = y_state[:, 1]
        positions1 = x_state[:, 0].unsqueeze(1)
        positions2 = y_state[:, 0]
        
        # Block should be exactly block_height above other block
        height_diff = heights1 - heights2
        position_diff = torch.abs(positions1 - positions2)
        
        correct_height = torch.sigmoid(-(torch.abs(height_diff - self.block_height) - 0.1) / self.temperature)
        aligned = torch.sigmoid(-(position_diff - 0.1) / self.temperature)
        
        return correct_height * aligned
    
    def clear(self, x_state: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        all_states = torch.cat([v["state"] for k, v in context.items() if isinstance(k, (int, str)) and k != "free"])
        
        height_diffs = all_states[:, 1].unsqueeze(1) - x_state[:, 1]
        position_diffs = torch.abs(all_states[:, 0].unsqueeze(1) - x_state[:, 0])
        
        is_above = torch.sigmoid((height_diffs - self.block_height/2) / self.temperature)
        is_aligned = torch.sigmoid(-(position_diffs - self.block_width/2) / self.temperature)
        
        has_block_above = torch.max(is_above * is_aligned, dim=0)[0]
        return 1 - has_block_above
    
    def above(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        heights1 = x_state[:, 1].unsqueeze(1)
        heights2 = y_state[:, 1]
        positions1 = x_state[:, 0].unsqueeze(1)
        positions2 = y_state[:, 0]
        
        is_higher = torch.sigmoid((heights1 - heights2) / self.temperature)
        aligned = torch.sigmoid(-(torch.abs(positions1 - positions2) - self.block_width/2) / self.temperature)
        
        return is_higher * aligned
    
    def holding(self, x_state: torch.Tensor) -> torch.Tensor:
        # Assuming holding means block is at a specific height
        holding_height = 3.0
        heights = x_state[:, 1]
        return torch.sigmoid(-(torch.abs(heights - holding_height) - 0.1) / self.temperature)
    
    def free(self, context: Dict[str, Any]) -> torch.Tensor:
        return context.get("free", torch.ones(1))
    
    def exists(self, x_state: torch.Tensor) -> torch.Tensor:
        return torch.ones(len(x_state))

    def visualize(self, states_dict, relation_matrix=None, program=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1]})
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Draw table
        table_width = 6
        table_height = 0.2
        table = patches.Rectangle((-table_width/2, self.table_height - table_height/2), 
                                table_width, table_height, color='brown')
        ax1.add_patch(table)
        
        # Plot blocks
        for i, (key, value) in enumerate(states_dict.items()):
            if isinstance(key, (int, str)) and key != "free":
                state = value["state"]
                color = colors[i % len(colors)]
                
                for j, pos in enumerate(state):
                    block = patches.Rectangle((pos[0].item() - self.block_width/2, 
                                            pos[1].item() - self.block_height/2),
                                           self.block_width, self.block_height, 
                                           color=color, alpha=0.7)
                    ax1.add_patch(block)
                    ax1.text(pos[0].item(), pos[1].item(), f'{key}_{j}',
                           ha='center', va='center')
        
        if relation_matrix is not None:
            im = ax2.imshow(relation_matrix.numpy(), cmap='viridis',
                          aspect='equal', interpolation='nearest')
            plt.colorbar(im, ax=ax2)
            ax2.set_title("Relation Matrix")
        
        ax1.grid(True)
        ax1.set_aspect('equal')
        ax1.set_ylim(-1, 4)
        ax1.set_xlim(-3, 3)
        ax1.set_title("Block Configuration")
        
        if program is not None:
            fig.suptitle(f"Program: {program}")
        
        plt.tight_layout()
        return fig

def test_predicates():
    predicates = BlockWorldPredicates()
    
    # Create test states
    block1 = torch.tensor([[0.0, 0.5]])  # On table
    block2 = torch.tensor([[0.0, 1.5]])  # On block1
    block3 = torch.tensor([[2.0, 0.5]])  # On table, separate
    block4 = torch.tensor([[0.0, 3.0]])  # Being held
    
    # Test on_table
    print("Testing on_table:")
    print(f"Block1: {predicates.on_table(block1).item():.3f}")
    print(f"Block2: {predicates.on_table(block2).item():.3f}")
    print(f"Block3: {predicates.on_table(block3).item():.3f}")
    
    # Test on
    print("\nTesting on:")
    print(f"Block2 on Block1: {predicates.on(block2, block1).item():.3f}")
    print(f"Block3 on Block1: {predicates.on(block3, block1).item():.3f}")
    
    # Test clear
    context = {
        1: {"state": block1},
        2: {"state": block2},
        3: {"state": block3}
    }
    print("\nTesting clear:")
    print(f"Block1 clear: {predicates.clear(block1, context).item():.3f}")
    print(f"Block2 clear: {predicates.clear(block2, context).item():.3f}")
    print(f"Block3 clear: {predicates.clear(block3, context).item():.3f}")
    
    # Test above
    print("\nTesting above:")
    print(f"Block2 above Block1: {predicates.above(block2, block1).item():.3f}")
    print(f"Block3 above Block1: {predicates.above(block3, block1).item():.3f}")
    
    # Test holding
    print("\nTesting holding:")
    print(f"Block1 holding: {predicates.holding(block1).item():.3f}")
    print(f"Block4 holding: {predicates.holding(block4).item():.3f}")
    
    # Visualize test configuration
    test_states = {
        1: {"state": block1},
        2: {"state": block2},
        3: {"state": block3},
        4: {"state": block4}
    }
    predicates.visualize(test_states)
    plt.show()

if __name__ == "__main__":
    test_predicates()