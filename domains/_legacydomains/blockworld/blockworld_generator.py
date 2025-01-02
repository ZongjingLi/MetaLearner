import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects
import random
from typing import List, Tuple, Dict, Set

class BlockWorldState:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.clear = set()
        self.on = {}
        self.ontable = set()
        self.holding = None
        self.handempty = True
        
        # Expanded Death Knight color palette
        dk_colors = [
            # Frost theme
            '#00A3FF',  # Pure frost
            '#3773C2',  # Frostmourne blue
            '#1F8EFA',  # Bright frost
            '#4CB9FF',  # Ice blue
            '#0077BE',  # Deep frost
            '#89CFF0',  # Baby blue (frozen)
            '#73C2FB',  # Maya blue
            
            # Blood theme
            '#C41E3A',  # Pure blood
            '#880808',  # Dark blood
            '#AA0000',  # Blood rune
            '#DC143C',  # Crimson
            '#B22222',  # Firebrick
            '#800000',  # Maroon
            '#8B0000',  # Darkred
            
            # Unholy theme
            '#33FF00',  # Pure unholy
            '#50C878',  # Emerald
            '#228B22',  # Forest green
            '#32CD32',  # Lime green
            '#66FF00',  # Bright plague
            '#008000',  # Dark plague
            '#355E3B',  # Hunter green
            
            # Extra variations
            '#4169E1',  # Royal blue
            '#CC0033',  # Deep red
            '#00FF7F',  # Spring green
            '#1E90FF',  # Dodger blue
            '#FF033E',  # American rose
            '#50B2B2',  # Blueish green
            '#CD5C5C'   # Indian red
        ]
        
        # Ensure we have enough colors
        while len(dk_colors) < num_blocks:
            dk_colors.extend(dk_colors)
        
        # Randomly select and assign colors without repetition
        self.block_colors = random.sample(dk_colors, num_blocks)

    def to_tensor(self) -> torch.Tensor:
        clear_tensor = torch.zeros(self.num_blocks)
        clear_tensor[list(self.clear)] = 1
        ontable_tensor = torch.zeros(self.num_blocks)
        ontable_tensor[list(self.ontable)] = 1
        on_matrix = torch.zeros((self.num_blocks, self.num_blocks))
        for b1, b2 in self.on.items():
            on_matrix[b1, b2] = 1
        holding_tensor = torch.zeros(self.num_blocks)
        if self.holding is not None:
            holding_tensor[self.holding] = 1
        handempty_tensor = torch.tensor([float(self.handempty)])
        return torch.cat([
            clear_tensor,
            ontable_tensor,
            on_matrix.flatten(),
            holding_tensor,
            handempty_tensor
        ])

class BlockWorldVisualizer:
    # Theme colors
    BACKGROUND_COLOR = '#FFFFFF'
    TABLE_COLOR = '#2B4C7C'
    TABLE_EDGE_COLOR = '#4A6F9E'
    CONNECTION_COLOR = '#1A3C6E'
    
    @staticmethod
    def visualize_state(state: BlockWorldState, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
            
        ax.clear()
        ax.set_facecolor(BlockWorldVisualizer.BACKGROUND_COLOR)
        if ax.figure:
            ax.figure.set_facecolor(BlockWorldVisualizer.BACKGROUND_COLOR)
        
        margin = 1
        ax.set_xlim(-margin, state.num_blocks * 1.5 + margin)
        ax.set_ylim(-margin, state.num_blocks * 1.5 + margin)
        
        # Table
        table_width = state.num_blocks * 1.5 + 1
        table_height = 0.3
        table = Rectangle((-0.5, -table_height), table_width, table_height,
                         facecolor=BlockWorldVisualizer.TABLE_COLOR,
                         edgecolor=BlockWorldVisualizer.TABLE_EDGE_COLOR,
                         linewidth=2)
        
        table.set_path_effects([
            PathEffects.withStroke(linewidth=4,
                                 foreground=BlockWorldVisualizer.TABLE_EDGE_COLOR,
                                 alpha=0.3),
            PathEffects.withStroke(linewidth=2,
                                 foreground='#88CCFF',
                                 alpha=0.2)
        ])
        ax.add_patch(table)
        
        positions = {}
        
        # Draw blocks on table
        x = 0
        for block in state.ontable:
            positions[block] = (x, 0)
            BlockWorldVisualizer._draw_block(ax, block, x, 0, 
                                           state.block_colors[block],
                                           state.clear)
            x += 1.5
            
        # Draw stacked blocks
        for b1, b2 in state.on.items():
            if b2 in positions:
                x, y = positions[b2]
                y += 1.1
                positions[b1] = (x, y)
                BlockWorldVisualizer._draw_block(ax, b1, x, y, 
                                               state.block_colors[b1],
                                               state.clear)
                BlockWorldVisualizer._draw_connection(ax, x, y-1.1, x, y)

        # Draw held block
        if state.holding is not None:
            held_x = state.num_blocks / 2
            held_y = state.num_blocks
            BlockWorldVisualizer._draw_chains(ax, held_x, held_y + 0.5, held_x, held_y)
            BlockWorldVisualizer._draw_block(ax, state.holding, held_x, held_y,
                                           state.block_colors[state.holding],
                                           state.clear,
                                           is_held=True)
        
        if title:
            title_text = ax.set_title(title, 
                                    color=BlockWorldVisualizer.CONNECTION_COLOR,
                                    pad=20,
                                    fontsize=14,
                                    fontweight='bold')
            
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    @staticmethod
    def _draw_block(ax, block_id: int, x: float, y: float, color: str,
                    clear_blocks: Set[int], is_held: bool = False):
        size = 1.0
        
        # Main block
        block = Rectangle(
            (x - size/2, y), size, size,
            facecolor=color,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(block)
        
        # Block number
        ax.text(x, y + size/2, str(block_id),
               horizontalalignment='center',
               verticalalignment='center',
               color='white',
               fontsize=12,
               fontweight='bold',
               path_effects=[PathEffects.withStroke(linewidth=2,
                                                  foreground='black')])
        
        # Clear indicator
        if block_id in clear_blocks:
            indicator = Rectangle(
                (x - size/2, y + size - 0.1),
                size, 0.1,
                facecolor='white',
                alpha=0.5
            )
            ax.add_patch(indicator)

    @staticmethod
    def _draw_connection(ax, x1: float, y1: float, x2: float, y2: float):
        ax.plot([x1, x2], [y1, y2],
                color=BlockWorldVisualizer.CONNECTION_COLOR,
                linewidth=2,
                linestyle='-',
                path_effects=[PathEffects.withStroke(linewidth=4,
                                                   foreground=BlockWorldVisualizer.TABLE_EDGE_COLOR,
                                                   alpha=0.3)])

    @staticmethod
    def _draw_chains(ax, x1: float, y1: float, x2: float, y2: float):
        points = []
        num_segments = 4
        for i in range(num_segments + 1):
            t = i / num_segments
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            if 0 < i < num_segments:
                x += random.uniform(-0.1, 0.1)
            points.append((x, y))
        
        for i in range(len(points)-1):
            ax.plot([points[i][0], points[i+1][0]],
                   [points[i][1], points[i+1][1]],
                   color=BlockWorldVisualizer.CONNECTION_COLOR,
                   linewidth=2,
                   path_effects=[PathEffects.withStroke(linewidth=4,
                                                      foreground=BlockWorldVisualizer.TABLE_EDGE_COLOR,
                                                      alpha=0.3)])

# Example usage
if __name__ == "__main__":
    num_blocks = 6
    num_samples = 2
    
    states = []
    for _ in range(num_samples):
        state = BlockWorldState(num_blocks)
        blocks = list(range(num_blocks))
        random.shuffle(blocks)
        state.ontable = set(blocks[:2])
        state.clear = set([blocks[0]])
        state.on = {blocks[2]: blocks[1], blocks[3]: blocks[2]}
        if random.random() > 0.5:
            state.holding = blocks[4]
            state.handempty = False
        else:
            state.ontable.add(blocks[4])
            state.clear.add(blocks[4])
        states.append(state)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    for i, (ax, state) in enumerate(zip(axes, states)):
        BlockWorldVisualizer.visualize_state(
            state, ax, title=f'Icecrown Citadel - Chamber {i+1}')
    
    plt.tight_layout()
    plt.show()