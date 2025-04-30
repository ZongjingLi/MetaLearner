#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : blockworld.py
# Author : Zongjing Li
# Modified: [Assistant]
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of BlockWorld predicates
# Distributed under terms of the MIT license.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    load_domain_string,
    domain_parser,
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow, GlobalContext
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

__all__ = [
    'BlockWorldDomain',
    'build_blockworld_executor'
]

# Domain definition
BLOCKWORLD_DOMAIN = """
(domain BlockWorld)
(:type
    state - vector[float,3]
    position - vector[float,2]
)
(:predicate
    block_position ?x-state -> position
    on ?x-state ?y-state -> boolean
    clear ?x-state -> boolean
    holding ?x-state -> boolean
    hand-free -> boolean
)
(:action
    (
        name: pick
        parameters: ?o1
        precondition: (and (clear ?o1) (hand-free) )
        effect:
        (and-do
            (and-do
                (assign (holding ?o1) true)
                (assign (clear ?o1) false)
            )
            (assign (hand-free) false)
        )
    )
    (
        name: place
        parameters: ?o1 ?o2
        precondition:
            (and (holding ?o1) (clear ?o2))
        effect :
            (and-do
            (and-do
                        (assign (hand-free) true)
                (and-do
                        (assign (holding ?o1) false)
                    (and-do
                        (assign (clear ?o2) false)
                        (assign (clear ?o1) true)
                    )
                )
                
            )
                (assign (on ?x ?y) true)
            )
    )
)

"""

class BlockWorldDomain:
    """Handler for BlockWorld predicates and spatial relations.
    
    Implements differentiable predicates for reasoning about blocks in a 2D space
    with stacking capabilities. Each block has a position (x,y) and holding state.
    Supports relations like on, clear, above, and hand operations.
    """
    
    def __init__(self, temperature: float = 0.01):
        """Initialize BlockWorld domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations (lower for sharper transitions)
        """
        self.temperature = temperature
        
        # Physical constants
        self.table_height = 0.0
        self.block_height = 1.0
        self.block_width = 1.0
        self.margin = 0.05  # Alignment tolerance
        self.holding_height = 3.0
        
        # Color scheme for visualization
        self.colors = [
            '#1f2937', '#94a3b8', '#0369a1', '#1e293b', '#0369a1',
            '#cbd5e1', '#0c4a6e', '#475569', '#164e63', '#e2e8f0',
            '#082f49', '#22d3ee'
        ]
    
    def block_position(self, x_state: torch.Tensor) -> torch.Tensor:
        """Extract 2D position from state.
        
        Args:
            x_state: [B, 3] tensor of block states [x, y, holding]
            
        Returns:
            [B, 2] tensor of positions
        """
        return x_state[:, :2]
    
    def on_table(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if blocks are on the table.
        
        Args:
            x_state: [B, 3] tensor of block states
            
        Returns:
            [B] tensor of on_table scores
        """
        heights = x_state[:, 1]
        target_height = self.table_height + self.block_height/2
        return torch.sigmoid(-(torch.abs(heights - target_height) - self.margin) / self.temperature)
    
    def on(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if blocks are directly on top of other blocks.
        
        Args:
            x_state: [B1, 3] tensor of upper block states
            y_state: [B2, 3] tensor of lower block states
            
        Returns:
            [B1, B2] tensor of on relation scores
        """
        heights1 = x_state[:, 1].unsqueeze(1)  # [B1, 1]
        heights2 = y_state[:, 1]               # [B2]
        positions1 = x_state[:, 0].unsqueeze(1)  # [B1, 1]
        positions2 = y_state[:, 0]               # [B2]
        
        height_diff = heights1 - heights2
        position_diff = torch.abs(positions1 - positions2)
        
        correct_height = torch.sigmoid(-(torch.abs(height_diff - self.block_height) - self.margin) / self.temperature)
        aligned = torch.sigmoid(-(position_diff - self.margin) / self.temperature)
        
        return correct_height * aligned
    
    def clear(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Check if blocks have no other blocks on top.
        
        Args:
            context: Dictionary containing state tensor [B, 3]
            
        Returns:
            [B] tensor of clear scores
        """
        x_state = context["state"]
        
        height_diffs = x_state[:, 1].unsqueeze(1) - x_state[:, 1]
        position_diffs = torch.abs(x_state[:, 0].unsqueeze(1) - x_state[:, 0])
        
        is_above = torch.sigmoid((height_diffs - self.block_height) / self.temperature)
        is_aligned = torch.sigmoid(-(position_diffs - self.margin) / self.temperature)
        
        # Mask diagonal (block compared to itself)
        mask = 1 - torch.eye(len(x_state), device=x_state.device)
        is_above = is_above * mask
        
        has_block_above = torch.max(is_above * is_aligned, dim=0)[0]
        return 1 - has_block_above
    
    def above(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if blocks are above (not necessarily directly) other blocks.
        
        Args:
            x_state: [B1, 3] tensor of upper block states
            y_state: [B2, 3] tensor of lower block states
            
        Returns:
            [B1, B2] tensor of above relation scores
        """
        heights1 = x_state[:, 1].unsqueeze(1)
        heights2 = y_state[:, 1]
        positions1 = x_state[:, 0].unsqueeze(1)
        positions2 = y_state[:, 0]
        
        min_height_diff = self.block_height * 0.1
        is_higher = torch.sigmoid((heights1 - heights2 - min_height_diff) / self.temperature)
        aligned = torch.sigmoid(-(torch.abs(positions1 - positions2) - self.margin) / self.temperature)
        
        return is_higher * aligned
    
    def holding(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if blocks are being held.
        
        Args:
            x_state: [B, 3] tensor of block states
            
        Returns:
            [B] tensor of holding scores
        """
        return torch.sigmoid(x_state[:, 2:3])
    
    def exists(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if blocks exist (always true for valid states).
        
        Args:
            x_state: [B, 3] tensor of block states
            
        Returns:
            [B] tensor of ones
        """
        return torch.ones(len(x_state))

    def visualize(self, context: Dict[str, torch.Tensor],
                 relation_matrix: Optional[torch.Tensor] = None,
                 program: Optional[str] = None) -> plt.Figure:
        """Visualize BlockWorld state and relations.
        
        Args:
            context: Dictionary containing state tensor
            relation_matrix: Optional tensor of relation scores
            program: Optional program string to display
            
        Returns:
            Matplotlib figure with visualization
        """
        # Setup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                      gridspec_kw={'width_ratios': [1, 1]})
        
        state = context["state"]
        n_blocks = len(state)
        
        # Draw table
        self._draw_table(ax1)
        
        # Draw blocks
        self._draw_blocks(ax1, state)
        
        # Setup grid and axes
        self._setup_axes(ax1)
        
        # Draw relation matrix if provided
        if relation_matrix is not None:
            self._draw_relation_matrix(ax2, relation_matrix, n_blocks)
        
        if program is not None:
            fig.suptitle(
                program,
                y=1.02,
                fontsize=13,
                color='#4a4a4a',
                fontweight='bold'
            )
        
        plt.tight_layout()
        return fig

    def _draw_table(self, ax: plt.Axes):
        """Draw the table surface."""
        table = patches.Rectangle(
            (-3, self.table_height - 0.1),
            6, 0.2,
            facecolor='#e5e5e5',
            edgecolor='#b0b0b0',
            linewidth=1,
            zorder=1
        )
        ax.add_patch(table)

    def _draw_blocks(self, ax: plt.Axes, state: torch.Tensor):
        """Draw blocks with labels and holding indicators."""
        for i in range(len(state)):
            pos = state[i]
            is_held = pos[2].item() > 0
            
            x = pos[0].item() - self.block_width/2
            y = pos[1].item() - self.block_height/2
            
            # Main block
            block = patches.Rectangle(
                (x, y),
                self.block_width,
                self.block_height,
                facecolor=self.colors[i % len(self.colors)],
                edgecolor='#4a4a4a',
                linewidth=1,
                alpha=0.9 if is_held else 0.8,
                zorder=2
            )
            ax.add_patch(block)
            
            # Label
            ax.text(
                pos[0].item(),
                pos[1].item(),
                f'B{i}',
                ha='center',
                va='center',
                color='white',
                fontsize=11,
                fontweight='bold',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='#2f2f2f')],
                zorder=3
            )
            
            if is_held:
                self._draw_holding_indicator(ax, pos, x, y)

    def _draw_holding_indicator(self, ax: plt.Axes,
                              pos: torch.Tensor,
                              block_x: float,
                              block_y: float):
        """Draw holding indicator for held blocks."""
        ax.plot(
            [pos[0].item(), pos[0].item()],
            [block_y + self.block_height, self.holding_height],
            color='#4a4a4a',
            linestyle=':',
            linewidth=1.5,
            alpha=0.6,
            zorder=1
        )
        
        highlight = patches.Rectangle(
            (block_x - 0.02, block_y - 0.02),
            self.block_width + 0.04,
            self.block_height + 0.04,
            facecolor='none',
            edgecolor='#f0f0f0',
            linewidth=1.5,
            alpha=0.7,
            zorder=2
        )
        ax.add_patch(highlight)

    def _setup_axes(self, ax: plt.Axes):
        """Setup axes properties."""
        ax.grid(True, linestyle='-', color='#f0f0f0', linewidth=0.5, zorder=0)
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        ax.set_ylim(-1, 4)
        ax.set_xlim(-3.5, 3.5)
        
        ax.tick_params(colors='#808080', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_title(
            "Block Configuration",
            pad=15,
            fontsize=12,
            color='#4a4a4a',
            fontweight='bold'
        )

    def _draw_relation_matrix(self, ax: plt.Axes,
                            relation_matrix: torch.Tensor,
                            n_blocks: int):
        """Draw relation matrix visualization."""
        im = ax.imshow(
            relation_matrix.detach().numpy(),
            cmap='Blues',
            aspect='equal',
            interpolation='nearest'
        )
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=10, colors='#4a4a4a')
        
        block_labels = [f'B{i}' for i in range(n_blocks)]
        ax.set_xticks(range(n_blocks))
        ax.set_yticks(range(n_blocks))
        ax.set_xticklabels(block_labels, fontsize=10, color='#4a4a4a')
        ax.set_yticklabels(block_labels, fontsize=10, color='#4a4a4a')
        
        ax.grid(False)
        ax.set_title(
            "Relation Matrix",
            pad=15,
            fontsize=12,
            color='#4a4a4a',
            fontweight='bold'
        )
        
        for spine in ax.spines.values():
            spine.set_visible(False)

    def setup_predicates(self, executor: CentralExecutor):
        """Setup all BlockWorld predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        state_type = tvector(treal, 3)  # [x, y, holding]
        position_type = tvector(treal, 2)  # [x, y]
        
        executor.update_registry({
            "block_position": Primitive(
                "block_position",
                arrow(state_type, position_type),
                lambda x: {**x, "end": self.block_position(x["state"])}
            ),
            
            "on_table": Primitive(
                "on_table",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.on_table(x["state"])}
            ),
            
            "on": Primitive(
                "on",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.on(x["state"], y["state"])}
            ),
            
            "clear": Primitive(
                "clear",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.clear(x)}
            ),
            
            "above": Primitive(
                "above",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.above(x["state"], y["state"])}
            ),
            "holding": Primitive(
                "holding",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.holding(x["state"])}
            ),
            
            "exists": Primitive(
                "exists",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.exists(x["state"])}
            ),
            
            "hand-free": Primitive(
                "hand-free",
                arrow(boolean),
                lambda x: {**x, "end": 1.0 - torch.max(x["state"][:, 2:3])}
            )
        })

    def observe_state(self, state: Dict[str, Any], scene: bool = False) -> Dict[str, Any]:
        """Process state to add derived properties like clear and hand-free.
        
        Args:
            state: Dictionary of state information
            scene: Whether state is a single scene or multiple states
            
        Returns:
            Augmented state dictionary with derived properties
        """
        if not scene:
            outputs = {}
            for i in state:
                holding = state[i]["state"][:, 2:3]
                clear_tensor = self.clear(state[i])
                outputs[i] = {
                    **state[i],
                    "hand-free": 1.0 - torch.max(holding),
                    "clear": clear_tensor
                }
            return outputs

        holding = state["state"][:, 2:3]
        clear_tensor = self.clear(state)
        return {
            **state,
            "hand-free": 1.0 - torch.max(holding),
            "clear": clear_tensor
        }


def build_blockworld_executor(temperature: float = 0.01) -> CentralExecutor:
    """Build BlockWorld executor with domain.
    
    Args:
        temperature: Temperature for smooth operations
        
    Returns:
        Initialized BlockWorld executor
    """
    # Load domain and create executor
    domain = load_domain_string(BLOCKWORLD_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup predicates
    blockworld_domain = BlockWorldDomain(temperature)
    blockworld_domain.setup_predicates(executor)
    
    # Add visualization and state observer
    executor.visualize = blockworld_domain.visualize
    executor.observe_state = blockworld_domain.observe_state
    
    return executor

# Create default executor instance
blockworld_executor = build_blockworld_executor()