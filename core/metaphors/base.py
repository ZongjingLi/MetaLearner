'''
 # @ Author: Zongjing Li
 # @ Create Time: 2025-01-01 13:54:27
 # @ Modified by: Zongjing Li
 # @ Modified time: 2025-01-01 13:54:31
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import math

class RelationalStateProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Calculate attention embedding dimension that's divisible by num_heads
        self.attention_dim = math.ceil(input_dim / num_heads) * num_heads
        
        # Add linear projection to make input compatible with attention dimension
        self.input_projection = nn.Linear(input_dim, self.attention_dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.attention_dim)
        
        # Final MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(self.attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D inputs

        if state.dim() == 2:
            pass
            #state = state.unsqueeze(0)  # Add batch dimension
        
        # Project input to attention-compatible dimension
        state = self.input_projection(state)
        
        # Apply self-attention
        attended_state, _ = self.self_attention(state, state, state)
        
        # Residual connection and layer norm
        state = self.layer_norm(state + attended_state)
        
        # Global average pooling over sequence dimension
        pooled_state = torch.mean(state, dim=0)
        
        # Final MLP projection

        return self.mlp(pooled_state)

class DomainEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(embedding_dim))
        
    def forward(self) -> torch.Tensor:
        return self.embedding

import torch
import torch.nn as nn

class StateMapper(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(source_dim, hidden_dim)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: Input tensor of shape (batch_size, num_states, source_dim)
                   If unbatched, shape should be (num_states, source_dim)
        Returns:
            Tensor of shape (batch_size, num_states, target_dim)
            If unbatched, shape will be (num_states, target_dim)
        """
        # Add batch dimension if necessary
        if states.dim() == 2:
            states = states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Initial projection
        x = self.input_proj(states)
        
        # Self-attention block
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward block
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Output projection
        output = self.output_proj(x)
        
        # Process specific components if needed (similar to original code)
        if output.shape[-1] > 2:
            output_before = output[..., :2]
            output_third = output[..., 2:3]
            output_after = output[..., 3:] if output.shape[-1] > 3 else None
            
            if output_after is not None:
                output = torch.cat([output_before, output_third, output_after], dim=-1)
            else:
                output = torch.cat([output_before, output_third], dim=-1)
        
        # Remove batch dimension if input was unbatched
        if squeeze_output:
            output = output.squeeze(0)
            
        return output
    
    def process_large_set(self, states: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        """
        Process large sets by chunking them to avoid memory issues.
        
        Args:
            states: Input tensor of shape (num_states, source_dim)
            chunk_size: Number of states to process at once
        Returns:
            Tensor of shape (num_states, target_dim)
        """
        if states.dim() != 2:
            raise ValueError("process_large_set expects unbatched input")
            
        num_states = states.shape[0]
        outputs = []
        
        for i in range(0, num_states, chunk_size):
            chunk = states[i:i + chunk_size]
            chunk_output = self.forward(chunk)
            outputs.append(chunk_output)
            
        return torch.cat(outputs, dim=0)

class StateClassifier(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(source_dim, hidden_dim)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

        # Additional components for logit computation
        self.logit_attention = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=1,
            batch_first=True
        )
        
        self.logit_norm = nn.LayerNorm(target_dim)
        
        # Projection for reducing N states to a single scalar
        self.state_reduction = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: Input tensor of shape (batch_size, num_states, source_dim)
                   If unbatched, shape should be (num_states, source_dim)
        Returns:
            Tensor of shape (batch_size, num_states, target_dim)
            If unbatched, shape will be (num_states, target_dim)
        """
        # Add batch dimension if necessary
        if states.dim() == 2:
            states = states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Initial projection
        x = self.input_proj(states)
        
        # Self-attention block
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward block
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Output projection
        output = self.output_proj(x)
        
        # Process specific components if needed
        if output.shape[-1] > 2:
            output_before = output[..., :2]
            output_third = output[..., 2:3]
            output_after = output[..., 3:] if output.shape[-1] > 3 else None
            
            if output_after is not None:
                output = torch.cat([output_before, output_third, output_after], dim=-1)
            else:
                output = torch.cat([output_before, output_third], dim=-1)
        
        # Remove batch dimension if input was unbatched
        if squeeze_output:
            output = output.squeeze(0)
            
        return output

    def compute_logit(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute a single scalar logit from an NxD tensor input by first mapping the states
        and then reducing them to a single value.
        
        Args:
            states: Input tensor of shape (num_states, source_dim)
        Returns:
            Tensor of shape (1,) containing a single scalar logit
        """
        # First get the mapped states using the regular forward pass
        mapped_states = self.forward(states)
        
        # Average pool across the N states dimension
        pooled = torch.mean(mapped_states, dim=0, keepdim=True)
        
        # Project to scalar logit
        logit = self.state_reduction(pooled)
        
        return logit.squeeze(0)  # Return scalar

def calculate_state_domain_connection(
    state_embedding: torch.Tensor,
    domain_embedding: torch.Tensor
) -> torch.Tensor:
    scale = np.sqrt(state_embedding.shape[0])
    return torch.sigmoid(torch.dot(state_embedding, domain_embedding) / scale)
