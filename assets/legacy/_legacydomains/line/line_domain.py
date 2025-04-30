#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : line.py
# Author : Zongjing Li
# Modified: [Assistant]
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of line predicates
# Distributed under terms of the MIT license.

import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    load_domain_string,
    domain_parser,
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

__all__ = [
    'LineDomain',
    'build_line_executor'
]

# Domain definition
LINE_DOMAIN = """
(domain Line)
(:type
    state - vector[float,1]
    position - float
    distance - float
)
(:predicate
    get_position ?x-state -> position
    start ?x-state -> boolean
    end ?x-state -> boolean
    near_start ?x-state -> boolean
    near_end ?x-state -> boolean
    before ?x-state ?y-state -> boolean
    after ?x-state ?y-state -> boolean
    distance ?x-state ?y-state -> distance
    close_to ?x-state ?y-state -> boolean
    far_from ?x-state ?y-state -> boolean
    between ?x-state ?y-state ?z-state -> boolean
)
"""

class LineDomain:
    """Handler for line predicates and linear ordering relations.
    
    Implements differentiable predicates for reasoning about points on a line
    segment normalized to [0,1]. Supports positional predicates (start, end),
    relative positioning (before, after), and distance-based relations.
    """
    
    def __init__(self, temperature: float = 0.1, epsilon: float = 1e-6):
        """Initialize line domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls transition sharpness
            epsilon: Small value for numerical stability
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize points to [0,1] range using sigmoid.
        
        Args:
            x: Input tensor of points
            
        Returns:
            Normalized points in [0,1]
        """
        return torch.sigmoid(x)

    def start(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if points are at start of line (x ≈ 0).
        
        Args:
            x_state: [B, 1] tensor of points
            
        Returns:
            [B] tensor of start scores
        """
        x_norm = self._normalize(x_state)
        return torch.sigmoid((-x_norm) / self.temperature)

    def end(self, x_state: torch.Tensor) -> torch.Tensor:
        """Check if points are at end of line (x ≈ 1).
        
        Args:
            x_state: [B, 1] tensor of points
            
        Returns:
            [B] tensor of end scores
        """
        x_norm = self._normalize(x_state)
        return torch.sigmoid((x_norm - 1.0) / self.temperature)

    def near_start(self, x_state: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
        """Check if points are near start of line.
        
        Args:
            x_state: [B, 1] tensor of points
            threshold: Distance threshold from start
            
        Returns:
            [B] tensor of near_start scores
        """
        x_norm = self._normalize(x_state)
        return torch.sigmoid((threshold - x_norm) / self.temperature)

    def near_end(self, x_state: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
        """Check if points are near end of line.
        
        Args:
            x_state: [B, 1] tensor of points
            threshold: Distance threshold from end
            
        Returns:
            [B] tensor of near_end scores
        """
        x_norm = self._normalize(x_state)
        return torch.sigmoid((threshold - (1.0 - x_norm)) / self.temperature)

    def before(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if x points come before y points on line.
        
        Args:
            x_state: [B1, 1] tensor of first points
            y_state: [B2, 1] tensor of second points
            
        Returns:
            [B1, B2] tensor of before scores
        """
        x_norm = self._normalize(x_state)
        y_norm = self._normalize(y_state)
        x_exp = x_norm.unsqueeze(1)  # [B1, 1]
        y_exp = y_norm.unsqueeze(0)  # [1, B2]
        return torch.sigmoid((y_exp - x_exp) / self.temperature).squeeze(-1)

    def after(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Check if x points come after y points on line.
        
        Args:
            x_state: [B1, 1] tensor of first points
            y_state: [B2, 1] tensor of second points
            
        Returns:
            [B1, B2] tensor of after scores
        """
        return 1.0 - self.before(x_state, y_state)

    def distance(self, x_state: torch.Tensor, y_state: torch.Tensor) -> torch.Tensor:
        """Compute distances between points on line.
        
        Args:
            x_state: [B1, 1] tensor of first points
            y_state: [B2, 1] tensor of second points
            
        Returns:
            [B1, B2] tensor of distances
        """
        x_norm = self._normalize(x_state)
        y_norm = self._normalize(y_state)
        x_exp = x_norm.unsqueeze(1)
        y_exp = y_norm.unsqueeze(0)
        return torch.abs(x_exp - y_exp)

    def close_to(self, x_state: torch.Tensor, y_state: torch.Tensor, 
                threshold: float = 0.3) -> torch.Tensor:
        """Check if points are close to each other.
        
        Args:
            x_state: [B1, 1] tensor of first points
            y_state: [B2, 1] tensor of second points
            threshold: Distance threshold for closeness
            
        Returns:
            [B1, B2] tensor of close_to scores
        """
        distances = self.distance(x_state, y_state)
        return torch.sigmoid((threshold - distances) / self.temperature)

    def far_from(self, x_state: torch.Tensor, y_state: torch.Tensor,
                threshold: float = 0.3) -> torch.Tensor:
        """Check if points are far from each other.
        
        Args:
            x_state: [B1, 1] tensor of first points
            y_state: [B2, 1] tensor of second points
            threshold: Distance threshold for farness
            
        Returns:
            [B1, B2] tensor of far_from scores
        """
        distances = self.distance(x_state, y_state)
        return torch.sigmoid((distances - threshold) / self.temperature)

    def between(self, x_state: torch.Tensor, y_state: torch.Tensor,
               z_state: torch.Tensor) -> torch.Tensor:
        """Check if x points lie between y and z points.
        
        Args:
            x_state: [B1, 1] tensor of test points
            y_state: [B2, 1] tensor of bound points
            z_state: [B3, 1] tensor of bound points
            
        Returns:
            [B1, B2, B3] tensor of between scores
        """
        x_norm = self._normalize(x_state)
        y_norm = self._normalize(y_state)
        z_norm = self._normalize(z_state)
        
        x_exp = x_norm.view(-1, 1, 1)  # [B1, 1, 1]
        y_exp = y_norm.view(1, -1, 1)  # [1, B2, 1]
        z_exp = z_norm.view(1, 1, -1)  # [1, 1, B3]
        
        # Check both orderings: y < x < z and z < x < y
        case1 = torch.sigmoid((x_exp - y_exp) / self.temperature) * \
                torch.sigmoid((z_exp - x_exp) / self.temperature)
        case2 = torch.sigmoid((x_exp - z_exp) / self.temperature) * \
                torch.sigmoid((y_exp - x_exp) / self.temperature)
        
        return case1 + case2

    def setup_predicates(self, executor: CentralExecutor):
        """Setup all line predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        state_type = tvector(treal, 1)  # 1D position
        position_type = treal  # scalar position
        distance_type = treal  # scalar distance
        
        executor.update_registry({
            "get_position": Primitive(
                "get_position",
                arrow(state_type, position_type),
                lambda x: {**x, "end": x["state"]}
            ),
            
            "start": Primitive(
                "start",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.start(x["state"])}
            ),
            
            "end": Primitive(
                "end",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.end(x["state"])}
            ),
            
            "near_start": Primitive(
                "near_start",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.near_start(x["state"])}
            ),
            
            "near_end": Primitive(
                "near_end",
                arrow(state_type, boolean),
                lambda x: {**x, "end": self.near_end(x["state"])}
            ),
            
            "before": Primitive(
                "before",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.before(x["state"], y["state"])}
            ),
            
            "after": Primitive(
                "after",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.after(x["state"], y["state"])}
            ),
            
            "distance": Primitive(
                "distance",
                arrow(state_type, arrow(state_type, distance_type)),
                lambda x: lambda y: {**x, "end": self.distance(x["state"], y["state"])}
            ),
            
            "close_to": Primitive(
                "close_to",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.close_to(x["state"], y["state"])}
            ),
            
            "far_from": Primitive(
                "far_from",
                arrow(state_type, arrow(state_type, boolean)),
                lambda x: lambda y: {**x, "end": self.far_from(x["state"], y["state"])}
            ),
            
            "between": Primitive(
                "between",
                arrow(state_type, arrow(state_type, arrow(state_type, boolean))),
                lambda x: lambda y: lambda z: {**x, "end": self.between(x["state"], y["state"], z["state"])}
            )
        })


def build_line_executor(temperature: float = 0.1) -> CentralExecutor:
    """Build line executor with domain.
    
    Args:
        temperature: Temperature for smooth operations
        
    Returns:
        Initialized line executor instance
    """
    # Load domain and create executor
    domain = load_domain_string(LINE_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup predicates
    line_domain = LineDomain(temperature)
    line_domain.setup_predicates(executor)
    
    # Add visualization from external module
    from .visualize import visualize_line_predicates
    executor.visualize = visualize_line_predicates
    
    return executor

# Create default executor instance
line_executor = build_line_executor()