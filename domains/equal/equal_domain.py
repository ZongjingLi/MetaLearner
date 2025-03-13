#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rcc8.py
# Author : Zongjing Li
# Modified: [Assistant]
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Binary relation matrix implementation of RCC8 (Region Connection Calculus)
# Distributed under terms of the MIT license.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    DifferentiableOps,
    load_domain_string,
    domain_parser,
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

# Equal domain definition
EQUAL_DOMAIN = """
(domain Equal)
(:type
    vector - vector[float, n]
    threshold - float
    boolean - vector[float, 1]
)
(:predicate
    equal ?x-vector ?y-vector -> boolean
    not_equal ?x-vector ?y-vector -> boolean
    approx_equal ?x-vector ?y-vector ?t-threshold -> boolean
    angle_between ?x-vector ?y-vector -> float
    cos_similarity ?x-vector ?y-vector -> float
    orthogonal ?x-vector ?y-vector -> boolean
    parallel ?x-vector ?y-vector -> boolean
)
"""

class EqualDomain:
    """Handler for vector equality predicates based on angular similarity.
    
    Implements differentiable predicates for comparing vectors based on
    cosine similarity and angular relationships.
    """
    
    def __init__(self, temperature: float = 0.1, epsilon: float = 1e-6):
        """Initialize equal domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls transition sharpness
            epsilon: Small value for numerical stability
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _normalize_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit length.
        
        Args:
            x: Input tensor of vectors
            
        Returns:
            Normalized vectors
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (norm + self.epsilon)
    
    def cos_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between vectors.
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            
        Returns:
            [B1, B2] tensor of cosine similarities in [-1, 1]
        """
        # Normalize vectors to unit length
        x_norm = self._normalize_vector(x)
        y_norm = self._normalize_vector(y)
        
        # Compute dot product across batch
        x_expanded = x_norm.unsqueeze(1)  # [B1, 1, D]
        y_expanded = y_norm.unsqueeze(0)  # [1, B2, D]
        
        # Dot product, result shape: [B1, B2]
        similarity = torch.sum(x_expanded * y_expanded, dim=-1)
        
        # Clamp to account for numerical issues
        return torch.clamp(similarity, -1.0 + self.epsilon, 1.0 - self.epsilon)
    
    def angle_between(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute angle between vectors in radians.
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            
        Returns:
            [B1, B2] tensor of angles in [0, π]
        """
        similarity = self.cos_similarity(x, y)
        return torch.acos(similarity)
    
    def equal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check if vectors are equal (pointing in same direction).
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            
        Returns:
            [B1, B2] tensor of equality scores
        """
        similarity = self.cos_similarity(x, y)
        # Vectors are equal if cosine similarity is close to 1
        return torch.sigmoid((similarity - 1.0 + self.epsilon) / self.temperature)
    
    def not_equal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check if vectors are not equal (different directions).
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            
        Returns:
            [B1, B2] tensor of inequality scores
        """
        return 1.0 - self.equal(x, y)
    
    def approx_equal(self, x: torch.Tensor, y: torch.Tensor, 
                     threshold: float = 0.1) -> torch.Tensor:
        """Check if vectors are approximately equal within angle threshold.
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            threshold: Angle threshold in radians (default ~5.7 degrees)
            
        Returns:
            [B1, B2] tensor of approximate equality scores
        """
        angles = self.angle_between(x, y)
        return torch.sigmoid((threshold - angles) / self.temperature)
    
    def orthogonal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check if vectors are orthogonal (90 degree angle).
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            
        Returns:
            [B1, B2] tensor of orthogonality scores
        """
        similarity = self.cos_similarity(x, y)
        # Vectors are orthogonal if cosine similarity is close to 0
        return torch.sigmoid(-(torch.abs(similarity)) / self.temperature)
    
    def parallel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check if vectors are parallel (0 or 180 degree angle).
        
        Args:
            x: [B1, D] tensor of first vectors
            y: [B2, D] tensor of second vectors
            
        Returns:
            [B1, B2] tensor of parallelism scores
        """
        similarity = self.cos_similarity(x, y)
        # Vectors are parallel if absolute cosine similarity is close to 1
        return torch.sigmoid((torch.abs(similarity) - 1.0 + self.epsilon) / self.temperature)
    
    def setup_predicates(self, executor: CentralExecutor):
        """Setup all Equal domain predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        vector_type = tvector(treal, 'n')  # n-dimensional vector
        boolean_type = tvector(treal, 1)   # boolean representation
        float_type = treal                 # scalar value
        
        executor.update_registry({
            "equal": Primitive(
                "equal",
                arrow(vector_type, arrow(vector_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.equal(x["state"], y["state"])}
            ),
            
            "not_equal": Primitive(
                "not_equal",
                arrow(vector_type, arrow(vector_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.not_equal(x["state"], y["state"])}
            ),
            
            "approx_equal": Primitive(
                "approx_equal",
                arrow(vector_type, arrow(vector_type, arrow(float_type, boolean_type))),
                lambda x: lambda y: lambda t: {**x, "end": self.approx_equal(x["state"], y["state"], threshold=t["state"])}
            ),
            
            "angle_between": Primitive(
                "angle_between",
                arrow(vector_type, arrow(vector_type, float_type)),
                lambda x: lambda y: {**x, "end": self.angle_between(x["state"], y["state"])}
            ),
            
            "cos_similarity": Primitive(
                "cos_similarity",
                arrow(vector_type, arrow(vector_type, float_type)),
                lambda x: lambda y: {**x, "end": self.cos_similarity(x["state"], y["state"])}
            ),
            
            "orthogonal": Primitive(
                "orthogonal",
                arrow(vector_type, arrow(vector_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.orthogonal(x["state"], y["state"])}
            ),
            
            "parallel": Primitive(
                "parallel",
                arrow(vector_type, arrow(vector_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.parallel(x["state"], y["state"])}
            )
        })