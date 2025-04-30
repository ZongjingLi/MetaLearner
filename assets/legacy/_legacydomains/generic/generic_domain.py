#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : generic_domain.py
# Author : Zongjing Li
# Modified: Yiqi Sun
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Description: Implementation of a generic domain with 256D state space
# Distributed under terms of the MIT license.

import os
import torch
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    load_domain_string,
    domain_parser,
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive
from rinarak.types import treal, tvector

__all__ = [
    'GenericDomain',
    'build_generic_executor'
]

# Domain definition
dim = 256
GENERIC_DOMAIN = f"""
(domain Generic)
(:type 
    state - vector[float,{dim}]
)
(:predicate
    get_state ?x-state -> state
)
"""

class GenericDomain:
    """Handler for a generic domain with 256D state space.
    
    Implements a basic domain without any specific predicates, allowing for 
    flexible use with various reasoning tasks and metaphorical mappings.
    """
    
    def __init__(self):
        """Initialize generic domain."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_types(self, executor: CentralExecutor):
        """Setup state type with 256D signature.
        
        Args:
            executor: Executor instance to register types with
        """
        #state_type = tvector(treal, dim)  # dim-D state space
        #executor.update_registry({"state": state_type})


def build_generic_executor() -> CentralExecutor:
    """Build generic executor with domain.
        
    Returns:
        Initialized generic executor instance
    """
    # Load domain and create executor
    domain = load_domain_string(GENERIC_DOMAIN, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup types
    generic_domain = GenericDomain()
    generic_domain.setup_types(executor)
    
    return executor

# Create default executor instance
generic_executor = build_generic_executor()