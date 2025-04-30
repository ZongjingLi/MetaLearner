#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : AI Assistant
# Date   : 12/10/2024
#
# This file is part of the Domain-Specific Executor Framework.
# Distributed under terms of the MIT license.

import torch
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from networkx.drawing.nx_agraph import to_agraph
from typing import Dict, List, Tuple, Union, Optional
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string, Domain
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
grammar_path = os.path.join(current_dir, "base.grammar")

__all__ = [
    'build_domain_executor',
    'build_domain_dag',
    'get_maps_by_types',
    'get_input_output_of_mapping',
    'viz_multigraph',
    'draw_labeled_multigraph',
    'draw_knowledge_graph',
    'DifferentiableOps'
]
domain_parser = Domain(grammar_path)

def build_domain_executor(domain: str, embedding_dim: int = 128) -> 'CentralExecutor':
    """Build domain executor from domain file.
    
    Args:
        domain: Path to domain file or domain string
        embedding_dim: Dimension for concept embeddings
    
    Returns:
        Initialized CentralExecutor instance
    """
    if isinstance(domain, str):
        meta_domain_str = ""
        with open(domain, "r") as domain_file:
            meta_domain_str = domain_file.read()
            
    executor_domain = load_domain_string(meta_domain_str, domain_parser)
    return CentralExecutor(executor_domain, "cone", embedding_dim)

def build_domain_dag(domain) -> Dict:
    """Build domain predicate graph.
    
    Args:
        domain: Domain specification
        
    Returns:
        Dictionary containing predicate graphs by arity
    """
    domain_predicate_graph = {"name": domain.domain_name}
    
    for predicate_name in domain.predicates:
        params = domain.predicates[predicate_name]["parameters"]
        output = domain.predicates[predicate_name]["type"]
        arity = len(params)

        input_types = [param.split("-")[1] for param in params]

        if arity not in domain_predicate_graph:
            domain_predicate_graph[arity] = nx.MultiGraph()
            
        inputs = ",".join(input_types)
        domain_predicate_graph[arity].add_edge(inputs, output, label=predicate_name)
        
    return domain_predicate_graph

def get_maps_by_types(domain_dag: Dict, expected_input_type: str, 
                     expected_output_type: str) -> List[Dict]:
    """Find mappings matching input and output types.
    
    Args:
        domain_dag: Domain predicate graph
        expected_input_type: Input type to match
        expected_output_type: Output type to match
        
    Returns:
        List of matching mappings
    """
    matching_maps = []

    for arity, graph in domain_dag.items():
        if not isinstance(graph, nx.MultiGraph):
            continue
            
        for src_type, dst_type, edge_data in graph.edges(data=True):
            if src_type == expected_input_type and dst_type == expected_output_type:
                matching_maps.append({
                    'predicate': edge_data['label'],
                    'input_type': src_type,
                    'output_type': dst_type
                })
    
    return matching_maps

def get_input_output_of_mapping(domain_dag: Dict, arity: int, 
                              mapping_name: str) -> Optional[Tuple[str, str]]:
    """Get input and output types of a mapping.
    
    Args:
        domain_dag: Domain predicate graph
        arity: Mapping arity
        mapping_name: Name of mapping
        
    Returns:
        Tuple of input and output types, or None if not found
    """
    if arity not in domain_dag:
        return None
    
    graph = domain_dag[arity]
    
    for edge in graph.edges(data=True):
        if edge[2]["label"] == mapping_name:
            return edge[0], edge[1]
    
    return None

def viz_multigraph(graph: nx.MultiGraph, name: str = -1):
    """Visualize multigraph using pygraphviz.
    
    Args:
        graph: NetworkX multigraph
        name: Graph name for file output
    """
    multigraph = to_agraph(graph)
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        e = multigraph.get_edge(u, v, key)
        e.attr['label'] = data['label']

    multigraph.layout(prog='dot')
    multigraph.draw(f'outputs/predicate_graph_arity_{name}.png')

def draw_labeled_multigraph(G: nx.MultiGraph, attr_name: str, ax=None):
    """Draw multigraph with labeled edges.
    
    Args:
        G: NetworkX multigraph
        attr_name: Name of edge attribute to display
        ax: Matplotlib axis
    """
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", ax=ax)

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    
    nx.draw_networkx_edge_labels(
        G, pos, labels,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )

def draw_knowledge_graph(graph: nx.DiGraph):
    """Draw knowledge graph with edge labels.
    
    Args:
        graph: NetworkX directed graph
    """
    edge_labels = {
        (u,v): graph.get_edge_data(u,v)["relation"] 
        for u,v in graph.edges
    }
    
    pos = nx.spring_layout(graph)
    nx.draw(
        graph, pos,
        edge_color='black',
        width=1,
        linewidths=1,
        node_size=500,
        node_color='cyan',
        alpha=0.9,
        labels={node: node for node in graph.nodes()}
    )
    
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_color='black'
    )
# Differentiable operations as standalone functions
def gaussian_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Differentiable gaussian kernel."""
    return torch.exp(-x**2 / (2 * sigma**2))

def smooth_min(x: torch.Tensor, dim: int = -1, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable minimum using log-sum-exp."""
    return -temperature * torch.logsumexp(-x/temperature, dim=dim)

def smooth_max(x: torch.Tensor, dim: int = -1, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable maximum using log-sum-exp."""
    return temperature * torch.logsumexp(x/temperature, dim=dim)

def smooth_and(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable AND operation."""
    return torch.sigmoid((x + y - 1.5) / temperature)

def smooth_or(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable OR operation."""
    return torch.sigmoid((x + y - 0.5) / temperature)

class DifferentiableOps:
    """Differentiable operations for neural reasoning."""
    gaussian_kernel = staticmethod(gaussian_kernel)
    smooth_min = staticmethod(smooth_min)
    smooth_max = staticmethod(smooth_max)
    smooth_and = staticmethod(smooth_and)
    smooth_or = staticmethod(smooth_or)