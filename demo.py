'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-11-10 12:01:37
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-12-28 18:23:31
 # @ Description: This file is distributed under the MIT license.
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rinarak.logger import get_logger, KFTLogFormatter
from rinarak.logger import set_logger_output_file

from rinarak.domain import load_domain_string
from rinarak.knowledge.executor import CentralExecutor

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = get_logger("Citadel", KFTLogFormatter)
set_logger_output_file("logs/citadel_logs.txt")

class MetaphorMorphism(nn.Module):
    """A conceptual metaphor from the source domain to the target domain. For this domain connection"""
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return x

class ConceptDiagram(nn.Module):
    """ a directed multi graph G=(V,E) where node set V is the set of learned domains, E as the multi edge set where a pair of node is connected by some abstraction-mappings.
    """
    def __init__(self):
        super().__init__()
        self.domains = nn.ModuleDict() # store each learned domain in as Diction items. each one is a central executor
        self.morphisms = nn.ModuleDict() # store each learned morphisms that connects various domains (should be sparse) 
        self.edge_indices = defaultdict(list)

        # Add log probability parameters
        self.domain_logits = nn.ParameterDict()  # store log p for domains
        self.morphism_logits = nn.ParameterDict()  # store log p for morphisms
    
    def add_domain(self, name : str, domain : nn.Module, p = 1.0) -> None:
        """add a domain to the current concept diagram
        Args:
            name   : the name of the domain to add on
            domain : the actual domain to add to the system
        """
        if name not in self.domains:
            self.domains[name] = domain
            if p > 1.0 or p < 0.0: logger.warning(f"Input p:{p} is not within the range of [0,1]")
            self.domain_logits[name] = nn.Parameter(torch.logit(torch.ones(1) * p, eps = 1e-6))
        else: 
            logger.warning(f"try to add domain `{name}` while this name is already occcupied, overriding")
            self.domains[name] = domain
        
    def add_morphism(self, source, target, morphism : nn.Module, name : Optional[str] = None) -> None:
        """
        Args:
            source   : the name of the source domain
            target   : the name of the target domain
            morphism : the nn.Module of the morphism
            name     : the name of the current morphism (allowed multi-graph)
        """
        if source not in self.domains or target not in self.domains:
            logger.warning(f"domain not found : source not in domains:{source not in self.domains}, target not in domains: {target not in self.domains}")
            raise ValueError(f"Domain not found: {source} or {target}")
        if name is None:
            name = f"morphism_{source}_{target}_{len(self.edge_indices[(source, target)])}"
        self.morphisms[name] = morphism
        self.edge_indices[(source, target)].append(name)

        self.morphism_logits[name] = nn.Parameter(torch.logit(torch.ones(1), eps = 1e-6)) # the default value of the new morphism

    def get_morphism(self, source: str, target: str, index: int = 0) -> nn.Module:
        """get the morphism between two specific domains
        Args:
            source: source domain name
            target: target domain name
            index: the index of the morphisms if there are multiple of them
        Returns:
            the morphism as nn.Module
        """
        morphism_names = self.edge_indices[(source, target)]
        if not morphism_names:
            raise ValueError(f"No morphism found from {source} to {target}")
        morphism_name = morphism_names[index]
        return self.morphisms[morphism_name]
    
    def get_domain_prob(self, name: str) -> torch.Tensor:
        """Get the probability of a domain's existence"""
        return torch.sigmoid(self.domain_logits[name])  # Using sigmoid for non-negative values

    def get_morphism_prob(self, name: str) -> torch.Tensor:
        """Get the probability of a morphism's existence"""
        return torch.sigmoid(self.morphism_logits[name]) # Using sigmoid for non-negative values


    def evaluate(self, state : torch.Tensor, predicate : str, domain : str = None):
        """ evaluate a predicate on the given state. if not specified, predicate is evaluate on the root domain.
        This is done by traversing the state through the shortest path that leads from the `domain` the state is in to the domain the predicate evaluated was in.
        Args:
            state     : a torch tensor of shape [nxd]
            predicate : a name of the predicate from some node
            domain    : the domain the encoded state is from, if none then evaluate on.
        Returns:
            output the evaluation result of the predicate on the state as [nx*] tensor
        """
        return
    
    def get_all_morphisms(self, source: str, target: str) -> List[nn.Module]:
        """get all the morphisms between the source domain and target domain: source -> target
        Args:
            source: the name of the source domain
            target: the name of the target domain
        Returns:
            a list contains the possible morphisms by the (name : str, morphism : nn.Module)
        """
        morphism_names = self.edge_indices[(source, target)]
        return [(name, self.morphisms[name]) for name in morphism_names]
    
    def get_path(self, 
                 source: str, 
                 target: str, 
                 max_length: int = 3) -> List[List[Tuple[str, str, int]]]:
        """find all the possible paths from source domain to the target domain.
        Args:
            source: the name of the source domain
            target: the name of the target domain
            max_length: maximum length of the 
            
        Returns:
            a list of all the paths, each path is a list of tuples of (source, target, index)
        """
        def dfs(current: str, 
               path: List[Tuple[str, str, int]], 
               visited: set) -> List[List[Tuple[str, str, int]]]:
            if len(path) > max_length:
                return []
            if current == target:
                return [path]
                
            paths = []
            for (src, tgt), morphism_names in self.edge_indices.items():
                if src == current and tgt not in visited:
                    for idx, _ in enumerate(morphism_names):
                        new_visited = visited | {tgt}
                        new_path = path + [(src, tgt, idx)]
                        new_paths = dfs(tgt, new_path, new_visited)
                        paths.extend(new_paths)
            return paths
            
        return dfs(source, [], {source})

    def get_path_prob(self, path: List[Tuple[str, str, int]]) -> torch.Tensor:
        """Calculate the log probability of a path by summing log probabilities"""
        log_prob = 0.0
        # Add source domain probability
        if path:
            source_domain = path[0][0]
            log_prob += torch.log(self.get_domain_prob(source_domain))

        # Add probabilities along the path
        for source, target, idx in path:
            log_prob += torch.log(self.get_domain_prob(target)) # Add target domain probability
            morphism_name = self.edge_indices[(source, target)][idx]
            log_prob += torch.log(self.get_morphism_prob(morphism_name))# Add morphism probability
            
        return log_prob

    def compose_path(self, path: List[Tuple[str, str, int]]) -> nn.Module:
        """compose morphisms along the path
        Args:
            path: path a list of tuples of (source, target,index)
        Returns:
            a composed module that applis the state transition according to path
        """
        class ComposedMorphism(nn.Module):
            def __init__(self, morphisms: List[nn.Module]):
                super().__init__()
                self.morphisms = nn.ModuleList(morphisms)
                
            def forward(self, x):
                for morphism in self.morphisms:
                    x = morphism(x)
                return x
                
        # get the morphisms along the path
        morphisms = []
        for source, target, idx in path:
            morphism = self.get_morphism(source, target, idx)
            morphisms.append(morphism)
            
        return ComposedMorphism(morphisms)

    def exists_path(self, source : str, target : str) -> torch.Tensor:
        """probability mask of there exists a path between source domain and target domain
        Args:
            source : the source domain name
            target : the target domain name
        Returns:
            the probbaility there exists a path between the source domain and the target domain
        """
        all_paths = self.get_path(source, target)
        
        if not all_paths:
            return torch.tensor(0.0)
            
        # Calculate log probability for each path
        path_log_probs = torch.stack([self.get_path_prob(path) for path in all_paths])
        
        # Return max probability (using log-sum-exp trick for numerical stability)
        max_log_prob = torch.max(path_log_probs)
        return max_log_prob.exp()
    
    def get_most_probable_path(self, source: str, target: str) -> Tuple[List[Tuple[str, str, int]], torch.Tensor]:
        """Get the path with highest probability and its probability"""
        all_paths = self.get_path(source, target)
        
        if not all_paths:
            return None, torch.tensor(0.0)
            
        path_probs = [(path, self.get_path_prob(path)) for path in all_paths]
        best_path, best_prob = max(path_probs, key=lambda x: x[1])
        
        return best_path, best_prob.exp()

    def visualize(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """Visualize the concept diagram with probabilities."""
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx is required for visualization")
            return
                
        fig, ax = plt.subplots(figsize=figsize)
        G = nx.DiGraph()
        
        # Add nodes with probabilities
        for domain_name in self.domains.keys():
            prob = self.get_domain_prob(domain_name).item()
            G.add_node(domain_name, probability=prob)
        
        # Add edges with probabilities and track morphisms
        edge_labels = {}
        for (source, target), morph_names in self.edge_indices.items():
            for idx, morph_name in enumerate(morph_names):
                prob = self.get_morphism_prob(morph_name).item()
                G.add_edge(source, target, 
                         probability=prob,
                         morphism_name=morph_name)
                # Add edge label with probability
                edge_labels[(source, target)] = f"{prob:.2f}"
        
        # Create layout with more spread
        pos = nx.kamada_kawai_layout(G)  # Better layout for small graphs
        
        # Draw nodes with probability-based coloring
        node_colors = [G.nodes[node]['probability'] for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(G, pos,
                                     node_color=node_colors,
                                     node_size=2000, 
                                     cmap='YlOrRd',
                                     vmin=0.0,
                                     vmax=1.0,
                                     ax=ax)
        
        # Draw edges
        edges = nx.draw_networkx_edges(G, pos,
                                     arrowsize=20,
                                     connectionstyle='arc3,rad=0.2',
                                     ax=ax,
                                     edge_color='darkgray',
                                     width=2,
                                     alpha=0.6)
        
        # Add node labels with probabilities
        labels = {node: f"{node}\n{G.nodes[node]['probability']:.2f}" 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)
        
        # Add edge labels (morphism probabilities)
        edge_label_pos = nx.draw_networkx_edge_labels(G, pos,
                                                    edge_labels=edge_labels,
                                                    label_pos=0.5,
                                                    font_size=8)
        
        # Add colorbar for probability scale
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
        plt.colorbar(sm, ax=ax, label='Domain Probability')
        
        ax.set_title('Concept Diagram\nNode color: Domain probability\nEdge labels: Morphism probability')
        ax.axis('off')
        
        # Adjust layout to prevent text overlap
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def visualize_path(self, source: str, target: str, save_path: Optional[str] = None):
        """Visualize the most probable path between two domains.
        
        Args:
            source: Source domain name
            target: Target domain name
            save_path: Optional path to save the visualization
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx is required for visualization")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Get most probable path
        path, prob = self.get_most_probable_path(source, target)
        if path is None:
            plt.text(0.5, 0.5, f"No path found from {source} to {target}",
                    ha='center', va='center')
            plt.axis('off')
            if save_path:
                plt.savefig(save_path, dpi=300)
            plt.show()
            return
        
        # Create graph
        G = nx.DiGraph()
        
        # Add all nodes and mark path nodes
        path_nodes = {source}.union(t for _, t, _ in path)
        for domain_name in self.domains.keys():
            G.add_node(domain_name, in_path=domain_name in path_nodes)
        
        # Add all edges and mark path edges
        path_edges = [(s, t) for s, t, _ in path]
        for (s, t), _ in self.edge_indices.items():
            G.add_edge(s, t, in_path=(s, t) in path_edges)
        
        # Create layout
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        
        # Draw non-path elements
        non_path_nodes = [n for n in G.nodes() if not G.nodes[n]['in_path']]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=non_path_nodes,
                              node_color='lightgray',
                              node_size=800)
        
        # Draw path elements
        path_nodes = [n for n in G.nodes() if G.nodes[n]['in_path']]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=path_nodes,
                              node_color='lightblue',
                              node_size=1000)
        
        # Draw edges
        non_path_edges = [(s, t) for s, t in G.edges() if (s, t) not in path_edges]
        nx.draw_networkx_edges(G, pos,
                              edgelist=non_path_edges,
                              edge_color='lightgray',
                              width=1,
                              alpha=0.5,
                              connectionstyle='arc3,rad=0.2')
        
        nx.draw_networkx_edges(G, pos,
                              edgelist=path_edges,
                              edge_color='blue',
                              width=2,
                              arrowsize=20,
                              connectionstyle='arc3,rad=0.2')
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(f'Most Probable Path: {source} → {target}\nPath Probability: {float(prob):.3f}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == "__main__":
    from domains.line.line_domain import line_executor
    from domains.rcc8.rcc8_domain import rcc8_executor
    from domains.curve.curve_domain import curve_executor
    from domains.direction.direction_domain import direction_executor
    logger.info("create the concept diagram with empty graph.")
    concept_diagram = ConceptDiagram()
    concept_diagram.add_domain("LineDomain", line_executor)
    concept_diagram.add_domain("CurveDomain", curve_executor)
    concept_diagram.add_domain("RCC8Domain", rcc8_executor)
    concept_diagram.add_domain("DirectionDomain", direction_executor)


    concept_diagram.add_morphism("CurveDomain", "LineDomain", nn.Linear(20,40))
    concept_diagram.add_morphism("LineDomain", "RCC8Domain", nn.Linear(20,409))
    concept_diagram.add_morphism("LineDomain", "RCC8Domain", nn.Linear(20,409))

    
    print(concept_diagram.get_all_morphisms("CurveDomain", "RCC8Domain"))

    print(concept_diagram.get_path("CurveDomain", "RCC8Domain"))

    print(concept_diagram.get_most_probable_path("CurveDomain", "RCC8Domain"))
    print(concept_diagram.exists_path("CurveDomain", "RCC8Domain"))
    print(concept_diagram.exists_path("CurveDomain", "DirectionDomain"))\
    
    # Visualize the entire concept diagram
    concept_diagram.visualize()
    concept_diagram.visualize_path("CurveDomain", "RCC8Domain")