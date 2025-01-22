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

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from core.metaphors.base import StateMapper
from core.metaphors.legacy import PredicateConnectionMatrix, ActionConnectionMatrix
from rinarak.utils.data import combine_dict_lists

logger = get_logger("Citadel", KFTLogFormatter)
set_logger_output_file("logs/citadel_logs.txt")

class MetaphorMorphism(nn.Module):
    """A conceptual metaphor from source domain to target domain"""
    def __init__(self, 
                 source_domain: CentralExecutor,
                 target_domain: CentralExecutor,
                 hidden_dim: int = 256):
        super().__init__()
        self.source_domain = source_domain
        self.target_domain = target_domain

        """f_a: used to check is the metaphor is applicable for the mapping"""
        
        """f_s: as the state mapping from source state to the target state"""
        self.state_mapper = StateMapper(
            source_dim=source_domain.state_dim[0],
            target_dim=target_domain.state_dim[0],
            hidden_dim=hidden_dim
        )
        
        """f_d: as the predicate and action connections between the source domain and target domain"""
        self.predicate_matrix = PredicateConnectionMatrix(
            source_domain.domain, target_domain.domain
        )
        self.action_matrix = ActionConnectionMatrix(
            source_domain.domain, target_domain.domain
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map state from source to target domain"""
        return self.state_mapper(state)
        
    def get_predicate_mapping(self, source_pred: str, target_pred: str) -> torch.Tensor:
        """Get mapping weight between predicates"""
        return self.predicate_matrix.get_connection_weight(source_pred, target_pred)
        
    def get_action_mapping(self, source_action: str, target_action: str) -> torch.Tensor:
        """Get mapping weight between actions"""
        return self.action_matrix.get_connection_weight(source_action, target_action)

class MetaphorBindingTracker:
    def __init__(self):
        self.bound_pairs = set()
        self.binding_strengths = {}
        
    def bind_metaphors(self, path1: List[Tuple[str, str, int]], 
                      path2: List[Tuple[str, str, int]], 
                      strength: float = 1.0):
        path_key = (tuple(path1), tuple(path2))
        self.bound_pairs.add(path_key)
        self.binding_strengths[path_key] = strength
        
    def get_binding_strength(self, path1: List[Tuple[str, str, int]], 
                           path2: List[Tuple[str, str, int]]) -> float:
        path_key = (tuple(path1), tuple(path2))
        return self.binding_strengths.get(path_key, 0.0)
        
    def are_paths_bound(self, path1: List[Tuple[str, str, int]], 
                       path2: List[Tuple[str, str, int]]) -> bool:
        path_key = (tuple(path1), tuple(path2))
        return path_key in self.bound_pairs


class ConceptDiagram(nn.Module):
    """A directed multi-graph G=(V,E) where node set V is the set of learned domains, 
    E as the multi edge set where a pair of nodes is connected by some abstraction-mappings."""
    
    def __init__(self):
        super().__init__()
        self.domains = nn.ModuleDict()  # Store domains (CentralExecutor instances)
        self.morphisms = nn.ModuleDict()  # Store morphisms (sparse connections)
        self.edge_indices = defaultdict(list)
        self.domain_logits = nn.ParameterDict()  # Store log p for domains
        self.morphism_logits = nn.ParameterDict()  # Store log p for morphisms
        self.evaluation_tracker = MetaphorBindingTracker()

    def add_domain(self, name: str, domain: nn.Module, p: float = 1.0) -> None:
        if name not in self.domains:
            self.domains[name] = domain
            if p > 1.0 or p < 0.0:
                logger.warning(f"Input p:{p} is not within the range of [0,1]")
            self.domain_logits[name] = nn.Parameter(torch.logit(torch.ones(1) * p, eps=1e-6))
        else:
            logger.warning(f"try to add domain `{name}` while this name is already occupied, overriding")
            self.domains[name] = domain

    def add_morphism(self, source: str, target: str, morphism: nn.Module, 
                    name: Optional[str] = None) -> None:
        if source not in self.domains or target not in self.domains:
            logger.warning(f"domain not found: source not in domains:{source not in self.domains}, "
                         f"target not in domains: {target not in self.domains}")
            raise ValueError(f"Domain not found: {source} or {target}")
            
        if name is None:
            name = f"morphism_{source}_{target}_{len(self.edge_indices[(source, target)])}"
            
        self.morphisms[name] = morphism
        self.edge_indices[(source, target)].append(name)
        self.morphism_logits[name] = nn.Parameter(torch.logit(torch.ones(1), eps=1e-6))

    def get_morphism(self, source: str, target: str, index: int = 0) -> nn.Module:
        morphism_names = self.edge_indices[(source, target)]
        if not morphism_names:
            raise ValueError(f"No morphism found from {source} to {target}")
        morphism_name = morphism_names[index]
        return self.morphisms[morphism_name]
    
    def get_all_morphisms(self, source: str, target: str) -> List[Tuple[str, nn.Module]]:
        """Get all morphisms between the source domain and target domain.
    
        Args:
            source (str): Name of the source domain
            target (str): Name of the target domain
        
        Returns:
            List of tuples containing (morphism_name, morphism_module)
        """
        morphism_names = self.edge_indices[(source, target)]
        return [(name, self.morphisms[name]) for name in morphism_names]

    def get_domain_prob(self, name: str) -> torch.Tensor:
        return torch.sigmoid(self.domain_logits[name])

    def get_morphism_prob(self, name: str) -> torch.Tensor:
        return torch.sigmoid(self.morphism_logits[name])

    def evaluate(self, state: torch.Tensor, predicate: str, domain: str = None, 
                eval_type: str = 'actual', top_k: int = 3) -> torch.Tensor:
        """Evaluate a predicate on the given state using specified evaluation method."""
        
        # Find predicate domain if not specified
        pred_domain = None
        pred_arity = -1
        for domain_name, domain_ in self.domains.items():
            for arity in domain_.predicates:
                for dom_pred in domain_.predicates[arity]:
                    if str(predicate) == str(dom_pred):
                        pred_domain = domain_name
                        pred_arity = arity
                        break
        if pred_domain is None or pred_arity == -1:
            raise ValueError(f"Predicate {predicate} not found in any domain")

        # If source domain not specified, find most probable domain for state
        if domain is None:
            domain_probs = {}
            for domain_name, domain_executor in self.domains.items():
                try:
                    prob = domain_executor.evaluate_state_compatibility(state)
                    domain_probs[domain_name] = prob
                except:
                    continue
            if not domain_probs:
                raise ValueError("Could not determine source domain for state")
            domain = max(domain_probs.items(), key=lambda x: x[1])[0]
        
        if pred_arity == 0:
            predicate = f"({predicate})"
        if pred_arity == 1:
            predicate = f"({predicate} $0)"
        if pred_arity == 2:
            predicate = f"({predicate} $0 $1)"

        # Choose evaluation method
        if eval_type == 'actual':
            return self._evaluate_actual(state, predicate, domain, pred_domain, top_k)
        elif eval_type == 'metaphor':
            return self._evaluate_metaphor(state, predicate, domain, pred_domain, top_k)
        elif eval_type == 'prob_metaphor':
            return self._evaluate_prob_metaphor(state, predicate, domain, pred_domain, top_k)
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

    def _evaluate_actual(self, state: torch.Tensor, predicate_expr: str, 
                        source_domain: str, target_domain: str, top_k: int) -> torch.Tensor:
        """Actual evaluation following all paths."""
        all_paths = self.get_path(source_domain, target_domain)
        if not all_paths:
            return torch.zeros_like(state[:, 0])

        states = []
        results = []
        path_probs = []
        """1. get all the paths from the source to target domain"""
        for path in all_paths[:top_k]:
            current_state = state
            path_prob = torch.exp(self.get_path_prob(path))
            """2. calculate the probability of this path exists"""
            for src, tgt, idx in path:
                morphism = self.get_morphism(src, tgt, idx)
                current_state = morphism(current_state)

            target_executor = self.domains[target_domain]

            assert isinstance(target_executor, CentralExecutor), "not an central executor"
            context = {0:{"state" : current_state}, 1:{"state" : current_state}}
            pred_result = target_executor.evaluate(predicate_expr, context)


            states.append(pred_result["state"])
            results.append(pred_result["end"])
            path_probs.append(path_prob)

        path_probs = torch.stack(path_probs)
        path_probs = path_probs / path_probs.sum()

        final_result = torch.zeros_like(results[0])
        for result, prob in zip(results, path_probs):
            final_result += result * prob

        return {"end":final_result, "states" : states, "results" : results}
    
    def _evaluate_metaphor(self, state: torch.Tensor, predicate_expr: str,
                          source_domain: str, target_domain: str, top_k: int) -> torch.Tensor:
        """Metaphorical evaluation using earliest valid evaluation point by tracing predicates backwards.
        For a predicate p in target domain, we trace back through the path to find where it
        originates from (where it has strong connections to source predicates).
        """
        all_paths = self.get_path(source_domain, target_domain)
        if not all_paths:
            return {"end": torch.zeros_like(state[:, 0]), "states": [], "results": []}

        results = []
        states = []
        path_probs = []
        connection_threshold = 0.5  # Threshold for significant connections
        
        for path in all_paths[:top_k]:
            current_state = state
            path_prob = torch.exp(self.get_path_prob(path))
            
            # Start from target domain and trace backwards
            current_pred = str(predicate_expr.split(" ")[0][1:])  # Target predicate
            evaluation_domain = target_domain
            path_reversed = list(reversed(path))
            
            # Transform state forward along path
            for src, tgt, idx in path:
                morphism = self.get_morphism(src, tgt, idx)
                current_state = morphism(current_state)
            
            # Trace predicate backwards until we find its origin
            for idx, (src, tgt, morph_idx) in enumerate(path_reversed[:-1]):  # Skip last step as it's source domain
                #print()
                morphism = self.get_morphism(src, tgt, morph_idx)
                
                # Get connection matrix for this step
                connection_matrix, _ = morphism.predicate_matrix()
                
                # Find strongly connected predicates in source domain
                src_domain = self.domains[src]
                tgt_domain = self.domains[tgt]
                
                # Check connections to source predicates
                found_connection = False
                for src_pred in src_domain.predicates.get(1, []):  # Assuming binary predicates for now
                    if src_pred in src_domain.predicates.get(1, []):
                        connection_strength = morphism.get_predicate_mapping(src_pred, current_pred)
                        if connection_strength > connection_threshold:
                            current_pred = src_pred
                            evaluation_domain = src
                            found_connection = True
                            break
                            
                if not found_connection:
                    break  # No strong connections found, evaluate at current domain
            
            # Evaluate at the determined domain
            domain_executor = self.domains[evaluation_domain]
            assert isinstance(domain_executor, CentralExecutor), "not an central executor"
            
            # Adjust predicate expression for the evaluation domain
            if evaluation_domain != target_domain:
                # Reconstruct predicate expression for the source predicate
                predicate_parts = predicate_expr.split(" ")
                predicate_parts[0] = f"({current_pred}"
                eval_predicate_expr = " ".join(predicate_parts)
            else:
                eval_predicate_expr = predicate_expr
                
            context = {0: {"state": current_state}, 1: {"state": current_state}}
            pred_result = domain_executor.evaluate(eval_predicate_expr, context)
            
            states.append(pred_result["state"])
            results.append(pred_result["end"])
            path_probs.append(path_prob)

        if results:
            path_probs = torch.stack(path_probs)
            path_probs = path_probs / path_probs.sum()

            final_result = torch.zeros_like(results[0])
            for result, prob in zip(results, path_probs):
                final_result += result * prob

            return {"end": final_result, "states": states, "results": results}
        
        return {"end": torch.zeros_like(state[:, 0]), "states": [], "results": []}

    def _evaluate_prob_metaphor(self, state: torch.Tensor, predicate: str,
                              source_domain: str, target_domain: str, top_k: int) -> torch.Tensor:
        """Probabilistic metaphorical evaluation with domain selection."""
        all_paths = self.get_path(source_domain, target_domain)
        if not all_paths:
            return torch.zeros_like(state[:, 0])

        results = []
        confidences = []

        for path in all_paths[:top_k]:
            current_state = state
            path_prob = self.get_path_prob(path).exp()

            for src, tgt, idx in path:
                morphism = self.get_morphism(src, tgt, idx)
                current_state = morphism(current_state)
                domain_prob = self.get_domain_prob(tgt)

                domain_executor = self.domains[tgt]
                if predicate in domain_executor.predicates:
                    pred_result = domain_executor.evaluate_predicate(predicate, current_state)
                    confidence = path_prob * domain_prob * self._compute_confidence(path, predicate)

                    # Consider bound paths
                    for other_path in all_paths:
                        if self.evaluation_tracker.are_paths_bound(path, other_path):
                            binding_strength = self.evaluation_tracker.get_binding_strength(
                                path, other_path)
                            confidence = confidence * binding_strength

                    results.append(pred_result)
                    confidences.append(confidence)

        if not results:
            return torch.zeros_like(state[:, 0])

        confidences = torch.stack(confidences)
        confidences = confidences / confidences.sum()

        final_result = torch.zeros_like(results[0])
        for result, conf in zip(results, confidences):
            final_result += result * conf

        return final_result

    def _compute_confidence(self, path: List[Tuple[str, str, int]], predicate: str) -> torch.Tensor:
        """Compute confidence score for a path and predicate."""
        confidence = torch.tensor(1.0)
        length_penalty = 1.0 / (len(path) + 1)
        confidence *= length_penalty

        for src, tgt, idx in path:
            morphism = self.get_morphism(src, tgt, idx)
            pred_matrix, _ = morphism.predicate_matrix()
            max_connection = pred_matrix.max()
            confidence *= max_connection

        return confidence

    
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

    def visualize_connection_matrices(self, path: List[Tuple[str, str, int]], predicate: str):
        """Visualize predicate connection matrices along a path.
        
        Args:
            path: List of tuples (source, target, idx) representing the path
            predicate: The predicate to track connections for
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not path:
            print("Empty path provided")
            return
            
        num_steps = len(path)
        fig, axes = plt.subplots(1, num_steps, figsize=(5*num_steps, 4))
        if num_steps == 1:
            axes = [axes]
        
        current_pred = predicate
        composite_matrix = None
        
        # For each step in the path
        for idx, (src, tgt, morph_idx) in enumerate(path):
            morphism = self.get_morphism(src, tgt, morph_idx)
            
            # Get source and target domain predicates
            src_domain = self.domains[src]
            tgt_domain = self.domains[tgt]
            
            # Get connection matrix for this morphism
            if isinstance(morphism, MetaphorMorphism):
                # Use forward method instead of accessing matrix directly
                connection_matrix, _ = morphism.predicate_matrix()
                connection_matrix = connection_matrix.detach().cpu()
                
                # If first step, initialize composite matrix
                if composite_matrix is None:
                    composite_matrix = connection_matrix
                else:
                    # Compose with previous matrices
                    composite_matrix = torch.matmul(composite_matrix, connection_matrix)
                
                # Plot the connection matrix
                ax = axes[idx]
                sns.heatmap(connection_matrix, ax=ax, cmap='YlOrRd', 
                           xticklabels=tgt_domain.predicates.get(1, []),
                           yticklabels=src_domain.predicates.get(1, []))
                
                # Highlight the current predicate
                if current_pred in src_domain.predicates.get(1, []):
                    pred_idx = src_domain.predicates.get(1, []).index(current_pred)
                    ax.get_yticklabels()[pred_idx].set_color('red')
                
                ax.set_title(f'Step {idx+1}: {src} → {tgt}\nMorphism {morph_idx}')
                
                # Update current predicate mapping
                if hasattr(morphism, 'map_predicate'):
                    current_pred = morphism.map_predicate(current_pred)
            
            plt.tight_layout()
        
        # Add a final plot for the composite matrix if path length > 1
        if len(path) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(composite_matrix, ax=ax, cmap='YlOrRd',
                       xticklabels=self.domains[path[-1][1]].predicates.get(1, []),
                       yticklabels=self.domains[path[0][0]].predicates.get(1, []))
            ax.set_title('Composite Connection Matrix\n(Full Path)')
            plt.tight_layout()
        
        plt.show()


    def metaphorical_evaluation(self, source_state: torch.Tensor, target_predicate: str,
                                source_predicate: Optional[str] = None, source_domain : Optional[str] = None, eval_type : str = "actual",
                                visualize: bool = False) -> Dict[str, Any]:
        """
        Perform metaphorical evaluation between source and target domains.

        Args:
            source_state: State tensor in the source domain
            target_predicate: Corresponding predicate in the target domain
            source_predicate: Predicate to evaluate in the source domain (optional)
            visualize: Whether to visualize the evaluation process (default: False)

        Returns:
            Dictionary containing evaluation results, states, and other relevant information
        """
        # Find source and target domain executors
        source_executor = None
        target_executor = None
        source_domain_name = source_domain
        target_domain_name = None

        for domain_name, executor in self.domains.items():
            for predicate in combine_dict_lists(executor.predicates):
                if source_predicate is not None and str(predicate) == source_predicate:
                    source_executor = executor
                    source_domain_name = domain_name
                if str(predicate) == target_predicate:
                    target_executor = executor
                    target_domain_name = domain_name
        
        if target_executor is None:
            raise ValueError(f"Could not find executor for target predicate: {target_predicate}")

        # If source predicate not provided, use target domain for source evaluation
        if source_predicate is None:
            source_executor = target_executor

        # Evaluate source predicate
        source_context = {
            0: {"end": 1.0, "state": source_state},
            1: {"end": 1.0, "state": source_state}
        }

        if source_predicate is not None:
            source_result = source_executor.evaluate(f"({source_predicate} $0 $1)", source_context)
        else:
            n = source_state.shape[0]
            source_result = {"end":torch.zeros([n,n]), "state" : source_state}

        # Perform metaphorical evaluation
        evaluation_result = self.evaluate(source_state, target_predicate,
                                          source_domain_name, eval_type)
        

        target_results = evaluation_result["results"]
        target_states = evaluation_result["states"]

        # Prepare target context for visualization

        #print("target:",target_states[0].shape)
        target_context = {
            0: {"end": 1.0, "state": target_states[0].detach()},
            1: {"end": 1.0, "state": target_states[0].detach()}
        }

        # Visualize source and target domains (optional)
        if visualize:
            if "Generic" not in source_domain:
                source_executor.visualize(source_context, source_result["end"].detach())
            target_executor.visualize(target_context, target_results[0].detach())
            plt.show()

        return {
            "source_result": source_result,
            "target_results": target_results,
            "target_states": target_states,
            "source_context": source_context,
            "target_context": target_context
        }

def visualize_predicate_tracing(
    path: List[Tuple[str, str, int]],
    domains: Dict[str, Any],
    morphisms: Dict[str, Any],
    target_predicate: str,
    figsize: Tuple[int, int] = (15, 5*3)
) -> None:
    """
    Visualize the predicate tracing process with connection matrices.
    
    Args:
        path: List of (source, target, morphism_idx) tuples defining the path
        domains: Dictionary of domain executors
        morphisms: Dictionary of morphisms between domains
        target_predicate: The predicate being traced
        figsize: Figure size for the plot
    """
    path_reversed = list(reversed(path))
    n_steps = len(path_reversed) - 1  # Exclude last step (source domain)
    
    # Create subplots - one row per step plus title
    fig, axes = plt.subplots(n_steps + 1, 2, figsize=figsize)
    fig.suptitle(f'Predicate Tracing Process for "{target_predicate}"', fontsize=16, y=0.95)
    
    # Keep track of predicates for labeling
    current_pred = target_predicate
    pred_trace = [current_pred]
    
    # Process each step in the path
    for idx, (src, tgt, morph_idx) in enumerate(path_reversed[:-1]):
        morphism = morphisms.get((src, tgt, morph_idx))
        if not morphism:
            continue
            
        # Get connection matrix
        connection_matrix, pred_mapping = morphism.predicate_matrix()
        
        # Get domain predicates
        src_domain = domains[src]
        tgt_domain = domains[tgt]
        src_preds = src_domain.predicates.get(1, [])  # Assuming binary predicates
        tgt_preds = tgt_domain.predicates.get(1, [])
        
        # Create heatmap of connection matrix
        ax_matrix = axes[idx][0]
        sns.heatmap(
            connection_matrix,
            ax=ax_matrix,
            cmap='YlOrRd',
            xticklabels=src_preds,
            yticklabels=tgt_preds,
            cbar_kws={'label': 'Connection Strength'}
        )
        ax_matrix.set_title(f'Connection Matrix: {tgt} → {src}')
        ax_matrix.set_xlabel('Source Predicates')
        ax_matrix.set_ylabel('Target Predicates')
        
        # Highlight current predicate
        current_pred_idx = tgt_preds.index(current_pred)
        ax_matrix.axhline(current_pred_idx + 0.5, color='blue', alpha=0.3)
        
        # Find strongest connection
        max_connection_idx = np.argmax(connection_matrix[current_pred_idx])
        new_pred = src_preds[max_connection_idx]
        pred_trace.append(new_pred)
        current_pred = new_pred
        
        # Create bar plot of connections for current predicate
        ax_bar = axes[idx][1]
        connections = connection_matrix[current_pred_idx]
        sns.barplot(
            x=src_preds,
            y=connections,
            ax=ax_bar,
            color='skyblue'
        )
        ax_bar.set_title(f'Connection Strengths for "{pred_trace[idx]}"')
        ax_bar.set_xlabel('Source Predicates')
        ax_bar.set_ylabel('Connection Strength')
        ax_bar.tick_params(axis='x', rotation=45)
        
        # Highlight strongest connection
        ax_bar.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax_bar.legend()
    
    # Add predicate trace summary
    ax_summary = axes[-1][0]
    ax_summary.axis('off')
    summary_text = 'Predicate Trace:\n' + ' → '.join(reversed(pred_trace))
    ax_summary.text(0.1, 0.5, summary_text, fontsize=12, wrap=True)
    
    # Remove unused subplot
    fig.delaxes(axes[-1][1])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from domains.generic.generic_domain import generic_executor
    from domains.line.line_domain import line_executor
    from domains.rcc8.rcc8_domain import rcc8_executor
    from domains.curve.curve_domain import curve_executor
    from domains.distance.distance_domain import distance_executor
    from domains.direction.direction_domain import direction_executor
    logger.info("create the concept diagram with empty graph.")
    concept_diagram = ConceptDiagram()
    concept_diagram.add_domain("GenericDomain", generic_executor)
    concept_diagram.add_domain("LineDomain", line_executor)
    concept_diagram.add_domain("CurveDomain", curve_executor)
    concept_diagram.add_domain("RCC8Domain", rcc8_executor)
    concept_diagram.add_domain("DistanceDomain", distance_executor)
    concept_diagram.add_domain("DirectionDomain", direction_executor)


    concept_diagram.add_morphism("GenericDomain", "LineDomain", MetaphorMorphism(generic_executor, line_executor))
    concept_diagram.add_morphism("GenericDomain", "DistanceDomain", MetaphorMorphism(generic_executor, distance_executor))
    concept_diagram.add_morphism("GenericDomain", "DirectionDomain", MetaphorMorphism(generic_executor, direction_executor))

    concept_diagram.add_morphism("DistanceDomain", "DirectionDomain", MetaphorMorphism(distance_executor, direction_executor))

    concept_diagram.add_morphism("CurveDomain", "LineDomain", MetaphorMorphism(curve_executor, line_executor))
    concept_diagram.add_morphism("LineDomain", "RCC8Domain", MetaphorMorphism(line_executor, rcc8_executor))
    concept_diagram.add_morphism("LineDomain", "RCC8Domain", MetaphorMorphism(line_executor, rcc8_executor))
    concept_diagram.add_morphism("DistanceDomain", "RCC8Domain", MetaphorMorphism(line_executor, rcc8_executor))
    
    print(concept_diagram.exists_path("GenericDomain", "DirectionDomain"))
    print(concept_diagram.exists_path("GenericDomain", "DistanceDomain"))


    path = concept_diagram.get_path("GenericDomain", "RCC8Domain")[0]  # Get first path



    """generic state space testing"""
    source_state = torch.randn([5, 256])
    evaluation_result = concept_diagram.metaphorical_evaluation(
        source_state, "near", source_domain = "GenericDomain", eval_type = "actual",  visualize = False
    )
    evaluation_result = concept_diagram.metaphorical_evaluation(
        source_state, "near", source_domain = "GenericDomain", eval_type= 'metaphor', visualize = False
    )
    #concept_diagram.visualize_metaphorical_trace(evaluation_result, "GenericDomain", "DirectionDomain", 
    #                                       "near", connection_threshold=0.5)
    evaluation_result["target_results"][0]
    print("Metaphor:", evaluation_result["target_results"][0])


    evaluation_result = concept_diagram.metaphorical_evaluation(
        source_state, "south_of", source_domain = "GenericDomain", visualize = False
    )


import sys
sys.exit()

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def optimize_concept_params(
    concept_diagram,
    initial_state: torch.Tensor,
    predicate: str = "north",
    source_domain: str = "GenericDomain",
    num_steps: int = 1000,
    lr: float = 0.01,
    visualization_steps: int = 1
):
    """
    Optimize source state parameters to minimize target results in concept diagram
    
    Args:
        concept_diagram: The concept diagram for evaluation
        initial_state: Initial state tensor to optimize
        predicate: Predicate to evaluate
        source_domain: Source domain name
        num_steps: Number of optimization steps
        lr: Learning rate
        visualization_steps: Steps between visualizations
    """
    # Create parameter to optimize
    source_state = torch.nn.Parameter(initial_state.clone())
    optimizer = optim.Adam([source_state], lr=lr)
    
    # Store losses for plotting
    losses = []
    
    # Optimization loop
    pbar = tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()
        
        # Evaluate current state
        evaluation_result = concept_diagram.metaphorical_evaluation(
            source_state, predicate, source_domain=source_domain,eval_type = "metaphor", visualize=False
        )
        
        # Get target results and compute loss
        target_results = evaluation_result["target_results"][0]
        loss = torch.sum(target_results ** 2)  # Try to make all elements zero
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Update progress bar
        pbar.set_description(f"Loss: {loss.item():.6f}")
        
        # Visualize every visualization_steps steps
        if step % visualization_steps == 0:
            """
            print(f"\nStep {step}")
            print("Target Results:", target_results.detach())
            
            # Plot loss curve
            plt.figure(figsize=(10, 4))
            plt.plot(losses)
            plt.title("Optimization Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.yscale('log')
            plt.grid(True)
            plt.show()
            plt.close()
            """
            final_evaluation = concept_diagram.metaphorical_evaluation(
        source_state, predicate, source_domain=source_domain, visualize=True
        )
            plt.pause(0.01)
            #print(final_evaluation["target_states"][0])
    
    # Final evaluation
    final_evaluation = concept_diagram.metaphorical_evaluation(
        source_state, predicate, source_domain=source_domain, visualize=True
    )
    
    return {
        "optimized_state": source_state.detach(),
        "final_evaluation": final_evaluation,
        "loss_history": losses
    }

# Example usage
if __name__ == "__main__":
    # Initial setup
    source_state = torch.randn([5, 256])
    
    # Run optimization
    results = optimize_concept_params(
        concept_diagram=concept_diagram,
        initial_state=source_state,
        predicate="near",
        num_steps=1000,
        lr=0.01
    )
    
    # Print final results
    print("\nFinal Results:")
    print("Target Results:", results["final_evaluation"]["target_results"][0])
    print("Final Loss:", results["loss_history"][-1])
    
    # Plot final loss curve
    plt.figure(figsize=(12, 4))
    plt.plot(results["loss_history"])
    plt.title("Complete Optimization History")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.show()