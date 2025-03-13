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
from matplotlib.colors import to_rgba
import networkx as nx
from io import BytesIO
import base64
from rinarak.logger import get_logger, KFTLogFormatter
from rinarak.logger import set_logger_output_file

from rinarak.domain import load_domain_string
from rinarak.knowledge.executor import CentralExecutor

from typing import Dict, Union, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from core.metaphors.base import StateMapper, StateClassifier
from core.metaphors.legacy import PredicateConnectionMatrix, ActionConnectionMatrix
from rinarak.utils.data import combine_dict_lists
from rinarak.knowledge.executor import type_dim

from graphviz import Digraph
import itertools

import torch

def display(obj):  
        # Base case: if object is a tensor, return its shape
        if isinstance(obj, torch.Tensor):
            return f"tensor{list(obj.shape)}"
        
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {key: display(value) for key, value in obj.items()}
        
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            result = [display(item) for item in obj]
            # Return as the same type as the input
            if isinstance(obj, tuple):
                return tuple(result)
            return result
            
        # Handle sets
        elif isinstance(obj, set):
            return {display(item) for item in obj}
            
        # Return other types unchanged
        else:
            return obj

class MetaphorMorphism(nn.Module):
    """A conceptual metaphor from source domain to target domain"""
    def __init__(self, 
                 source_domain: CentralExecutor,
                 target_domain: CentralExecutor,
                 hidden_dim: int = 256):
        super().__init__()
        self.source_domain = source_domain
        self.target_domain = target_domain

        # Extract source & target type names
        self.source_types = list(source_domain.types.keys())
        self.target_types = list(target_domain.types.keys())

        self.num_source_types = len(self.source_types)
        self.num_target_types = len(self.target_types)

        # Learnable type matching matrix (num_source_types × num_target_types)
        self.type_matching = nn.Parameter(torch.randn(self.num_source_types, self.num_target_types))

        """f_a: used to check is the metaphor is applicable for the mapping"""

        self.mappers = nn.ModuleDict({})
        for s_type in source_domain.types:
            s_dim = type_dim(source_domain.types[s_type])[0][0]
            for t_type in target_domain.types:
                t_dim = type_dim(target_domain.types[t_type])[0][0]
                """f_s: as the state mapping from source state to the target state"""
                self.mappers[f"{s_type}_{t_type}_map"] = \
                    StateMapper(
                        source_dim=s_dim,
                        target_dim=t_dim,
                        hidden_dim=hidden_dim
                    )
                self.mappers[f"{s_type}_{t_type}_classify"] = \
                    StateMapper(
                        source_dim = s_dim,
                        target_dim = 1,
                        hidden_dim = hidden_dim
                    )

        """f_d: as the predicate and action connections between the source domain and target domain"""
        self.predicate_matrix = PredicateConnectionMatrix(
            source_domain.domain, target_domain.domain
        )
        self.action_matrix = ActionConnectionMatrix(
            source_domain.domain, target_domain.domain
        )

    def get_best_match(self, s_type: str) -> Tuple[str, float]:
        """
        Get the best matching target type for a given source type based on the learned weights.
        
        Args:
            s_type (str): Source type name.

        Returns:
            Tuple[str, float]: Best matching target type and its corresponding probability.
        """
        if s_type not in self.source_types:
            raise ValueError(f"Unknown source type: {s_type}")

        # Get index of source type
        s_index = self.source_types.index(s_type)

        # Normalize matching weights (row-wise softmax)
        #matching_probs = F.softmax(self.type_matching, dim=1)
        matching_probs = torch.sigmoid(self.type_matching)

        # Get target index with highest probability
        best_t_index = torch.argmax(matching_probs[s_index]).item()
        best_prob = matching_probs[s_index, best_t_index]


        return self.target_types[best_t_index], best_prob

    def get_match(self, s_type, t_type):

        if s_type not in self.source_types:
            raise ValueError(f"Unknown source type: {s_type}")
        if t_type not in self.target_types:
            raise ValueError(f"Unknown target type: {t_type} not in {self.target_types}")
    
        # Get indices of source and target types
        s_index = self.source_types.index(s_type)
        t_index = self.target_types.index(t_type)
    
        # Normalize matching weights (row-wise softmax)
        #matching_probs = F.softmax(self.type_matching, dim=1)
        matching_probs = torch.sigmoid(self.type_matching)


        # Get the specific probability for this source-target pair
        prob = matching_probs[s_index, t_index]
    
        return prob

    def get_type_matching(self):return  F.softmax(self.type_matching, dim=1)

    def forward(self, args: Union[List[Tuple[Dict, torch.Tensor]], List[Dict]], best_match_only: bool = True, expect_type = None) -> List[Tuple[Dict, torch.Tensor]]:
        """
        Maps states from source to target domain based on learned type matching.

        Args:
            args (Union[List[Tuple[Dict, torch.Tensor]], List[Dict]]): 
                - If a list of tuples: Each tuple contains:
                    - context (Dict): A dictionary with keys:
                        - "end" (Tensor): Input state tensor.
                        - "scores": (Tensor): Input score tensor.
                        - "domain" (str): Domain it belongs to.
                        - "type" (str): Variable type.
                    - score (Tensor): Logit mask.
                - If a list of dictionaries: Each dictionary contains:
                    - "end" (Tensor): Input state tensor.
                    - "scores": (Tensor): Input score tensor.
                    - "domain" (str): Domain it belongs to.
                    - "type" (str): Variable type.
                - In this case, scores are assumed to be `1.0`.

            best_match_only (bool): If True, only returns the transformation for the best matching type.

        Returns:
            List[Tuple[Dict, Tensor]]: Transformed contexts with updated scores.
        """
        transformed_args = []

        # Ensure input format is consistent

        if isinstance(args[0], dict):  
            args = [(context, torch.tensor(1.0, dtype=torch.float32)) for context in args]

        # Compute normalized type matching probabilities
        matching_probs = F.softmax(self.type_matching, dim=1)

        for context, score in args:
            s_type = context["type"]

            if s_type not in self.source_types:
                assert s_type in self.source_types, f"illegal types `{s_type}` not in {self.source_types}"
                continue  # Skip unknown types
            
            s_index = self.source_types.index(s_type)
            state = context["end"]

            if best_match_only and expect_type is None:

                # Only process the best matching type
                best_t_type, best_prob = self.get_best_match(s_type)
                t_index = self.target_types.index(best_t_type)

                state_mapper = self.mappers[f"{s_type}_{best_t_type}_map"]
                state_classifier = self.mappers[f"{s_type}_{best_t_type}_classify"]

                if state_mapper and state_classifier:
                    transformed_state = state_mapper(state)
                    transformed_score = torch.sigmoid(state_classifier(state)).reshape([-1,1])

                    transformed_score = torch.min(context["score"].reshape([-1,1]),transformed_score)


                    transformed_args.append((
                        {
                            "end": transformed_state,
                            "domain": self.target_domain.domain.domain_name,
                            "type": best_t_type,
                            "score" :  transformed_score
                        },
                        best_prob
                    ))
            elif expect_type:
                match_prob = self.get_match(s_type, expect_type)
                # Only process the best matching type
                best_t_type, best_prob = expect_type ,match_prob


                state_mapper = self.mappers[f"{s_type}_{best_t_type}_map"]
                state_classifier = self.mappers[f"{s_type}_{best_t_type}_classify"]

                if state_mapper and state_classifier:
                    transformed_state = state_mapper(state)
                    transformed_score = torch.sigmoid(state_classifier(state)).reshape([-1,1])

                    transformed_score = torch.min(context["score"].reshape([-1,1]),transformed_score)


                    transformed_args.append((
                        {
                            "end": transformed_state,
                            "domain": self.target_domain.domain.domain_name,
                            "type": best_t_type,
                            "score" :  transformed_score
                        },
                        best_prob
                    ))
            else:
                # Transform for all matching target types
                for t_index, t_type in enumerate(self.target_types):
                    match_prob = matching_probs[s_index, t_index].item()
                    if match_prob < 0.01:  # Skip weak mappings
                        continue

                    state_mapper = self.mappers.get(f"{s_type}_{t_type}_map")
                    state_classifier = self.mappers.get(f"{s_type}_{t_type}_classify")

                    if state_mapper and state_classifier:
                        transformed_state = state_mapper(state)
                        transformed_score = torch.min(score, torch.sigmoid(state_classifier(state)))

                        transformed_args.append((
                            {
                                "end": transformed_state,
                                "domain": self.target_domain.domain_name,
                                "type": t_type,
                                "score" : transformed_score
                            },
                            match_prob
                        ))

        return transformed_args
        
    def get_predicate_mapping(self, source_pred: str, target_pred: str) -> torch.Tensor:
        """Get mapping weight between predicates"""
        return self.predicate_matrix.get_connection_weight(source_pred, target_pred)
        
    def get_action_mapping(self, source_action: str, target_action: str) -> torch.Tensor:
        """Get mapping weight between actions"""
        return self.action_matrix.get_cnnection_weight(source_action, target_action)

class ConceptDiagram(nn.Module):
    """A directed multi-graph G=(V,E) where node set V is the set of learned domains, 
    E as the multi edge set where a pair of nodes is connected by some abstraction-mappings."""
    
    def __init__(self):
        super().__init__()
        self.device = (
            "cuda:0" if torch.cuda.is_available()
            else "mps:0" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Core structures
        self.domains = nn.ModuleDict()  # Stores learned domain executors
        self.morphisms = nn.ModuleDict()  # Stores morphisms (mapping between domains)
        self.edge_indices = defaultdict(list)  # Tracks edges in the graph

        # Probabilistic parameters (logits for differentiability)
        self.domain_logits = nn.ParameterDict()
        self.morphism_logits = nn.ParameterDict()

        # Logger
        self.logger = get_logger("concept-diagram", KFTLogFormatter)

        # Default root domain name
        self.root_name = "Generic"

        # Move to appropriate device
        self.to(self.device)
    
    def get_lexicon_entries(self):
        """should be a tuple of 
        Entry Name : as a list of lexicon entry names : [Domain]:[Predicate]
        """
        return 

    def to_dict(self):
        """Serialize the architecture (excluding weights) for reconstruction."""
        return {
            "domains": list(self.domains.keys()),  # Store domain names
            "domain_probs": {k: v.item() for k, v in self.domain_logits.items()},  # Store domain probabilities
            "morphisms": {
                name: {
                    "source": source,
                    "target": target,
                    "morphism_name": name
                }
                for (source, target), morphism_names in self.edge_indices.items()
                for name in morphism_names
            }
        }

    def tree_expansion(self, start: str):
        """Expands the concept diagram into a tree starting from a given domain.
        
        This creates a tree representation where domains are nodes and morphisms are edges.
        No edge is reused in the tree expansion to avoid cycles.
        
        Args:
            start (str): The name of the starting domain (root of the tree)
        
        Returns:
            Tuple[Dict, Dict, Dict]: A tuple containing:
                - tree: Dictionary mapping node IDs to domain names
                - edges: Dictionary mapping edge IDs to tuples of (source_id, target_id, morphism_name)
                - node_map: Dictionary mapping (domain_name, parent_domain_name, morphism_name) to node ID
        """
        if start not in self.domains:
            self.logger.warning(f"Starting domain '{start}' not found in domains")
            raise ValueError(f"Domain not found: {start}")
        
        # Initialize the tree structure
        tree = {}  # Maps node ID to domain name
        edges = {}  # Maps edge ID to (source_id, target_id, morphism_name)
        visited_edges = set()  # Set of (source, target, morphism_name) tuples that have been used
        node_map = {}  # Maps (domain_name, parent_domain_name, morphism_used) to node ID
        
        # Initialize the BFS queue with (domain_name, parent_domain_name, parent_node_id, morphism_used)
        queue = [(start, None, None, None)]
        node_count = 0
        edge_count = 0
        
        # BFS traversal to build the tree
        while queue:
            domain, parent, parent_id, morphism_used = queue.pop(0)
            
            # Create new node
            new_node_id = node_count
            node_count += 1
            
            # Store domain in tree
            tree[new_node_id] = domain
            node_map[(domain, parent, morphism_used)] = new_node_id
            
            # Add edge from parent if it exists
            if parent_id is not None and morphism_used is not None:
                edges[edge_count] = (parent_id, new_node_id, morphism_used)
                edge_count += 1
            
            # Process outgoing edges
            for (source, target), morphism_names in self.edge_indices.items():
                if source == domain:  # This domain is the source of the edge
                    # For each morphism between source and target
                    for morphism_name in morphism_names:
                        edge = (source, target, morphism_name)
                        if edge not in visited_edges:
                            visited_edges.add(edge)
                            queue.append((target, source, new_node_id, morphism_name))
        
        return tree, edges, node_map

    def visualize_tree(self, start: str = None, filename="tree_expansion"):
        """Visualizes the tree expansion using graphviz.
        
        Args:
            start (str): The name of the starting domain (root of the tree)
            filename (str): The filename to save the visualization (without extension)
            
        Returns:
            str: Path to the generated visualization file
        """
        if start is None: start = self.root_name
        try:
            import graphviz
        except ImportError:
            self.logger.warning("Graphviz is not installed. Please install it with 'pip install graphviz'")
            return None

        # Generate the tree
        tree, edges, node_map = self.tree_expansion(start)
        
        # Create graphviz Digraph
        dot = graphviz.Digraph(format="png")
        
        # Add nodes
        for node_id, domain_name in tree.items():
            dot.node(str(node_id), label=domain_name)
        
        # Add edges
        for _, (src, tgt, morphism_name) in edges.items():
            # Extract just the morphism name for cleaner labels
            edge_label = morphism_name.split('_')[-1] if '_' in morphism_name else morphism_name
            dot.edge(str(src), str(tgt), label=edge_label)
        
        # Render the graph
        try:
            dot.render(filename, cleanup=True)
            self.logger.info(f"Tree visualization saved to {filename}.png")
            return f"{filename}.png"
        except Exception as e:
            self.logger.error(f"Failed to render graph: {e}")
            return None

    def add_domain(self, name: str, domain: nn.Module, p: float = 1.0) -> None:
        if name not in self.domains:
            self.domains[name] = domain
            if p > 1.0 or p < 0.0:
                self.logger.warning(f"Input p:{p} is not within the range of [0,1]")
            self.domain_logits[name] = nn.Parameter(torch.logit(torch.ones(1) * p, eps=1e-6))
        else:
            self.logger.warning(f"try to add domain `{name}` while this name is already occupied, overriding")
            self.domains[name] = domain

    def get_domain_prob(self, name: str) -> torch.Tensor: return torch.sigmoid(self.domain_logits[name]).to(self.device)

    def add_morphism(self, source: str, target: str, morphism: nn.Module, name: Optional[str] = None) -> None:
        if source not in self.domains or target not in self.domains:
            self.logger.warning(f"domain not found: source not in domains:{source not in self.domains}, "
                         f"target not in domains: {target not in self.domains}")
            raise ValueError(f"Domain not found: {source} or {target}")
            
        if name is None:
            name = f"morphism_{source}_{target}_{len(self.edge_indices[(source, target)])}"
        #if name == "morphism_DistanceDomain_RCC8Domain_0":
            #print(morphism)
        self.morphisms[name] = morphism.to(self.device)
        self.edge_indices[(source, target)].append(name)
        self.morphism_logits[name] = nn.Parameter(torch.logit(torch.ones(1), eps=1e-6)).to(self.device)

    def get_morphism(self, source: str, target: str, index: int = 0) -> MetaphorMorphism:
        morphism_names = self.edge_indices[(source, target)]
        if not morphism_names: raise ValueError(f"No morphism found from {source} to {target}")
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

    def get_morphism_prob(self, name: str) -> torch.Tensor: return torch.sigmoid(self.morphism_logits[name]).to(self.device)

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

    def get_path(self, source: str, target: str, max_length: int = 10) -> List[List[Tuple[str, str, int]]]:
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
        """
        Compose morphisms along the given path into a single module.
        
        Args:
            path (List[Tuple[str, str, int]]): 
                A list of tuples where each tuple contains:
                - (source, target, morphism index)

        Returns:
            nn.Module: A composed module that applies the state transition along the path.
        """
        class ComposedMorphism(nn.Module):
            def __init__(self, morphisms: List[nn.Module]):
                super().__init__()
                self.morphisms = nn.ModuleList(morphisms)

            def forward(self, inputs: List[Tuple[Dict, torch.Tensor]], expect_type = None) -> List[Tuple[Dict, torch.Tensor]]:
                """
                Applies the composed transformation sequentially.

                Args:
                    inputs (List[Tuple[Dict, torch.Tensor]]): 
                        - Each element is a tuple containing:
                            - context (Dict): Dictionary with keys:
                                - "end" (Tensor): Input state tensor.
                                - "domain" (str): Domain name.
                                - "type" (str): Variable type.
                            - score (Tensor): Probability of transformation success.

                Returns:
                    List[Tuple[Dict, torch.Tensor]]: Transformed states and updated scores.
                """
                info = {"process" : [], "scores" : []}

                if isinstance(inputs[0], dict):
                    inputs = [(context, torch.ones(1, device=context["end"].device)) for context in inputs]


                #print(torch.cat([score for _, score in inputs]).shape)
                current_scores = torch.cat([score for _, score in inputs])

                #current_scores = torch.tensor([score for _, score in inputs])
                info["scores"].append(current_scores)  # Track initial scores


                for i,morphism in enumerate(self.morphisms):
                    if ((i+1) == len(self.morphisms)):

                        result = morphism(inputs, expect_type = expect_type)
                    else:

                        result = morphism(inputs)
                        
                    info["process"].append(result)

                    inputs = result
                    update_scores = []
                    for i in range(len(inputs)):
                        _, new_score = inputs[i]
                        update_scores.append( new_score ) # Chain probability update
                    update_scores = torch.stack(update_scores)
                    info["scores"].append(update_scores)

                return inputs, info

        # Retrieve the morphisms along the path
        morphisms = []
        for source, target, idx in path:
            morphism = self.get_morphism(source, target, idx)

            morphisms.append(morphism)

        return ComposedMorphism(morphisms)

    def gather(self, results):
        """ Process the results from sample_state_path to extract path probabilities, masks, and states.
        Only keeps the final tensor in each sublist of masks and states.
        
        Args:
            results (dict): The results dictionary returned by sample_state_path
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                - Path probabilities (identical to path_dist)
                - Path masks for each path, only the final tensor in each sublist
                - Path states for each path, only the final tensor in each sublist
        """
        # Extract path distribution (probabilities)
        path_probs = results["path_dist"]
        
        # Extract path masks from all elements
        all_masks = results["path_masks"]
        
        # Extract path states from all elements
        all_states = results["path_states"]
        
        # If the shape of the results doesn't match our assumptions, we'll need to restructure
        if not isinstance(all_masks, list) or not isinstance(all_states, list):
            raise ValueError("Unexpected format in results dictionary")
        
        # Keep only the final element for each sublist in masks and states
        final_masks = []
        final_states = []
        
        # Process each list of mask lists
        for path_masks in all_masks:
            path_final_masks = []
            for mask_sublist in path_masks:
                if isinstance(mask_sublist, list) and len(mask_sublist) > 0:
                    path_final_masks.append(mask_sublist[-1])  # Take the last element
                else:
                    path_final_masks.append(mask_sublist)  # If not a list or empty, keep as is
            final_masks.append(path_final_masks)
        
        # Process each list of state lists
        for path_states in all_states:
            path_final_states = []
            for state_sublist in path_states:
                if isinstance(state_sublist, list) and len(state_sublist) > 0:
                    path_final_states.append(state_sublist[-1])  # Take the last element
                else:
                    path_final_states.append(state_sublist)  # If not a list or empty, keep as is
            final_states.append(path_final_states)
        # Return the tuple of path probabilities, simplified masks, and simplified states
        return path_probs, final_masks, final_states

    def sample_state_path(self, state, source, target, expect_type = None, sample=False):
        """
        Samples a transformation path from source to target domain, considering both path probabilities
        and the success probability of transforming the specific state along each path.
        
        Args:
            state (List[Tuple[Dict, Tensor]]): A list of tuples where each tuple contains:
                - context (Dict): Dictionary with keys:
                    - "end" (Tensor): The input state tensor.
                    - "domain" (str): Domain name.
                    - "type" (str): Variable type.
                - score (Tensor): Initial probability score.
            source (str): The source domain.
            target (str): The target domain.
            sample (bool, optional): Whether to sample a path (True) or choose the max probability path (False).
            
        Returns:
            Tuple:
                - path (List[Tuple[str, str, int]]): The selected transformation path.
                - final_state (List[Tuple[Dict, Tensor]]): Transformed state along the chosen path.
                - info (Dict): Contains additional information, including:
                    - "dist" (Tensor): Probability distribution over all paths.
                    - "paths" (List): All possible paths from source to target.
                    - "path_scores" (List[Tensor]): Scores for each path.
                    - "transformation_scores" (List[Tensor]): State transformation scores for each path.
                    - "combined_scores" (List[Tensor]): Combined path and transformation scores.
                    - "visualization" (str): SVG visualization of the path selection process.
        """
        # Get all possible paths from source to target
        num_args = len(state)
        all_paths = self.get_path(source, target)
        if not all_paths:
            return None, None, None  # No valid paths exist
        
        path_modules = []
        path_scores = []
        path_probs = []
        path_masks = [[] for _ in range(num_args)]
        path_states = [[] for _ in range(num_args)]

        transformation_scores = []
        path_details = []  # Store detailed transformation info for visualization
        
        # For each possible path
        for path in all_paths:
            # Compose a transformation module for the current path
            path_module = self.compose_path(path).to(self.device)
            
            # Get the path probability based on model parameters
            path_prob = torch.exp(self.get_path_prob(path))  # Convert log probability to normal scale
            
            # Apply the module to get both the transformed state and transformation info
            transformed_state, transform_info = path_module(state, expect_type = expect_type)

            # Extract the final transformation score for this path
            # This represents how well this specific state transformed along this path

            edge_scores = transform_info["scores"][1:]

            path_scores.append(edge_scores)


            process = transform_info["process"]
           

            for idx in range(num_args):
                path_len = len(process) 
                current_arg_masks = [process[i][idx][0]["score"] for i in range(path_len)]
                current_arg_states = [process[i][idx][0]["end"] for i in range(path_len)]


                path_masks[idx].append(current_arg_masks)
                path_states[idx].append(current_arg_states)

            # Average the scores if there are multiple states

            transform_score = torch.prod(torch.stack(edge_scores), dim = 0)

            # Store the results
            path_modules.append((path, path_module, transformed_state))
            path_probs.append(path_prob)
            transformation_scores.append(transform_score)


            path_details.append({
                "path": path,
                "path_prob": path_prob.item(),
                "final_score": transform_score
            })
        
        # Convert to tensors
        path_probs = torch.stack(path_probs)
        transformation_scores = torch.stack(transformation_scores)


        # Combine path probabilities and transformation scores
        # We multiply them because both represent probabilities
        combined_scores = (path_probs * transformation_scores).permute(1,0)[0]



        # Normalize to get a valid probability distribution

        if torch.sum(combined_scores) > 0:
            path_dist = combined_scores/torch.sum(combined_scores, dim = -1, keepdim = True) #/ torch.sum(combined_scores, dim = 0)
        else:
            # Fallback to uniform distribution if all scores are zero
            path_dist = torch.ones_like(combined_scores) / len(combined_scores)

        # Choose a path based on the combined score

        if sample:
            sampled_idx = torch.multinomial(path_dist, num_samples=1).item()
        else:
            sampled_idx = torch.argmax(path_dist).item()

        sample_prob = combined_scores[sampled_idx]
        
        # Get the selected path and its final state
        sampled_path, _, final_state = path_modules[sampled_idx]
        #sampled_path, final_state = None, None

        
        # Prepare additional info
        info = {
            "dist": path_dist, 
            "paths": all_paths,
            "path_scores": path_scores,
            "path_dist": path_dist,
            "path_prob": combined_scores,
            "path_masks" : path_masks,
            "path_states" : path_states, 
            "transformation_scores": transformation_scores,
            "path_details": path_details
        }
        
        return sampled_path, sample_prob, info


    def visualize_paths(self, paths, paths_weights, masks=None, idx=0):
        """
        Creates a Graphviz visualization of the paths with weights and optional masks.
        
        Args:
            paths (List[List[Tuple[str, str, int]]]): List of paths where each path is a list of edges.
                Each edge is represented as a tuple (source, target, edge_index).
            paths_weights (List[List[str]]): List of weights for each path.
                Each weight is a string representation of a tensor shape.
            masks (List[List[Tensor]], optional): List of masks for each path.
                Each mask corresponds to an edge in the path.
            idx (int, optional): Index to use for extracting values from weights. Default is 0.
                
        Returns:
            str: Graphviz DOT string representation of the paths graph.
        """
        try:
            import graphviz
            import random
        except ImportError:
            return "Graphviz or random module not available."
        
        # Create a new directed graph
        dot = graphviz.Digraph(comment='Paths Visualization')
        
        # Set graph attributes
        dot.attr(rankdir='LR', size='10,8', fontname='Arial', label='Paths Visualization')
        
        # Set node attributes
        dot.attr('node', shape='ellipse', style='filled', fillcolor='white', 
                fontname='Arial', fontsize='12', margin='0.2')
        
        # Set edge attributes
        dot.attr('edge', fontname='Arial', fontsize='10')
        
        # Track already added nodes to avoid duplicates
        added_nodes = set()
        
        # Collect all unique nodes across all paths
        all_nodes = set()
        for path in paths:
            for src, tgt, _ in path:
                all_nodes.add(src)
                all_nodes.add(tgt)
        
        # Add all nodes to the graph
        for node in all_nodes:
            if node not in added_nodes:
                dot.node(node, node)
                added_nodes.add(node)
        
        # Function to generate weight label (simple numerical value)
        def generate_weight_label(weight_value):
            # Convert weight to float
            weight_float = float(weight_value)
            
            # Create a simple HTML label with just the weight value
            html = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            html += f'<TR><TD ALIGN="CENTER">{weight_float:.2f}</TD></TR>'
            html += '</TABLE>>'
            return html
        
        # Generate color palettes for mask patterns
        def generate_color_palette(is_gold=True):
            """Generate a palette of colors for the mask pattern based on the image example."""
            if is_gold:
                # Gold/yellow palette from top image
                return ['#7EBBC8']#, '#FFFFFF', '#D6B655', '#FFFFFF', '#D6B655', '#F5E6C4', '#D6B655']
            else:
                # Blue palette from bottom image
                return ['#2A5A66']#, '#C8D1E7', '#7D92C2', '#B8C5E2', '#95A7CF', '#FFFFFF', '#A4B6D7']
        
        # Function to generate mask visualization using the specified color patterns
        def generate_colorful_mask_label(mask_tensor, is_gold=True):
            try:
                # Convert tensor to a list of values
                if hasattr(mask_tensor, 'flatten'):
                    # PyTorch or TensorFlow tensor
                    mask_values = mask_tensor.flatten().tolist()
                else:
                    # Already a list or other iterable
                    mask_values = list(mask_tensor)
                
                # Limit number of cells to display
                max_cells = 7  # To match the palette length
                if len(mask_values) > max_cells:
                    mask_values = mask_values[:max_cells]
                
                # Get the appropriate color palette
                palette = generate_color_palette(is_gold)
                
                # Create HTML table for the mask pattern
                html = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
                
                # Add color cells as a row
                html += '<TR>'

                for i, value in enumerate(mask_values):
                    # Convert value to float between 0 and 1
                    intensity = min(max(float(value), 0), 1)
       
                    # Get color from palette
                    base_color = palette[i % len(palette)]
                    
                    # Adjust color intensity based on mask value
                    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
                    # Blend with white based on intensity (higher intensity = more vibrant color)
                    r = int(r * intensity + 255 * (1-intensity))
                    g = int(g * intensity + 255 * (1-intensity))
                    b = int(b * intensity + 255 * (1-intensity))
                    adjusted_color = f'#{r:02X}{g:02X}{b:02X}'
                    
                    # Square cell
                    html += f'<TD BGCOLOR="{adjusted_color}" WIDTH="10" HEIGHT="10"></TD>'
                html += '</TR>'
                
                # Add a row with the numerical values
                #html += '<TR>'
                #for value in mask_values:
                #    html += f'<TD ALIGN="CENTER" WIDTH="5">`{float(value):.3f}`</TD>'
                #html += '</TR>'
                
                html += '</TABLE>>'
                return html
            except Exception as e:

                # If any error occurs, return a simple text label
                return f"<<mask error>>"
        
        # Add edges for each path
        for path_idx, (path, weights) in enumerate(zip(paths, paths_weights)):
            # Get masks for this path if available
            path_masks = None

            if masks is not None and path_idx < len(masks):
                path_masks = masks[path_idx]
            
            # Use gold palette for even paths, blue for odd
            is_gold = (path_idx % 2 == 0)
            
            # Create a subgraph for this path
            with dot.subgraph(name=f'cluster_path_{path_idx}') as path_graph:
                # Process each edge in the path
                for edge_idx, ((src, tgt, _), weight) in enumerate(zip(path, weights)):
                    # Get weight value
                    weight_str = weight[idx]
                    weight_float = float(weight_str)
                    
                    # Set edge thickness based on weight
                    penwidth = str(0.5 + weight_float * 2)
                    
                    # Check if we have a mask for this edge
                    edge_mask = None

                    if path_masks is not None and edge_idx < len(path_masks):
                        edge_mask = path_masks[edge_idx]
                    
                    # Create the label - either with colorful mask or just weight
                    if edge_mask is not None:

                        # Use the colorful mask visualization with the appropriate palette
                        edge_label = generate_colorful_mask_label(edge_mask, is_gold)
                        #edge_label = generate_weight_label(weight_str)
                    else:

                        # Just use simple weight label
                        edge_label = generate_weight_label(weight_str)
                    
                    # Add the edge with the appropriate label
                    dot.edge(src, tgt, 
                            label=edge_label,
                            penwidth=penwidth,
                            fontname='Arial')
        
        # Add a legend
        legend_lines = [f"Path {i+1}: {' → '.join([f'{src}→{tgt}' for src, tgt, _ in path])}" 
                        for i, path in enumerate(paths)]
        
        dot.node('legend', label='\\n'.join(legend_lines), 
                shape='note', fontsize='10', margin='0.3')
        
        return dot.source

    def pred_domain(self, predicate):
        """
        return the domain, param types, return type 
        """
        pred_domain = None
        pred_arity = -1
        params = None
        returns = None
        for domain_name, domain_ in self.domains.items():
            for arity in domain_.predicates:
                for dom_pred in domain_.predicates[arity]:
                    if str(predicate) == str(dom_pred):
                        pred_domain = domain_name
                        pred_arity = arity
                        params = domain_.predicate_params_types[predicate]
                        returns = domain_.predicate_output_types[predicate]
                        #print(input_types)
                        #print(output_types)
                        break
        if pred_domain is None or pred_arity == -1:
            raise ValueError(f"Predicate {predicate} not found in any domain")
        return pred_domain, params, returns

    def evaluate_predicate(self, predicate: str, args):
        """ evaluate predicate name and a list of args
        Args:
            predicate : a str of the name of predicate to evaluate
            args : a list of evaluate bind context where each context is 
            - context (Dict): A dictionary with keys:
                        - "end" (Tensor): Input state tensor.
                        - "scores": (Tensor): Input score tensor.
                        - "domain" (str): Domain it belongs to.
                        - "type" (str): Variable type.
        Returns:
            the expected evaluation binding context
            - context (Dict): A dictionary with keys:
                        - "end" (Tensor): Output state tensor.
                        - "scores": (Tensor): Output score tensor.
                        - "domain" (str): Domain it belongs to.
                        - "type" (str): Output Variable type.
        """
        target_domain, arg_types, output_type = self.pred_domain(predicate)
        
        # Collect path distributions, masks, and states for each argument
        all_path_dists = []
        all_path_masks = []
        all_path_states = []
        
        for i, arg_type in enumerate(arg_types):
            sample_path, sample_prob, results = self.sample_state_path(args[i:i+1], args[0]["domain"], target_domain, arg_type)

            path_dist, masks, states = self.gather(results)

            all_path_dists.append(path_dist)
            all_path_masks.append(masks[0])
            all_path_states.append(states[0])
        
        # Handle the Cartesian product of paths using itertools.product
        path_combinations = list(itertools.product(*[range(len(dist)) for dist in all_path_dists]))
        
        final_dist = []
        final_masks = []
        final_states = []


        # Process each combination in the Cartesian product
        for combination in path_combinations:
            # Calculate combined distribution by multiplying individual probabilities
            prob = 1.0
            for arg_idx, path_idx in enumerate(combination):
                prob *= all_path_dists[arg_idx][path_idx]
            final_dist.append(prob)
            
            # Get masks for this combination
            masks_to_combine = [all_path_masks[arg_idx][path_idx] for arg_idx, path_idx in enumerate(combination)]
            
            # Handle the masks reshaping for proper broadcasting
            # For example, if we have [n1,1] and [n2,1], reshape them to [n1,1,1] and [1,n2,1]
            reshaped_masks = []
            for i, mask in enumerate(masks_to_combine):
                # Create a shape with 1s except at position i
                new_shape = [1] * len(masks_to_combine) + [1]  # +1 for the last dimension

                new_shape[i] = mask.shape[0]  # Keep original dimension at position i
                reshaped_masks.append(mask.reshape(new_shape))
            
            # Multiply the masks using broadcasting
            combined_mask = reshaped_masks[0]
            for mask in reshaped_masks[1:]:
                combined_mask = combined_mask * mask
            
            final_masks.append(combined_mask)

            
            # Get states for this combination and concatenate them on the last dimension
            states_to_combine = [all_path_states[arg_idx][path_idx] for arg_idx, path_idx in enumerate(combination)]
            reshaped_states = []
            for i, state in enumerate(states_to_combine):
                # Create a shape with 1s except at position i
                new_shape = [1] * len(states_to_combine)
                new_shape[i] = state.shape[0]  # Keep original dimension at position i
                # Keep the feature dimension as is
                new_shape.append(state.shape[1])
                reshaped_states.append(state.reshape(new_shape))
            
            # Concatenate the states along the feature dimension
            # First, broadcast to ensure compatible shapes for concatenation
            broadcasted_states = []
            for state in reshaped_states:
                # Get the broadcast shape (maximum size for each dimension)
                broadcast_shape = list(combined_mask.shape[:-1])  # Use mask shape without last dimension
                broadcast_shape.append(state.shape[-1])  # Add feature dimension
                
                # Broadcast the state to this shape
                broadcasted_state = state.expand(broadcast_shape)
                broadcasted_states.append(broadcasted_state)
            
            # Concatenate along the last dimension (feature dimension)
            combined_state = torch.cat(broadcasted_states, dim=-1)
            final_states.append(combined_state)

        final_dist = torch.stack(final_dist)
        final_states = torch.stack(final_states)
        final_masks = torch.stack(final_masks)

        """
        print(display(final_dist))
        print(display(final_states))
        print(display(final_masks))
        tensor[16]
        tensor[16, 5, 3, 256]
        tensor[16, 5, 3, 1]
        """
        executor = self.domains[target_domain]

        return {
            "end": final_states,
            "scores": final_dist,
            "domain": target_domain,
            "type": output_type
        }

    def evaluate_expression():
        return

    def traverse_state_graph(self, state : Union[List[Tuple[Dict, torch.Tensor]], List[Dict]]):
        """
        Args:
            args (Union[List[Tuple[Dict, torch.Tensor]], List[Dict]]): 
                - If a list of tuples: Each tuple contains:
                    - context (Dict): A dictionary with keys:
                        - "end" (Tensor): Input state tensor.
                        - "scores": (Tensor): Input score tensor.
                        - "domain" (str): Domain it belongs to.
                        - "type" (str): Variable type.
                    - score (Tensor): Logit mask.
                - If a list of dictionaries: Each dictionary contains:
                    - "end" (Tensor): Input state tensor.
                    - "scores": (Tensor): I nput score tensor.
                    - "domain" (str): Domain it belongs to.
                    - "type" (str): Variable type.
                - In this case, scores are assumed to be `1.0`.
        """
        return 

    def colimit(self, subgraph_nodes, subgraph_probabilities = None):
        """
        Given a probabilistic subgraph, find the colimit node as a probability distribution.
        Each edge and node has a probability of existence, and the colimit node is determined based on reachability probabilities.
        """
        if subgraph_probabilities is None:
            subgraph_probabilities = {node: 1.0 for node in subgraph_nodes}

        candidates = set(self.domains.keys()) - set(subgraph_nodes)
        colimit_probs = {node: 0 for node in candidates}
        
        edge_probabilities = {key: torch.sigmoid(self.morphism_logits[key]).item() for key in self.morphism_logits}
        node_probabilities = {key: torch.sigmoid(self.domain_logits[key]).item() for key in self.domain_logits}

        for node in candidates:
            reach_probs = []
            for n in subgraph_nodes:

                #paths = list(nx.all_simple_paths(self.to_dict(), n, node))
                paths = self.get_path(n, node)

                path_probs = [np.prod([edge_probabilities.get((u, v), 1.0) * node_probabilities.get(v, 1.0) for u, v in zip(path, path[1:])]) for path in paths]
                reach_probs.append(subgraph_probabilities.get(n, 1.0) * (1 - np.prod([1 - p for p in path_probs])))  # Probability at least one path exists
            colimit_probs[node] = np.prod(reach_probs) * node_probabilities.get(node, 1.0)  # Incorporate node probability
        
        return colimit_probs

    def batch_evaluation(self, sample_dict : Dict, eval_type = "literal"):
        """ take a diction of sample inputs and outut the evaluation of predicates of result on a batch
        TODO: This batch like operation sounds incredibly stupid, try to figure this out.
        Inputs:
            sample_dict: a diction that contains
                features : b x n x d shape tensor reprsenting the state features
                end: b x n shape tensor representing the probbaility of existence of each object
                predicates : a list of len [b] that contains predicate to evaluate at each batch
        Returns:
            outputs: a diction that contains 
                results : a list of [b] elements each representing the evaluation result on the 
                conf : a list of [b] scalars each representing the probability of that evaluation
                end : same as the outputs
        """
        features = sample_dict.get('features')  # (b, n, d)
        end = sample_dict.get('end')            # (b, n)
        predicates = sample_dict.get('predicates')
        domains = sample_dict.get("domains") if "domains" in sample_dict else None
        if features is None or end is None: raise ValueError("sample_dict must contain 'features' and 'end' keys")

        batch_size = features.shape[0]
        outputs = {
            'results': [],
            'conf': [],
            'end': end
        }

        for i in range(batch_size):
            state = features[i]           # (n, d)
            predicate = predicates[i]
            domain = domains[i] if domains is not None else domains

            results = self.evaluate(state, predicate, eval_type = eval_type)
            result = results["results"][0]
            confidence = results["probs"][0]
        
            outputs['results'].append(result)
            outputs['conf'].append(confidence)

        return outputs

    def metaphor_transform(self, state: torch.Tensor, paths, top_k: int, eps : float = 0.001, count = 10) -> torch.Tensor:
        """Metaphorical evaluation using earliest valid evaluation point by tracing predicates backwards.
        For a predicate p in target domain, we trace back through the path to find where it
        originates from (where it has strong connections to source predicates). The evaluation position is chosen
        undeterminstically controllerd by the path probability.
        """
        
        """[1]. get all the paths from the source to target domain"""

        if not paths: raise Exception(f"no path found between domain {source_domain} and {target_domain}")

        paths_of_apply = [] # each path is a sequence of appliability
        paths_of_state = [] # each path is a sequence of state (no cumulative)
        paths_of_probs = [] # each path actually applicable in the concept diagram

        """[2]. calculate each possible metaphor path if it is applicable for the init_state"""
        for path in paths[:top_k]:
            backsource_state = state # start with the working current state
            apply_path = [1.0] # maintain a sequence of apply, the first state is always applicable
            state_path = [backsource_state] # maintain a sequence of state, correspond with applicable
    
            apply_prob = 1.0 # cumulative applicable probability along a path
            for src, tgt, idx in path:
                morphism = self.get_morphism(src, tgt, idx)

                applicable_logit, transformed_state = morphism(backsource_state)
    
                backsource_state = transformed_state # iterate to the next state
                apply_prob = apply_prob * torch.sigmoid(applicable_logit) # maintain the apply_prob

                # add transoformed states and path probability according to the way
                apply_path.append(apply_prob)
                state_path.append(backsource_state)
            
            paths_of_apply.append(apply_path)
            paths_of_state.append(state_path)
            paths_of_probs.append(apply_prob * torch.exp(self.get_path_prob(path))  )# control the probability of this path actually exists.
        
        # sort the top-K metpahor paths
        sorted_indices      =  torch.argsort(torch.stack(paths_of_probs).flatten(), descending = True)
        sorted_probs        =  [paths_of_probs[i] for i in sorted_indices]
        sorted_state_paths  =  [paths_of_state[i] for i in sorted_indices]
        sorted_apply_paths  =  [paths_of_apply[i] for i in sorted_indices]
        sorted_paths        =  [all_paths[i]     for i in sorted_indices]
        """[3]. calculate the probability each predicate path is actually feasible (backward search)"""
        paths_of_symbols = [] # a symbol_path is a sequence [p0,c1,p1,...], each path controlled by the probs


        for i,path in enumerate(sorted_paths):
            target_symbol = predicate_expr.split(" ")[0][1:] # TODO: something very stupid, recall to replace by regex
            backward_path = list(reversed(path))
            symbolic_path = [target_symbol] # contains (pi,fi+1, pi+1) tuples
            
            meta_count = 0 # maximum allowed number of retract, count = 0 means the literal evaluation
            for src, tgt, idx in backward_path:
                meta_count += 1
                morph = self.get_morphism(src, tgt, idx)
                f_conn = morph.predicate_matrix #TODO: write a method that handles the Action Also
                #source_vocab = f_conn.source_predicates
                #target_vocab = f_conn.target_predicates
                #connection, reg_loss  = f_conn()
                # get probability of a pair of predicate p, p'
                source_symbol, conn = f_conn.get_best_match(target_symbol)
                if conn > eps and meta_count < count:
                    target_symbol = source_symbol
                    symbolic_path.append(conn)
                    symbolic_path.append(source_symbol)
                else:
                    break

            paths_of_symbols.append(symbolic_path)
        # the final output probability of each path is compose of two parts 1) the path is valid 2) the retreat is valid.

        """[4]. choose the most probable predicate path, executed on the cooresponding repr in metaphor path"""
        final_results = []
        final_states  = []
        final_domains = []
        final_conf    = []
        target_symbol = predicate_expr.split(" ")[0][1:]
        for i,symbol_path in enumerate(paths_of_symbols):
            retract_length = (len(symbol_path) - 1 ) // 2 # retract along the metaphor path length

            backsource_domain = sorted_paths[i][ - 1 - retract_length][1] # retract to one of the source domain
            backsource_state  = sorted_state_paths[i][ - 1 - retract_length] # retract to 
            backsource_state.to(self.device)
            source_symbol = symbol_path[-1]

            final_states.append(backsource_state)
            final_domains.append(backsource_domain)
    
            backsource_executor = self.domains[backsource_domain] # find the executor for the final state.
            assert isinstance(backsource_executor, CentralExecutor), "not an central executor"
            backsource_context = {0:{"state" : backsource_state}, 1:{"state" : backsource_state}} # create the evaluation context
            #print("Domain:",backsource_domain, "state:",backsource_state.shape, predicate_expr.replace(target_symbol, source_symbol))

            pred_result = backsource_executor.evaluate(predicate_expr.replace(target_symbol, source_symbol), backsource_context)

            dual_path_conf = sorted_apply_paths[i][ - 1 - 0] # TODO: 0 and retract_length??? consider the contribution from both parts
            for j in range(retract_length):
                dual_path_conf = dual_path_conf * symbol_path[1 + 2 * j]

            final_results.append(pred_result["end"].squeeze(-1)) # append the output diction
            final_conf.append(dual_path_conf)

        outputs = {
            "results" : final_results,
            "probs"   : final_conf,
            "states"  : final_states,
            "state_path" : sorted_state_paths,
            "apply_path" : sorted_apply_paths,
            "metas_path" : sorted_paths,
            "symbol_path": paths_of_symbols}
        return outputs

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

    def metaphorical_evaluation(self, source_state: torch.Tensor, target_predicate: str,
                                source_predicate: Optional[str] = None, source_domain : Optional[str] = None, eval_type : str = "literal",
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

    def visualize_path(self, state_path, metas_path, result = None, save_dir="outputs"):
        """
        Visualizes each state in the path using the corresponding executors.

        Args:
            state_path (list): List of states along the path.
            metas_path (list): List of tuples (source, target, morphism index) representing metaphors.
            save_dir (str): Directory to save visualized images.
        """
        os.makedirs(save_dir, exist_ok=True)
        visualizations = []

        for i, ((src_domain, tgt_domain, morphism_index), state) in enumerate(zip(metas_path[:], state_path[1:])):
            if src_domain not in self.domains or tgt_domain not in self.domains:
                print(f"Domain missing: {src_domain} or {tgt_domain}")
                continue

            # Get the executor for the target domain
            target_executor = self.domains[tgt_domain]
            assert isinstance(target_executor, CentralExecutor), "Target domain must be a CentralExecutor"

            # Create context
            state = state.cpu().detach()
            context = {0: {"state": state}, 1: {"state": state}}

            # Generate visualization
            fig, ax = plt.subplots()
            try:
                target_executor.visualize(context, result.cpu().detach())
            except:
                print(target_executor.domain.domain_name)
            ax.set_title(f"Step {i}: {src_domain} → {tgt_domain}")

            # Save image
            img_path = os.path.join(save_dir, f"path_step_{i}.png")
            plt.savefig(img_path)
            plt.close(fig)

            # Convert to base64 for inline display
            img_buffer = BytesIO()
            with open(img_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode()
            visualizations.append({"step": i, "source": src_domain, "target": tgt_domain, "image": base64_image})

        return visualizations

    def visualize_symbol_path(self, metaphor_path=None, symbol_path=None):
        """
        Visualize the concept diagram as a directed graph.

        Args:
            metaphor_path (list, optional): List of domain names to highlight
            symbol_path (list, optional): List of symbols to annotate nodes
        """
        # Create directed graph
        symbol_path = [1.] + symbol_path
        G = nx.DiGraph()

        # Add nodes with probabilities
        for domain_name in self.domains:
            prob = torch.sigmoid(self.domain_logits[domain_name]).item()
            G.add_node(domain_name, probability=prob)

        # Add edges with probabilities
        for (src, dst), idx_list in self.edge_indices.items():

            for idx in idx_list:
                #idx = idx.split("_")[-1]
                morphism_key = idx
                if morphism_key in self.morphism_logits:
                    prob = torch.sigmoid(self.morphism_logits[morphism_key]).item()

                    G.add_edge(src, dst, probability=prob, key=idx)

        # Set up the plot
        plt.figure(figsize=(12, 8))

        # Create layout (you might want to experiment with different layouts)
        pos = nx.spring_layout(G)

        # Draw nodes
        node_colors = []
        for node in G.nodes():
            base_color = 'lightblue'
            prob = G.nodes[node]['probability']
            if metaphor_path and node in metaphor_path:
                base_color = 'lightcoral'  # Highlight nodes in metaphor path
            rgba_color = to_rgba(base_color, alpha=max(0.2, prob))
            node_colors.append(rgba_color)

        nx.draw_networkx_nodes(G, pos,
                               node_color=node_colors,
                               node_size=2000)

        # Draw edges
        for (u, v, data) in G.edges(data=True):
            prob = data['probability']
            edge_color = 'gray'
            if metaphor_path and u in metaphor_path and v in metaphor_path:
                # Check if nodes are adjacent in metaphor_path
                if abs(metaphor_path.index(u) - metaphor_path.index(v)) == 1:
                    edge_color = 'red'

            nx.draw_networkx_edges(G, pos,
                                   edgelist=[(u, v)],
                                   edge_color=edge_color,
                                   alpha=max(0.2, prob),
                                   arrows=True,
                                   arrowsize=20)

        # Draw labels
        labels = {}
        for node in G.nodes():
            label = node
            if symbol_path and metaphor_path and node in metaphor_path:

                # Find position in metaphor path (from end)
                pos_from_end = len(metaphor_path) - 1 - metaphor_path.index(node)

                if pos_from_end < len(symbol_path):
                    # Only add symbols (even indices in symbol_path)
                    if pos_from_end * 2 < len(symbol_path):
                        label = f"{node}\n{symbol_path[pos_from_end * 2 + 1]} p:{float(symbol_path[pos_from_end * 2])}"
            labels[node] = label

        nx.draw_networkx_labels(G, pos, labels)

        # Add title and adjust layout
        plt.title("Concept Diagram")
        plt.axis('off')

        # Show plot
        plt.tight_layout()
        plt.show()

        return G 

    def visualize(self, save_name = "concept-diagram"):
        from graphviz import Digraph

        graph = Digraph()
        for domain in self.domains:
            graph.node(f"{domain}") # \n(p={float(self.domain_logits[domain].sigmoid()):.2f})
        for morphism in self.morphisms:
            _, source, target, idx = morphism.split("_")
            weight = round(float(self.morphism_logits[morphism].sigmoid()),2)
            if weight > 0.5:
                color = None
                width = 1.0
            else:
                color = "#3333FF80"
                width = 0.5
            graph.edge(source, target, label = str(weight), color=color, penwidth=str(width))

        graph.render(save_name, format = "png")
        return 

