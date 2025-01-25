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
from io import BytesIO
import base64

from rinarak.logger import get_logger, KFTLogFormatter
from rinarak.logger import set_logger_output_file

from rinarak.domain import load_domain_string
from rinarak.knowledge.executor import CentralExecutor

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from core.metaphors.base import StateMapper, StateClassifier
from core.metaphors.legacy import PredicateConnectionMatrix, ActionConnectionMatrix
from rinarak.utils.data import combine_dict_lists

logger = get_logger("Citadel", KFTLogFormatter)
set_logger_output_file("logs/citadel_logs.txt")

device = "mps" if torch.backends.mps.is_available() else "cpu"

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
        #print(source_domain.domain.domain_name,source_domain.state_dim[0])
        self.state_checker = StateClassifier(
            source_dim = source_domain.state_dim[0],
            latent_dim = hidden_dim,
            hidden_dim = hidden_dim
        )

        
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
        """Map state from source to target domain
        Inputs:
            state : should be a 
        """
        return self.state_checker.compute_logit(state), self.state_mapper(state)
        
    def get_predicate_mapping(self, source_pred: str, target_pred: str) -> torch.Tensor:
        """Get mapping weight between predicates"""
        return self.predicate_matrix.get_connection_weight(source_pred, target_pred)
        
    def get_action_mapping(self, source_action: str, target_action: str) -> torch.Tensor:
        """Get mapping weight between actions"""
        return self.action_matrix.get_connection_weight(source_action, target_action)



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
        #if name == "morphism_DistanceDomain_RCC8Domain_0":
            #print(morphism)
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
                eval_type: str = 'literal', top_k: int = 3) -> torch.Tensor:
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
        if eval_type == 'literal':
            return self._evaluate_literal(state, predicate, domain, pred_domain, top_k)
        elif eval_type == 'metaphor':
            return self._evaluate_metaphor(state, predicate, domain, pred_domain, top_k)
        elif eval_type == 'prob_metaphor':
            return self._evaluate_prob_metaphor(state, predicate, domain, pred_domain, top_k)
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

    def _evaluate_literal(self, state: torch.Tensor, predicate_expr: str, 
                        source_domain: str, target_domain: str, top_k: int) -> torch.Tensor:
        """literal evaluation following all paths."""

        """1. get all the paths from the source to target domain"""
        all_paths = self.get_path(source_domain, target_domain)
        if not all_paths:
            raise Exception(f"no path found between domain {source_domain} and {target_domain}")

        path_of_apply = []
        path_of_state = []
        
        path_probs = []
        results = []
       
        for path in all_paths[:top_k]:
            """initalize the state, apply, result paths along the way"""
            current_state = state
            apply_path = [1.0]
            state_path = [current_state]

            
            path_prob = torch.exp(self.get_path_prob(path)) # control the probability of this path actually exists.
            apply_prob = 1.0
            """2. calculate the probability of this path exists"""

            for src, tgt, idx in path:
                morphism = self.get_morphism(src, tgt, idx)

                applicable_prob, transformed_state = morphism(current_state)

                current_state = transformed_state
                apply_prob = apply_prob * torch.sigmoid(applicable_prob)

                # add transoformed states and path probability according to the way
                apply_path.append(apply_prob)
                state_path.append(current_state)


            target_executor = self.domains[target_domain] # find the executor for the final state.
            assert isinstance(target_executor, CentralExecutor), "not an central executor"
    
            context = {0:{"state" : current_state}, 1:{"state" : current_state}} # create the evaluation context
            pred_result = target_executor.evaluate(predicate_expr, context)

            path_of_apply.append(apply_path)
            path_of_state.append(state_path)

            path_probs.append(apply_prob * path_prob)
            results.append(pred_result["end"])

        path_probs = torch.stack(path_probs)
        results = torch.stack(results)

        return {"results" : results, "probs" : path_probs, "state_path" : path_of_state, "apply_path" : path_of_apply, "metas_path" : all_paths}
    
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
            target_executor.visualize(context, result.cpu().detach())

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
    curve_executor.to(device)
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
    concept_diagram.add_morphism("DistanceDomain", "RCC8Domain", MetaphorMorphism(distance_executor, rcc8_executor))

    concept_diagram.add_morphism("GenericDomain", "CurveDomain", MetaphorMorphism(generic_executor, curve_executor))
    

    path = concept_diagram.get_path("GenericDomain", "RCC8Domain")[0]  # Get first path

    line_executor.domain.get_summary()

    concept_diagram.to(device)

    """generic state space testing"""
    source_state = torch.randn([5, 256]).to(device)
    context = {
        0 : {"state" : source_state},
        1 : {"state" : source_state}
        
    }
    #print(concept_diagram.get_morphism('DistanceDomain', 'RCC8Domain', 0))

    evaluation_result = concept_diagram.evaluate(source_state, "parallel_to", "GenericDomain", eval_type = "literal")
    apply_path = evaluation_result["apply_path"][0] # [1.0, tensor([0.5170], device='mps:0', grad_fn=<MulBackward0>), tensor([0.2620], device='mps:0', grad_fn=<MulBackward0>)]
    state_path = evaluation_result["state_path"][0]
    metas_path = evaluation_result["metas_path"][0] # [('GenericDomain', 'DistanceDomain', 0), ('DistanceDomain', 'DirectionDomain', 0)]

    visualizations = concept_diagram.visualize_path(state_path, metas_path, evaluation_result["results"][0].cpu().detach())

    # Example: Display the first visualization

    direction_executor.visualize(context, evaluation_result["results"][0])
    print((evaluation_result["results"][0] + 0.5).int())
    plt.savefig("outputs/save1.png")



"""
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
"""