'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-11-10 07:24:53
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-11-14 03:34:08
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
from typing import Dict, Set, Optional, List, Tuple, Any
from .base import *
from .types import *

class MetaphoricalExecutor(nn.Module):
    def __init__(
            self,
            base_executor: Any,
            embedding_dim: int,
            state_input_dim: int,
            state_hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_executor = base_executor
        self.domain = base_executor.domain
        self.state_dim = state_input_dim
        
        # Embeddings and projections
        self.domain_embedding = DomainEmbedding(embedding_dim)
        self.state_projector = RelationalStateProjector(
            state_input_dim,
            state_hidden_dim,
            embedding_dim
        )
        
        # Connection mappings
        self.predicate_connections: Dict[str, PredicateConnectionMatrix] = nn.ModuleDict()
        self.action_connections: Dict[str, ActionConnectionMatrix] = nn.ModuleDict()
        self.state_mappers: Dict[str, StateMapper] = nn.ModuleDict()
        self.connected_executors: Dict[str, 'MetaphoricalExecutor'] = {}

        self.evaluation_tracker = None
        
    def add_executor_connection(self, target_executor: 'MetaphoricalExecutor'):
        """Add connection to another domain executor"""
        target_name = target_executor.domain.domain_name
        connection_key = f"{self.domain.domain_name}_to_{target_name}"
        
        # Create connection matrices
        self.predicate_connections[connection_key] = PredicateConnectionMatrix(
            self.domain, target_executor.domain
        )
        self.action_connections[connection_key] = ActionConnectionMatrix(
            self.domain, target_executor.domain
        )
        
        # Create state mapper
        self.state_mappers[connection_key] = StateMapper(
            source_dim=self.state_projector.input_dim,
            target_dim=target_executor.state_projector.input_dim,
            hidden_dim= 256
        )
        
        self.connected_executors[target_name] = target_executor
    
    def get_state_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get embedded representation of state"""
        return self.state_projector(state)
    
    def get_target_executor(self, target_name: str) -> 'MetaphoricalExecutor':
        """Get connected executor by domain name"""
        if target_name not in self.connected_executors:
            raise ValueError(f"No connection found to domain {target_name}")
        return self.connected_executors[target_name]

    def _initialize_evaluation(self, tracker=None):
        """Initialize or set evaluation tracker"""
        if tracker is not None:
            self.evaluation_tracker = tracker
        elif self.evaluation_tracker is None:
            from ..utils import EvaluationGraphTracker
            self.evaluation_tracker = EvaluationGraphTracker()

    def _parse_expression(self, expr: str) -> Tuple[str, int]:
        """Parse expression into operation and state index"""
        parts = expr.strip('()').split()
        op_name = parts[0]
        state_idx = int(parts[1].strip('$'))
        return op_name, state_idx, parts

    def _evaluate_on_target(
            self,
            expr: str,
            context: Dict[int, Dict[str, torch.Tensor]],
            scene : bool,
            target_key: str,
            op_name: str,
            parts: List[str],
            state_embedding: torch.Tensor,
            mode: str = 'expectation'
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Evaluate expression on a target domain with sanity checks"""
        target_name = target_key.split('_to_')[1]
        target_executor = self.get_target_executor(target_name)
        
        # Get components
        pred_matrix = self.predicate_connections[target_key]
        state_mapper = self.state_mappers[target_key]
        
        # Map states
        target_context = {
            idx: {
                "state": state_mapper(context[idx]['state']),
                "end": context[idx]["end"]
            }
            for idx in context
        }
        
        # Calculate connection probability
        connection_prob = calculate_state_domain_connection(
            state_embedding,
            target_executor.domain_embedding()
        )
        connection_prob = 1.0
        
        # Initialize results
        total_prob = 0.0
        probs = []
        measures = []
        names = []
        
        # Handle predicates
        if op_name in self.domain.predicates:
            valid_weights = []
            valid_predicates = []
            
            # First pass: collect all valid connections
            for target_pred in target_executor.domain.predicates:
                weight = pred_matrix.get_connection_weight(op_name, target_pred)
                if weight > 0.0:  # Only consider meaningful connections
                    valid_weights.append(weight)
                    valid_predicates.append(target_pred)

            # Second pass: evaluate only if valid connections exist
            if valid_predicates:
                for weight, target_pred in zip(valid_weights, valid_predicates):
                    prob = connection_prob * weight
                    target_expr = f"({target_pred} {' '.join(parts[1:])})"
                    
                    self.evaluation_tracker.add_evaluation_step(
                        source_domain=self.domain.domain_name,
                        source_expr=expr,
                        target_domain=target_name,
                        target_expr=target_expr,
                        weight=prob.item()
                    )
                    
                    curr_result = target_executor.evaluate(
                        target_expr,
                        target_context,
                        scene = scene,
                        mode=mode,
                        tracker=self.evaluation_tracker
                    )
                    
                    if weight > 0.0:
                        total_prob += prob
                        probs.append(prob)
                        measures.append(curr_result["end"])
                        names.append(target_pred)
        
        # Handle actions similar to predicates...
        elif op_name in self.domain.actions:
            # ... (action handling code remains the same)
            pass

        # Return results based on mode
        infos = {
            "context": target_context,
            "probs": probs,
            "measures": measures,
            "names": names,
        }
        
        if total_prob == 0:
            return {"end": 0.0}, 0.0, infos
        
        if mode == 'expectation':
            # Calculate weighted average
            sumup_measure = 0.0
            total_prob = sum(probs)
            for i in range(len(names)):
                sumup_measure = sumup_measure + probs[i] / total_prob * measures[i]
        elif mode == 'maximum':
            # Find the index of maximum probability
            max_idx = max(range(len(probs)), key=lambda i: probs[i])
            sumup_measure = measures[max_idx]
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        results = {"end": sumup_measure}
        return results, total_prob, infos

    def evaluate(
            self,
            expr: str,
            context: Dict[int, Dict[str, torch.Tensor]],
            scene = False,
            mode: str = 'expectation',
            visited_executors: Optional[Set[str]] = None,
            tracker = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate expression across all connected domains
        
        Args:
            expr: Expression to evaluate
            context: Evaluation context
            mode: Either 'expectation' (weighted average) or 'maximum' (highest probability)
            visited_executors: Set of already visited executors
            tracker: Evaluation tracker
        """
        self._initialize_evaluation(tracker)
        
        if visited_executors is None:
            visited_executors = set()
        visited_executors.add(self.domain.domain_name)
        
        # Parse expression
        op_name, state_idx, parts = self._parse_expression(expr)
        state = context[state_idx]['state']
        state_embedding = self.get_state_embedding(state)
        
        # Get base evaluation
        base_result = self.base_executor.evaluate(expr, context, scene = scene)
        self.evaluation_tracker.add_evaluation_step(
            source_domain=self.domain.domain_name,
            source_expr=expr,
            result=base_result
        )
        
        # Initialize results and tracking lists
        results = base_result.copy()
        probs_spectrum = [1.0]
        measure_spectrum = [results["end"]]
        
        # Evaluate on all connected domains
        for target_key in self.predicate_connections:
            target_name = target_key.split('_to_')[1]
            if target_name in visited_executors:
                continue
                
            target_results, target_prob, target_context = self._evaluate_on_target(
                expr, context, scene, target_key, op_name, parts,
                state_embedding, mode
            )
            
            if target_prob > 0:
                if mode == 'expectation':
                    results["end"] = results["end"] + target_results["end"]
                elif mode == 'maximum':
                    # Compare with current maximum
                    if target_prob > max(probs_spectrum):
                        results["end"] = target_results["end"]
                        probs_spectrum = [target_prob]
                        measure_spectrum = [target_results["end"]]
                        continue
                
                probs_spectrum.append(target_prob)
                measure_spectrum.append(target_results["end"])
        

        if mode == 'expectation':
            results["end"] = results["end"] / sum(probs_spectrum)
        
        return results

    def evaluate_on_domain(
            self,
            expr: str,
            context: Dict[int, Dict[str, torch.Tensor]],
            target_domain: str,
            mode: str = 'expectation',
            tracker = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Evaluate expression on base domain and specific target domain"""
        self._initialize_evaluation(tracker)
        
        # Parse expression
        op_name, state_idx, parts = self._parse_expression(expr)
        state = context[state_idx]['state']
        state_embedding = self.get_state_embedding(state)
        
        # Get base evaluation
        base_result = self.base_executor.evaluate(expr, context)
        self.evaluation_tracker.add_evaluation_step(
            source_domain=self.domain.domain_name,
            source_expr=expr,
            result=base_result
        )
        
        # Check target domain exists
        target_key = f"{self.domain.domain_name}_to_{target_domain}"
        if target_key not in self.predicate_connections:
            raise ValueError(f"No connection found to domain {target_domain}")
        
        # Evaluate on target domain
        target_results, total_prob, target_context = self._evaluate_on_target(
            expr, context, target_key, op_name, parts,
            state_embedding, mode
        )
        
        # Normalize target results
        
        if total_prob > 0:
            for k in target_results:
                target_results[k] = target_results[k] / total_prob
                
        return base_result, target_results, target_context
    
    def add_cues(self, cues: List[Tuple[str, str, str]], value: float = 1.0):
        """
        Add metaphorical cues by forcing specific connection weights.
        
        Args:
            cues: List of tuples (domain_name, source_item, target_item)
            value: The connection weight value to set (default: 1.0)
        """
        for domain_name, source_item, target_item in cues:
            connection_key = f"{self.domain.domain_name}_to_{domain_name}"
            
            # Check if connection exists
            if connection_key not in self.predicate_connections:
                raise ValueError(f"No connection found to domain {domain_name}")
                
            # Try setting in predicate matrix
            if source_item in self.domain.predicates:
                self.predicate_connections[connection_key].set_connection_weight(
                    source_item, target_item, value
                )
                
            # Try setting in action matrix
            elif source_item in self.domain.actions:
                self.action_connections[connection_key].set_connection_weight(
                    source_item, target_item, value
                )
            else:
                raise ValueError(f"Source item {source_item} not found in domain {self.domain.domain_name}")