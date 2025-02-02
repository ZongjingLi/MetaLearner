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
        return self.action_matrix.get_cnnection_weight(source_action, target_action)





class ConceptDiagram(nn.Module):
    """A directed multi-graph G=(V,E) where node set V is the set of learned domains, 
    E as the multi edge set where a pair of nodes is connected by some abstraction-mappings."""
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.domains = nn.ModuleDict()  # Store domains (CentralExecutor instances)
        self.morphisms = nn.ModuleDict()  # Store morphisms (sparse connections)
        self.edge_indices = defaultdict(list)
        self.domain_logits = nn.ParameterDict()  # Store log p for domains
        self.morphism_logits = nn.ParameterDict()  # Store log p for morphisms
        self.logger = get_logger("concept-diagram", KFTLogFormatter)
        self.root_name = "Generic"

    def add_domain(self, name: str, domain: nn.Module, p: float = 1.0) -> None:
        if name not in self.domains:
            self.domains[name] = domain
            if p > 1.0 or p < 0.0:
                self.logger.warning(f"Input p:{p} is not within the range of [0,1]")
            self.domain_logits[name] = nn.Parameter(torch.logit(torch.ones(1) * p, eps=1e-6))
        else:
            self.logger.warning(f"try to add domain `{name}` while this name is already occupied, overriding")
            self.domains[name] = domain

    def add_morphism(self, source: str, target: str, morphism: nn.Module, 
                    name: Optional[str] = None) -> None:
        if source not in self.domains or target not in self.domains:
            self.logger.warning(f"domain not found: source not in domains:{source not in self.domains}, "
                         f"target not in domains: {target not in self.domains}")
            raise ValueError(f"Domain not found: {source} or {target}")
            
        if name is None:
            name = f"morphism_{source}_{target}_{len(self.edge_indices[(source, target)])}"
        #if name == "morphism_DistanceDomain_RCC8Domain_0":
            #print(morphism)
        self.morphisms[name] = morphism
        self.edge_indices[(source, target)].append(name)
        self.morphism_logits[name] = nn.Parameter(torch.logit(torch.ones(1), eps=1e-6))

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

    def get_domain_prob(self, name: str) -> torch.Tensor:
        return torch.sigmoid(self.domain_logits[name])

    def get_morphism_prob(self, name: str) -> torch.Tensor:
        return torch.sigmoid(self.morphism_logits[name])

    def evaluate(self, state: torch.Tensor, predicate: str, domain: str = None, 
                eval_type: str = 'literal', top_k: int = 5, count = 10) -> Dict:
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
        if domain is None: domain = "GenericDomain"
        
        if pred_arity == 0:
            predicate = f"({predicate})"
        if pred_arity == 1:
            predicate = f"({predicate} $0)"
        if pred_arity == 2:
            predicate = f"({predicate} $0 $1)"

        # Choose evaluation method
        if eval_type == 'literal':
            return self._evaluate_metaphor(state, predicate, domain, pred_domain, top_k, count = 0)
        elif eval_type == 'metaphor':
            return self._evaluate_metaphor(state, predicate, domain, pred_domain, top_k, count = count)
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

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

    def _evaluate_metaphor(self, state: torch.Tensor, predicate_expr: str,
                          source_domain: str, target_domain: str, top_k: int, eps : float = 0.001, count = 10) -> torch.Tensor:
        """Metaphorical evaluation using earliest valid evaluation point by tracing predicates backwards.
        For a predicate p in target domain, we trace back through the path to find where it
        originates from (where it has strong connections to source predicates). The evaluation position is chosen
        undeterminstically controllerd by the path probability.
        """
        
        """[1]. get all the paths from the source to target domain"""
        all_paths = self.get_path(source_domain, target_domain)

        if not all_paths: raise Exception(f"no path found between domain {source_domain} and {target_domain}")

        paths_of_apply = [] # each path is a sequence of appliability
        paths_of_state = [] # each path is a sequence of state (no cumulative)
        paths_of_probs = [] # each path actually exist in the concept diagram

        """[2]. calculate each possible metaphor path if it is applicable for the init_state"""
        
        for path in all_paths[:top_k]:
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

    def get_path(self, 
                 source: str, 
                 target: str, 
                 max_length: int = 10) -> List[List[Tuple[str, str, int]]]:
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
