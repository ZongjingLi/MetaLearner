from typing import List, Tuple, Dict, Set, Union, Optional, Any
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

class SparseWeightedGraph(nn.Module):

    def __init__(self, node_labels: List[str], edge_list: List[Tuple[Union[str, int], Union[str, int]]]):
        """ a sparse weighted graph with learnable edge weights.
        Args:
            node_labels: List of strings representing node labels
            edge_list: List of tuples (i, j) representing edges from node i to node j
        """
        super(SparseWeightedGraph, self).__init__()
        
        self.node_labels: List[str] = node_labels
        self.num_nodes: int = len(node_labels)
        self.node_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(node_labels)}
        
        # convert string labels to indices
        self.edge_indices: List[Tuple[int, int]] = []
        for i, j in edge_list:
            if isinstance(i, str): i = self.node_to_idx[i]
            if isinstance(j, str): j = self.node_to_idx[j]
            self.edge_indices.append((i, j))
        
        """ create learnable weights for each edge """
        self.edge_weights: nn.Parameter = nn.Parameter(torch.randn(len(self.edge_indices)))
        
        """ create sparse adjacency matrix indices """
        self.sparse_indices: torch.LongTensor = torch.LongTensor([
            [i for i, j in self.edge_indices], 
            [j for i, j in self.edge_indices]
        ])
    def add_node(self, node_label: str) -> bool:
        """ Add a new node to the graph.
        Args:
            node_label: Label for the new node
        Returns:
            True if the node was added, False if it already exists
        """
        
        if node_label in self.node_to_idx: return False # the node should not already exists
        
        """create the new node in the node labels and append the index"""
        self.node_labels.append(node_label)
        self.node_to_idx[node_label] = self.num_nodes
        self.num_nodes += 1
        
        return True

    def add_edge(self, node1: Union[str, int], node2: Union[str, int], 
                initial_weight: Optional[float] = None) -> bool:
        """ Add a new edge between two nodes
        Args:
            node1: Source node (label or index)
            node2: Target node (label or index)
            initial_weight: Initial weight value (before sigmoid). 
                           If None, uses a small random value.        
        Returns:
            True if the edge was added, False if it already exists or nodes don't exist
        """
        # Convert string labels to indices
        if isinstance(node1, str):
            if node1 not in self.node_to_idx:
                return False
            node1 = self.node_to_idx[node1]
        
        if isinstance(node2, str):
            if node2 not in self.node_to_idx:
                return False
            node2 = self.node_to_idx[node2]
        
        # Check if nodes exist
        if node1 >= self.num_nodes or node2 >= self.num_nodes:
            return False
        
        # Check if edge already exists
        for i, j in self.edge_indices:
            if i == node1 and j == node2:
                return False
        
        # Add the new edge
        self.edge_indices.append((node1, node2))
        
        # Create a new parameter tensor with one more element
        new_weights = torch.empty(len(self.edge_indices), device=self.edge_weights.device)
        new_weights[:-1] = self.edge_weights.data
        
        # Set the weight for the new edge
        if initial_weight is not None:
            # Use inverse sigmoid to convert from probability to weight
            if 0 < initial_weight < 1:
                import math
                initial_weight = math.log(initial_weight / (1 - initial_weight))
            new_weights[-1] = initial_weight
        else:
            # Small random value
            new_weights[-1] = torch.randn(1).item() * 0.1
        
        # Replace the parameter
        self.edge_weights = nn.Parameter(new_weights)
        
        # Update sparse indices
        self.sparse_indices = torch.LongTensor([
            [i for i, j in self.edge_indices], 
            [j for i, j in self.edge_indices]
        ])
        
        return True    

    def get_adjacency_matrix(self) -> torch.sparse.FloatTensor:
        """Returns the sparse adjacency matrix with learned weights"""
        adj_matrix = torch.sparse.FloatTensor(
            self.sparse_indices, 
            self.edge_weights,
            torch.Size([self.num_nodes, self.num_nodes])
        )
        return adj_matrix
    
    def get_edge_weight(self, node1: Union[str, int], node2: Union[str, int]) -> Optional[float]:
        """Get the current weight between two nodes"""
        if isinstance(node1, str): node1 = self.node_to_idx[node1]
        if isinstance(node2, str): node2 = self.node_to_idx[node2]
            
        for idx, (i, j) in enumerate(self.edge_indices):
            if i == node1 and j == node2:
                return self.edge_weights[idx].item()
        return None  # Edge doesn't exist

    def reachable_nodes_with_probability(self, query: Union[str, int]) -> Dict[str, float]:
        """ Finds all nodes reachable from the query node along with their reachability probabilities.
        Args:
            query: The query node label or index.
        Returns:
            Dictionary mapping node labels to reachability probabilities
        """
        # Handle string labels
        if isinstance(query, str): 
            query = self.node_to_idx[query]

        # Initialize reachability probabilities
        reach_prob: torch.Tensor = torch.zeros(self.num_nodes, device=self.edge_weights.device)
        reach_prob[query] = 1.0  # Query node has reachability probability 1

        # BFS-like traversal to compute reachability probabilities
        visited: Set[int] = set()
        queue: List[Tuple[int, float]] = [(query, 1.0)]  # (current_node, accumulated_prob)

        while queue:
            current, prob = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)

            # Traverse neighbors
            for idx, (src, dst) in enumerate(self.edge_indices):
                if src == current:
                    weight = torch.sigmoid(self.edge_weights[idx]).item()
                    new_prob = prob * weight

                    if new_prob > reach_prob[dst]:
                        reach_prob[dst] = new_prob
                        queue.append((dst, new_prob))

        # Convert back to node labels and return
        reachable_nodes: Dict[str, float] = {
            self.node_labels[i]: reach_prob[i].item()
            for i in range(self.num_nodes) if reach_prob[i].item() > 0
        }

        return reachable_nodes
        
    def find_most_far_reaching_nodes(self, query: Union[str, int]) -> Dict[str, Any]:
        """
        Finds all nodes reachable from the query node and determines which ones are
        the most far-reaching nodes, including the query node itself.
        
        A node is considered far-reaching if it has a high probability of being 
        reachable from the query node (p1) and a low probability of reaching any 
        other node from the reachable set (p2). The combined score is p1 * (1 - p2).
        
        For the query node, p1 = 1.0 and we calculate p2 as the probability it 
        can reach any other node.
        
        Args:
            query: The query node label or index.
            
        Returns:
            Dictionary with three items:
                'reachable_nodes': {node_label: reachability_probability}
                'far_reaching_scores': {node_label: far_reaching_score}
                'most_far_reaching': list of node labels with highest far-reaching scores
        """
        # Convert string query to index if needed
        if isinstance(query, str):
            query = self.node_to_idx[query]
        
        # Step 1: Calculate p1 - probability of reaching each node from query node
        p1_probs: List[float] = self._calculate_reachability_probs(query)
        
        # Get set of reachable nodes (indices)
        reachable_indices: List[int] = [i for i in range(self.num_nodes) if p1_probs[i] > 0]
        
        # Step 2: For each reachable node, calculate p2 - probability it can reach other nodes
        p2_probs: Dict[int, float] = {}
        for node_idx in reachable_indices:
            # Calculate reachability probabilities from this node
            node_reach_probs: List[float] = self._calculate_reachability_probs(node_idx)
            
            # Calculate probability this node can reach any other reachable node
            # We only consider nodes that are reachable from the original query
            other_reachable: List[int] = [i for i in reachable_indices if i != node_idx]
            
            if not other_reachable:  # No other reachable nodes
                p2_probs[node_idx] = 0.0
                continue
                
            # Calculate probability of reaching at least one other node
            # P(reach at least one) = 1 - P(reach none) = 1 - ∏(1 - P(reach i))
            prob_reach_none: float = 1.0
            for i in other_reachable:
                prob_reach_none *= (1.0 - node_reach_probs[i])
                
            p2_probs[node_idx] = 1.0 - prob_reach_none
        
        # Step 3: Calculate far-reaching scores as p1 * (1 - p2)
        far_reaching_scores: Dict[int, float] = {}
        for node_idx in reachable_indices:
            p1: float = p1_probs[node_idx]
            p2: float = p2_probs.get(node_idx, 0.0)
            far_reaching_scores[node_idx] = p1 * (1.0 - p2)
        
        # Find the nodes with the highest far-reaching scores
        if far_reaching_scores:
            max_score: float = max(far_reaching_scores.values())
            most_far_reaching_indices: List[int] = [
                idx for idx, score in far_reaching_scores.items() 
                if score == max_score
            ]
        else:
            most_far_reaching_indices = []
        
        # Convert results back to node labels
        result: Dict[str, Any] = {
            'reachable_nodes': {self.node_labels[i]: p1_probs[i] for i in reachable_indices},
            'far_reaching_scores': {self.node_labels[i]: far_reaching_scores[i] 
                                   for i in far_reaching_scores},
            'most_far_reaching': [self.node_labels[i] for i in most_far_reaching_indices]
        }
        
        return result

    def _calculate_reachability_probs(self, start_node: int) -> List[float]:
        """
        Helper method to calculate reachability probabilities from a start node to all other nodes.
        
        Args:
            start_node: Index of the starting node
            
        Returns:
            List of probabilities, where index i contains probability of reaching node i
        """
        # Initialize reachability probabilities
        reach_probs: List[float] = [0.0] * self.num_nodes
        reach_probs[start_node] = 1.0  # Start node has reachability probability 1
        
        # BFS-like traversal to compute reachability probabilities
        visited: Set[int] = set()
        queue: List[Tuple[int, float]] = [(start_node, 1.0)]  # (current_node, accumulated_prob)
        
        while queue:
            current, prob = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Traverse neighbors
            for idx, (src, dst) in enumerate(self.edge_indices):
                if src == current:
                    # Get edge weight as a probability using sigmoid
                    weight: float = torch.sigmoid(self.edge_weights[idx]).item()
                    new_prob: float = prob * weight
                    
                    # Update probability if higher
                    if new_prob > reach_probs[dst]:
                        reach_probs[dst] = new_prob
                        queue.append((dst, new_prob))
        
        return reach_probs
        
    def visualize_graph(self, 
                       query: Optional[Union[str, int]] = None, 
                       far_reaching_results: Optional[Dict[str, Any]] = None, 
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Visualize the graph with optional highlighting of query and far-reaching nodes.
        
        Args:
            query: The query node to highlight
            far_reaching_results: Results from find_most_far_reaching_nodes
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib figure: The visualization figure
        """
        # Create a networkx graph
        G = nx.DiGraph()
        
        # Add nodes with labels
        for i, label in enumerate(self.node_labels):
            G.add_node(i, label=label)
        
        # Add edges with weights
        for idx, (i, j) in enumerate(self.edge_indices):
            weight = torch.sigmoid(self.edge_weights[idx]).item()
            G.add_edge(i, j, weight=weight, width=weight*3)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=42)
        
        # Default node colors and sizes
        node_colors: List[str] = ['skyblue'] * self.num_nodes
        node_sizes: List[int] = [500] * self.num_nodes
        
        # Highlight query node and far-reaching nodes if provided
        if query is not None:
            if isinstance(query, str):
                query = self.node_to_idx[query]
            node_colors[query] = 'green'
            node_sizes[query] = 700
            
            if far_reaching_results is not None:
                for node_label in far_reaching_results['most_far_reaching']:
                    node_idx = self.node_to_idx[node_label]
                    node_colors[node_idx] = 'red'
                    node_sizes[node_idx] = 700
        
        # Draw nodes with custom size and color
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes)
        
        # Draw edges with varying width based on weight
        edge_widths = [G[u][v]['width'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                              edge_color='gray', arrowsize=15)
        
        # Draw node labels
        labels = {i: data['label'] for i, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        # Draw edge weights
        edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add title
        if query is not None:
            query_label = self.node_labels[query]
            plt.title(f"Graph with query node '{query_label}'")
        else:
            plt.title("Graph Visualization")
            
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
        
    def visualize_far_reaching_analysis(self, query: Union[str, int]) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Perform far-reaching analysis and visualize the results.
        
        Args:
            query: The query node
            
        Returns:
            Tuple containing (matplotlib figure, far_reaching_results)
        """
        # Perform the analysis
        results: Dict[str, Any] = self.find_most_far_reaching_nodes(query)
        
        # Visualize the graph with highlighted nodes
        fig: plt.Figure = self.visualize_graph(query, results)
        
        return fig, results
    

from .types import TypeCaster

class ReductiveUnifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.casters = {}

    def register_func(self, name, args_type, out_type):
        return

    def register_func_caster(self, func1_name, func2_name):
        return

    def reducting_evaluation(self, args):
        return
    
    def evaluate(self, executer):
        return