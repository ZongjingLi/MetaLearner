import numpy as np
import matplotlib.pyplot as plt
import graphviz
from scipy.special import softmax
import torch


def generate_random_matrix(n_nodes):
    """
    Generate a random matrix representing potential graph edges
    
    Parameters:
    - n_nodes: number of nodes in the graph
    
    Returns:
    - matrix: a random matrix with values representing edge probabilities
    """
    # Create a random matrix
    matrix = np.random.rand(n_nodes, n_nodes)
    
    # Zero out the diagonal (no self-loops)
    np.fill_diagonal(matrix, 0)
    
    return matrix

def normalize_weights(matrix):
    """
    Normalize the weights using softmax function
    """
    return torch.sigmoid(torch.tensor(matrix))
    # Apply softmax row-wise to get probability distributions
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        # Skip rows that are all zeros
        if np.sum(matrix[i]) > 0:
            normalized_matrix[i] = softmax(matrix[i])
    
    return normalized_matrix

def construct_maximum_probability_graph(matrix, threshold=0.3, normalize=True):
    """
    Construct a graph by keeping edges with probabilities above a threshold
    
    Parameters:
    - matrix: edge probability matrix
    - threshold: minimum probability threshold for including an edge
    - normalize: whether to normalize the matrix first
    
    Returns:
    - graph_matrix: adjacency matrix of the maximum probability graph
    """
    n_nodes = matrix.shape[0]
    
    if normalize:
        # Normalize the matrix row-wise
        prob_matrix = normalize_weights(matrix)
    else:
        prob_matrix = matrix.copy()
    
    # Initialize an empty graph
    graph_matrix = np.zeros_like(prob_matrix)
    
    # Keep edges with probabilities above the threshold
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and prob_matrix[i][j] > threshold:
                graph_matrix[i][j] = prob_matrix[i][j]
    
    # Ensure the graph is connected
    # If not connected, add high probability edges until it is
    connected = is_connected(graph_matrix)
    if not connected:
        # Sort all edges by probability
        edges = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and graph_matrix[i][j] == 0:  # Only consider edges not already in the graph
                    edges.append((i, j, prob_matrix[i][j]))
        
        # Sort by probability, highest first
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # Add edges until connected or no more edges
        for i, j, prob in edges:
            graph_matrix[i][j] = prob
            if is_connected(graph_matrix):
                break

    return graph_matrix

def is_connected(matrix):
    """
    Check if a graph represented by an adjacency matrix is connected
    
    Parameters:
    - matrix: adjacency matrix
    
    Returns:
    - connected: boolean indicating if the graph is connected
    """
    n_nodes = matrix.shape[0]
    
    # Convert to a binary adjacency matrix
    binary_matrix = (matrix > 0).astype(int)
    
    # Start DFS from node 0
    visited = set()
    
    def dfs(node):
        visited.add(node)
        for next_node in range(n_nodes):
            if binary_matrix[node][next_node] > 0 and next_node not in visited:
                dfs(next_node)
    
    dfs(0)
    
    # Check if all nodes are visited
    return len(visited) == n_nodes

def visualize_graph(matrix):
    """
    Visualize the graph using Graphviz
    
    Parameters:
    - matrix: graph adjacency matrix with edge probabilities
    
    Returns:
    - graphviz.Digraph object
    """
    n_nodes = matrix.shape[0]
    # Create a Graphviz graph
    dot = graphviz.Digraph(comment='Maximum Probability Graph')
    
    # Add nodes
    for i in range(n_nodes):
        dot.node(str(i), f"{i}")
    
    # Add edges with weights
    for i in range(n_nodes):
        for j in range(n_nodes):
            weight = matrix[i][j]
            if weight > 0:
                dot.edge(str(i), str(j), label=f"{weight:.3f}", penwidth=str(1 + 5 * weight))
    
    return dot

def main():
    # Set random seed for reproducibility
    #np.random.seed(42)
    
    # Number of nodes in the graph
    n_nodes = 10
    
    # Generate a random matrix
    random_matrix = generate_random_matrix(n_nodes)
    print("Original random matrix:")
    print(random_matrix)
    
    # Normalize weights using softmax (row-wise)
    normalized_matrix = normalize_weights(random_matrix)
    print("\nNormalized matrix (softmax row-wise):")
    print(normalized_matrix)
    
    # Try different thresholds
    thresholds = [0.9, 0.5, 0.1]
    for threshold in thresholds:
        # Construct the maximum probability graph
        graph_matrix = construct_maximum_probability_graph(random_matrix, threshold=threshold)
        print(f"\nMaximum probability graph matrix (threshold={threshold}):")
        print(graph_matrix)
        
        # Count edges
        edge_count = np.sum(graph_matrix > 0)
        print(f"Number of edges: {edge_count}")
        
        # Visualize the graph
        dot = visualize_graph(graph_matrix)
        
        # Render the graph
        dot.render(f'outputs/maximum_probability_graph_t{threshold}', format='png', cleanup=True)
        print(f"\nGraph visualization saved as 'maximum_probability_graph_t{threshold}.png'")
    
    # Return the graph with middle threshold
    return visualize_graph(construct_maximum_probability_graph(random_matrix, threshold=0.2))

if __name__ == "__main__":
    main()