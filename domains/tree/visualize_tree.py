import numpy as np
import matplotlib.pyplot as plt
import graphviz
from scipy.special import softmax

def generate_random_matrix(n_nodes):
    """
    Generate a random matrix representing potential tree edges
    
    Parameters:
    - n_nodes: number of nodes in the tree
    
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
    # Apply softmax row-wise to get probability distributions
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        # Skip rows that are all zeros
        if np.sum(matrix[i]) > 0:
            normalized_matrix[i] = softmax(matrix[i])
    
    return normalized_matrix

def construct_maximum_spanning_tree(matrix, normalize=True):
    """
    Construct a maximum spanning tree from the probability matrix
    
    Parameters:
    - matrix: edge probability matrix
    - normalize: whether to normalize the matrix first
    
    Returns:
    - tree_matrix: adjacency matrix of the maximum spanning tree
    """
    n_nodes = matrix.shape[0]
    
    if normalize:
        # Normalize the matrix row-wise
        prob_matrix = normalize_weights(matrix)
    else:
        prob_matrix = matrix.copy()
    
    # Initialize an empty tree
    tree_matrix = np.zeros_like(prob_matrix)
    
    # Keep track of nodes in the tree
    nodes_in_tree = set()
    remaining_nodes = set(range(n_nodes))
    
    # Start with a random node
    start_node = np.random.randint(0, n_nodes)
    nodes_in_tree.add(start_node)
    remaining_nodes.remove(start_node)
    
    # Add edges until we have a complete tree
    while remaining_nodes and len(nodes_in_tree) < n_nodes:
        max_prob = -1
        best_edge = None
        
        # Find the maximum probability edge from a node in the tree to a node outside
        for i in nodes_in_tree:
            for j in remaining_nodes:
                if prob_matrix[i][j] > max_prob:
                    max_prob = prob_matrix[i][j]
                    best_edge = (i, j)
        
        if best_edge:
            i, j = best_edge
            # Add the edge to the tree
            tree_matrix[i][j] = prob_matrix[i][j]
            nodes_in_tree.add(j)
            remaining_nodes.remove(j)
        else:
            # No more valid edges, break to avoid infinite loop
            break
    
    return tree_matrix

def visualize_tree(matrix):
    """
    Visualize the tree using Graphviz
    
    Parameters:
    - matrix: tree adjacency matrix with edge probabilities
    
    Returns:
    - graphviz.Digraph object
    """
    n_nodes = matrix.shape[0]
    
    # Create a Graphviz graph
    dot = graphviz.Digraph(comment='Maximum Probability Tree')
    
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
    
    # Number of nodes in the tree
    n_nodes = 10
    
    # Generate a random matrix
    random_matrix = generate_random_matrix(n_nodes)
    print("Original random matrix:")
    print(random_matrix)
    
    # Normalize weights using softmax (row-wise)
    normalized_matrix = normalize_weights(random_matrix)
    print("\nNormalized matrix (softmax row-wise):")
    print(normalized_matrix)
    
    # Construct the maximum spanning tree
    tree_matrix = construct_maximum_spanning_tree(random_matrix)
    print("\nMaximum probability tree matrix:")
    print(tree_matrix)
    
    # Visualize the tree
    dot = visualize_tree(tree_matrix)
    
    # Render the graph
    dot.render('outputs/maximum_probability_tree', format='png', cleanup=True)
    print("\nTree visualization saved as 'maximum_probability_tree.png'")
    
    # Display in notebook (if running in a notebook)
    return dot

if __name__ == "__main__":
    main()