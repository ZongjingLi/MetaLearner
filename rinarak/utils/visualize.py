import numpy as np

def visualize_pointclouds():
    return 

def visualize_image_batch():
    # [B,W,H,C]
    return 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from adjustText import adjust_text
from rinarak.knowledge.grammar import LexicalEntry, match_entries

def visualize_embeddings(entries: List[LexicalEntry], 
                         distribution: Optional[torch.Tensor] = None,
                         method: str = 'pca',
                         n_components: int = 2,
                         figsize: Tuple[int, int] = (12, 10),
                         title: Optional[str] = None,
                         show_labels: bool = True,
                         highlight_top_n: int = 5,
                         cmap: str = 'viridis',
                         return_projection: bool = False,
                         perplexity: int = 30,
                         learning_rate: Union[str, float] = 'auto'):
    """
    Visualize lexical entry embeddings using dimensionality reduction.
    
    Args:
        entries: List of LexicalEntry objects
        distribution: Optional probability distribution over entries
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components to reduce to (2 or 3)
        figsize: Figure size as (width, height)
        title: Optional title for the plot
        show_labels: Whether to show labels for points
        highlight_top_n: Number of top probability entries to highlight
        cmap: Colormap to use for probability visualization
        return_projection: Whether to return the projection model
        perplexity: Perplexity parameter for t-SNE
        learning_rate: Learning rate for t-SNE
        
    Returns:
        If return_projection is True, returns the projection model
    """
    if not entries:
        print("No entries to visualize")
        return
        
    # Extract embeddings and labels
    embeddings = torch.stack([entry.embedding.detach() for entry in entries]).numpy()
    labels = [f"{entry.word}: {entry.syntactic_type}" for entry in entries]
    
    # Perform dimensionality reduction
    if method.lower() == 'pca':
        projection = PCA(n_components=n_components)
        reduced_data = projection.fit_transform(embeddings)
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        projection = TSNE(n_components=n_components, perplexity=perplexity, 
                          learning_rate=learning_rate, n_iter=1000)
        reduced_data = projection.fit_transform(embeddings)
        method_name = 't-SNE'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create 2D or 3D plot
    if n_components == 2:
        ax = fig.add_subplot(111)
        
        # Plot points
        if distribution is not None:
            # Use distribution values for coloring points
            probs = distribution.detach().cpu().numpy()
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                c=probs, cmap=cmap, alpha=0.8, s=100)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Probability')
            
            # Highlight top N entries
            if highlight_top_n > 0:
                top_indices = np.argsort(probs)[-highlight_top_n:]
                ax.scatter(reduced_data[top_indices, 0], reduced_data[top_indices, 1], 
                          s=200, facecolors='none', edgecolors='red', linewidths=2)
        else:
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.8, s=100)
        
        # Add labels
        if show_labels:
            texts = []
            for i, label in enumerate(labels):
                texts.append(ax.text(reduced_data[i, 0], reduced_data[i, 1], label, 
                                   fontsize=9, ha='center', va='center'))
            
            # Adjust text positions to avoid overlap
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
            
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        if distribution is not None:
            probs = distribution.detach().cpu().numpy()
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                c=probs, cmap=cmap, alpha=0.8, s=100)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Probability')
            
            # Highlight top N entries
            if highlight_top_n > 0:
                top_indices = np.argsort(probs)[-highlight_top_n:]
                ax.scatter(reduced_data[top_indices, 0], reduced_data[top_indices, 1], 
                          reduced_data[top_indices, 2], s=200, facecolors='none', 
                          edgecolors='red', linewidths=2)
        else:
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                      alpha=0.8, s=100)
        
        # Add labels
        if show_labels:
            for i, label in enumerate(labels):
                ax.text(reduced_data[i, 0], reduced_data[i, 1], reduced_data[i, 2], 
                       label, fontsize=9)
    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'{method_name} visualization of lexical entries')
        
    ax.set_xlabel(f'Component 1')
    ax.set_ylabel(f'Component 2')
    if n_components == 3:
        ax.set_zlabel(f'Component 3')
    
    plt.tight_layout()
    plt.show()
    
    if return_projection:
        return projection


# Example usage
def example_visualization():
    # Create some sample lexical entries with different syntactic types
    embedding_dim = 128
    entries = [
        LexicalEntry("car", "car", "N", embedding_dim=embedding_dim),
        LexicalEntry("drive", "λx.drive(x)", "N\\S", embedding_dim=embedding_dim),
        LexicalEntry("fast", "λx.fast(x)", "N/N", embedding_dim=embedding_dim),
        LexicalEntry("quickly", "λP.λx.quickly(P(x))", "(S\\N)/(S\\N)", embedding_dim=embedding_dim),
        LexicalEntry("the", "λP.λQ.the(P,Q)", "N/N", embedding_dim=embedding_dim),
        LexicalEntry("red", "λx.red(x)", "N/N", embedding_dim=embedding_dim),
        LexicalEntry("book", "book", "N", embedding_dim=embedding_dim),
        LexicalEntry("read", "λx.λy.read(x,y)", "(S\\N)/N", embedding_dim=embedding_dim),
        LexicalEntry("eat", "λx.λy.eat(x,y)", "(S\\N)/N", embedding_dim=embedding_dim),
        LexicalEntry("John", "john", "N", embedding_dim=embedding_dim),
        LexicalEntry("Mary", "mary", "N", embedding_dim=embedding_dim),
    ]
    
    # Create a query embedding
    query_embedding = torch.randn(embedding_dim)
    
    # Get distribution over entries
    distribution = match_entries(query_embedding, entries, similarity_type="cosine")
    
    # Visualize with PCA
    visualize_embeddings(entries, distribution=distribution, method='pca', 
                         title='PCA visualization with probability distribution')
    
    # Visualize with t-SNE
    visualize_embeddings(entries, distribution=distribution, method='tsne', 
                         title='t-SNE visualization with probability distribution',
                         perplexity=5)  # Lower perplexity for small dataset
    
    # Visualize without distribution
    visualize_embeddings(entries, method='pca', title='PCA visualization without distribution')
    
    # 3D visualization
    visualize_embeddings(entries, distribution=distribution, method='pca', n_components=3,
                         title='3D PCA visualization')

# Run the example if this script is executed directly
if __name__ == "__main__":
    example_visualization()