# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:25:05
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-20 08:59:38
import torch
import torch.nn as nn
import itertools
from typing import List, Tuple, Set, Optional, Any
import torch.nn.functional as F
import math
from dataclasses import dataclass

class PrimitiveType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class ComplexType:
    def __init__(self, left, right, direction):
        self.left = left
        self.right = right
        self.direction = direction

    def __repr__(self):
        return f"{self.left}{self.direction}{self.right}"

class LexicalEntry(nn.Module):
    def __init__(self, word, program, syntactic_type, init_embedding = None, embedding_dim = 128):
        super(LexicalEntry, self).__init__()
        self.word = word
        self.program = program
        self.syntactic_type = syntactic_type
        #self.weight = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float32))
        self.embedding = nn.Parameter(init_embedding) if init_embedding is not None else nn.Parameter(torch.randn(embedding_dim))

    def forward(self):
        return self.weight

    def __repr__(self):
        return f"{self.word}: {self.syntactic_type}, embedding:{list(self.embedding.shape)}, {self.program}"


def match_entries(key: torch.Tensor, entries: List[LexicalEntry], temperature: float = 1.0, 
                 similarity_type: str = "scaled_dot_product") -> torch.Tensor:
    """
    Match a key embedding against a list of lexical entries to produce a distribution.
    
    Args:
        key: The query embedding tensor of shape [embedding_dim]
        entries: List of LexicalEntry objects to match against
        temperature: Temperature parameter for controlling the sharpness of the distribution
        similarity_type: The type of similarity to use: "dot_product", "scaled_dot_product", or "cosine"
        
    Returns:
        Probability distribution over entries as a tensor of shape [len(entries)]
    """
    # Check if the list is empty
    if not entries:
        return torch.tensor([])
        
    # Extract embeddings from all entries into a single tensor
    # Shape: [num_entries, embedding_dim]
    entry_embeddings = torch.stack([entry.embedding for entry in entries])
    
    # Ensure key is properly shaped [embedding_dim]
    if key.dim() > 1:
        key = key.squeeze()
    
    if similarity_type == "dot_product":
        # Simple dot product
        # Shape: [num_entries]
        scores = torch.matmul(entry_embeddings, key)
        
    elif similarity_type == "scaled_dot_product":
        # Scaled dot product (as in attention mechanisms)
        # Scale by square root of dimension to stabilize gradients
        dim = key.size(0)
        scores = torch.matmul(entry_embeddings, key) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        
    elif similarity_type == "cosine":
        # Cosine similarity
        # Normalize embeddings
        normalized_key = F.normalize(key, p=2, dim=0)
        normalized_entries = F.normalize(entry_embeddings, p=2, dim=1)
        scores = torch.matmul(normalized_entries, normalized_key)
    
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    # Apply temperature scaling
    scores = scores / temperature
    
    # Convert to probability distribution using softmax
    probs = F.softmax(scores, dim=0)
    
    return probs

def enumerate_ccg_types(parameters: List[str], result_type: str) -> Set:
    """
    Enumerate all possible CCG types for a function with given parameters.
    
    For a function f(a,b,c) that returns r, we generate types like:
    - a/b/c/r  (all forward slashes)
    - a\b\c\r  (all backward slashes)
    - a/b\c/r  (mixed slashes)
    - etc.
    
    Args:
        parameters: List of parameter names
        result_type: The return type
        
    Returns:
        Set of all possible CCG types
    """
    # Create primitive types for parameters and result
    param_types = [PrimitiveType(param) for param in parameters]
    result = PrimitiveType(result_type)
    
    # Generate all possible slash patterns (2^n combinations)
    n = len(parameters)
    all_types = set()
    
    # For each possible slash pattern
    for pattern in range(2**n):
        # Build the type from right to left (curried)
        current = result
        
        for i in range(n-1, -1, -1):
            param = param_types[i]
            # 0 represents forward slash, 1 represents backward slash
            direction = '\\' if (pattern & (1 << (n-1-i))) else '/'
            current = ComplexType(param, current, direction)
        
        all_types.add(current)
    
    return all_types

@dataclass
class ParseNode:
    """Node in a parse tree"""
    word: str  # The word or operation at this node
    left_child: Optional['ParseNode'] = None  # Left child, if any
    right_child: Optional['ParseNode'] = None  # Right child, if any
    weight: float = 1.0  # Weight of this parse tree
    
    def to_string(self) -> str:
        """Convert parse tree to string representation"""
        if self.left_child is None and self.right_child is None:
            return self.word
        
        if self.left_child is not None and self.right_child is not None:
            return f"{self.word}({self.left_child.to_string()}, {self.right_child.to_string()})"
        
        if self.left_child is not None:
            return f"{self.word}({self.left_child.to_string()})"
        
        return f"{self.word}({self.right_child.to_string()})"

@dataclass
class CellEntry:
    """Entry in a cell of the parsing chart"""
    parse_node: ParseNode
    lex_entry: Optional[LexicalEntry] = None  # For leaf nodes
    
    @property
    def weight(self) -> float:
        return self.parse_node.weight

class CKYE2Parser:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def can_combine(self, left_type, right_type):
        """Check if two types can be combined using CCG rules"""
        # Implement basic CCG combination rules
        # This is a simplified version - actual CCG would have more rules
        
        # Forward Application: X/Y Y => X
        if isinstance(left_type, ComplexType) and left_type.direction == '/' and str(left_type.right) == str(right_type):
            return True, left_type.left, "fa"
            
        # Backward Application: Y X\Y => X
        if isinstance(right_type, ComplexType) and right_type.direction == '\\' and str(right_type.right) == str(left_type):
            return True, right_type.left, "ba"
            
        return False, None, None
    
    def get_combination_result(self, left_entry: CellEntry, right_entry: CellEntry):
        """Try to combine two chart entries using CCG rules"""
        # Get syntactic types
        left_type = left_entry.lex_entry.syntactic_type if left_entry.lex_entry else None
        right_type = right_entry.lex_entry.syntactic_type if right_entry.lex_entry else None
        
        # For simplicity in this implementation, we're skipping the full CCG type checking
        # In a real implementation, you'd check CCG combinatory rules here
        
        # Create a combined parse node - using the left entry's word as operation 
        # (this is simplified; real CCG would determine the head differently)
        left_word = left_entry.parse_node.word
        combined_weight = left_entry.weight * right_entry.weight
        
        combined_node = ParseNode(
            word=left_word,
            left_child=left_entry.parse_node,
            right_child=right_entry.parse_node,
            weight=combined_weight
        )
        
        return CellEntry(parse_node=combined_node)
    
    def expected_execution(self, derivations: List[CellEntry]) -> List[CellEntry]:
        """Implement the EXPECTEDEXECUTION procedure from the algorithm"""
        if not derivations:
            return []
            
        # Keep processing until no more merges are possible
        while True:
            merged = False
            
            # Find pairs of derivations that can be merged
            for i in range(len(derivations)):
                if merged:
                    break
                    
                for j in range(i+1, len(derivations)):
                    # In the algorithm, we should check if x and y are identical except for subtrees
                    # of the same type. Here we'll just use a simple check.
                    x = derivations[i]
                    y = derivations[j]
                    
                    # Simple check: trees with same structure can be merged
                    if self._same_structure(x.parse_node, y.parse_node):
                        # Create a new merged node z
                        z_node = self._merge_nodes(x.parse_node, y.parse_node)
                        z = CellEntry(parse_node=z_node)
                        
                        # Replace x and y with z
                        derivations.pop(j)
                        derivations.pop(i)
                        derivations.append(z)
                        
                        merged = True
                        break
            
            if not merged:
                break
                
        return derivations
    
    def _same_structure(self, node1: ParseNode, node2: ParseNode) -> bool:
        """Check if two parse nodes have the same structure"""
        # Check if both nodes are leaves or both are non-leaves
        if (node1.left_child is None and node1.right_child is None) != (node2.left_child is None and node2.right_child is None):
            return False
            
        # For leaf nodes, check if the words are the same
        if node1.left_child is None and node1.right_child is None:
            return node1.word == node2.word
            
        # For non-leaf nodes, check if the operations are the same and recursively check children
        if node1.word != node2.word:
            return False
            
        left_same = True
        if node1.left_child is not None and node2.left_child is not None:
            left_same = self._same_structure(node1.left_child, node2.left_child)
        elif node1.left_child is not None or node2.left_child is not None:
            left_same = False
            
        right_same = True
        if node1.right_child is not None and node2.right_child is not None:
            right_same = self._same_structure(node1.right_child, node2.right_child)
        elif node1.right_child is not None or node2.right_child is not None:
            right_same = False
            
        return left_same and right_same
    
    def _merge_nodes(self, node1: ParseNode, node2: ParseNode) -> ParseNode:
        """Merge two parse nodes with the same structure"""
        # For leaf nodes, just add weights
        if node1.left_child is None and node1.right_child is None:
            return ParseNode(word=node1.word, weight=node1.weight + node2.weight)
            
        # For non-leaf nodes, recursively merge children
        left_child = None
        if node1.left_child is not None and node2.left_child is not None:
            left_child = self._merge_nodes(node1.left_child, node2.left_child)
            
        right_child = None
        if node1.right_child is not None and node2.right_child is not None:
            right_child = self._merge_nodes(node1.right_child, node2.right_child)
            
        return ParseNode(
            word=node1.word,
            left_child=left_child,
            right_child=right_child,
            weight=node1.weight + node2.weight
        )
    
    def parse(self, sentence: torch.Tensor, concept_diagram: Any) -> Tuple[List[str], torch.Tensor]:
        """
        Parse a sentence using the CKY-E² algorithm.
        
        Args:
            sentence: Tensor of shape [n, d] where n is the sentence length and d is embedding dimension
            concept_diagram: Object containing lexical entries
            
        Returns:
            List of parse tree strings and their probabilities
        """
        # Get sentence length
        L = sentence.shape[0]
        
        # Initialize the chart
        chart = {}
        
        # Initialize cells for single words (lines 1-3 in the algorithm)
        for i in range(L):
            word_embedding = sentence[i]
            entries = concept_diagram.get_lexicon_entries()  # This should return lexical entries for this position
            
            # Get distribution over lexical entries
            distrib = match_entries(word_embedding, entries, temperature=self.temperature)
            
            chart[(i, i+1)] = []
            
            # Create cell entries with corresponding weights
            for j, entry in enumerate(entries):
                if distrib[j] > 0:  # Only include non-zero probability entries
                    parse_node = ParseNode(word=entry.word, weight=distrib[j].item())
                    cell_entry = CellEntry(parse_node=parse_node, lex_entry=entry)
                    chart[(i, i+1)].append(cell_entry)
        
        # Dynamic programming to build larger spans (lines 4-14 in the algorithm)
        for length in range(2, L+1):
            for left in range(0, L-length+1):
                right = left + length
                
                chart[(left, right)] = []
                
                # Try all possible split points
                for k in range(left+1, right):
                    # Get entries from left and right spans
                    left_entries = chart.get((left, k), [])
                    right_entries = chart.get((k, right), [])
                    
                    # Try to combine each pair of entries
                    for left_entry in left_entries:
                        for right_entry in right_entries:
                            # Try to combine the entries
                            combined_entry = self.get_combination_result(left_entry, right_entry)
                            if combined_entry:
                                chart[(left, right)].append(combined_entry)
                
                # Apply expected execution to the current cell
                if chart[(left, right)]:
                    chart[(left, right)] = self.expected_execution(chart[(left, right)])
        
        # Extract final results from the top cell
        final_entries = chart.get((0, L), [])
        
        # Convert to output format
        parse_trees = []
        probabilities = []
        
        total_weight = sum(entry.weight for entry in final_entries)
        
        if total_weight > 0:
            for entry in final_entries:
                # Normalize weights to probabilities
                probability = entry.weight / total_weight
                
                parse_trees.append(entry.parse_node.to_string())
                probabilities.append(probability)
        
        return parse_trees, torch.tensor(probabilities)


# Example usage function
def parse_sentence(sentence_tensor: str, lexicon_backend) -> Tuple[List[str], torch.Tensor]:
    """
    Parse a sentence using the CKY-E² algorithm.
    
    Args:
        sentence_text: Input sentence as a string
        concept_diagram: Concept diagram with lexical entries
        
    Returns:
        List of parse tree strings and their probabilities
    """
    # In a real implementation, you would have a proper tokenizer and embedder
    # Here we'll just use random embeddings for demonstration

    
    parser = CKYE2Parser(temperature=1.0)
    return parser.parse(sentence_tensor, lexicon_backend)

# Example function to create sample data for demonstration
def create_sample_data():
    # Create some primitive types
    N = PrimitiveType("N")
    NP = PrimitiveType("NP")
    S = PrimitiveType("S")
    
    # Create some complex types
    NP_N = ComplexType(NP, N, "/")  # NP/N - a determiner
    S_NP = ComplexType(S, NP, "\\")  # S\NP - a verb phrase
    
    # Create some lexical entries
    entries = [
        LexicalEntry("the", "the", NP_N),
        LexicalEntry("a", "a", NP_N),
        LexicalEntry("green", "green", ComplexType(N, N, "/")),
        LexicalEntry("dog", "dog", N),
        LexicalEntry("cat", "cat", N),
        LexicalEntry("runs", "runs", S_NP),
        LexicalEntry("jumps", "jumps", S_NP)
    ]
    
    # Create a concept diagram with these entries
    concept_diagram = ConceptDiagram(entries)
    
    return concept_diagram

# Demonstrate usage
def demo():
    concept_diagram = create_sample_data()
    sentence = "the green dog runs"
    
    parse_trees, probabilities = parse_sentence(sentence, concept_diagram)
    
    print("Parse results:")
    for tree, prob in zip(parse_trees, probabilities):
        print(f"Tree: {tree}")
        print(f"Probability: {prob.item():.4f}")
        print()

    print(len(probabilities))

if __name__ == "__main__":
    demo()