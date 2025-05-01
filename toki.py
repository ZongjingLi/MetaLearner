import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import LexiconEntry, SemProgram,CCGSyntacticType
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

def create_and_train_parser():
    """Example of creating and training a parser with centralized weights"""
    
    # Create basic lexicon
    obj_type = CCGSyntacticType("objset")
    pred_type = CCGSyntacticType("pred", obj_type, obj_type, "\\")
    
    # Create lexicon entries
    lexicon = {}
    
    # Add entries with initial weights
    lexicon["cat"] = [
        LexiconEntry("cat", obj_type, SemProgram("cat"), torch.tensor(0.8)),
    ]
    
    lexicon["dog"] = [
        LexiconEntry("dog", obj_type, SemProgram("dog"), torch.tensor(0.9)),
    ]
    
    lexicon["runs"] = [
        LexiconEntry("runs", pred_type, SemProgram("run"), torch.tensor(0.7)),
    ]
    
    # Create parser with our lexicon
    parser = ChartParser(lexicon)
    
    # Verify parameters are registered
    print(f"Total parameters: {sum(p.numel() for p in parser.parameters())}")
    print("Parameter names:")
    for name, param in parser.named_parameters():
        print(f"  - {name}: {param.shape}")
    
    # Example training data: (sentence, index of correct parse)
    training_data = [
        ("cat runs", 0),
        ("dog runs", 0)
    ]
    
    # Training loop
    optimizer = optim.Adam(parser.parameters(), lr=0.01)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for sentence, target_idx in training_data:
            # Reset gradients
            optimizer.zero_grad()
            
            # Parse sentence
            parses = parser.parse(sentence)
            
            if not parses:
                print(f"Warning: No parse found for '{sentence}'")
                continue
            
            # Calculate loss
            loss = parser.compute_loss(parses, target_idx)
            
            # Backpropagation
            loss.backward()
            
            # Debug gradient flow (uncomment to check)
            # debug_gradient_flow(parser)
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(training_data):.4f}")
    
    # Show final lexicon weights
    print("\nFinal lexicon weights:")
    for word, entries in parser.lexicon.items():
        for idx in range(len(entries)):
            weight_key = f"{word}_{idx}"
            weight = parser.lexicon_weight[weight_key].item()
            print(f"  {word} ({idx}): {weight:.4f}")
    
    # Test parser on a sentence
    test_sentence = "cat runs"
    parses = parser.parse(test_sentence)
    
    print(f"\nParses for '{test_sentence}':")
    log_probs = parser.get_parse_probability(parses)
    
    for i, (parse, log_prob) in enumerate(zip(parses, log_probs)):
        prob = torch.exp(log_prob).item()
        print(f"Parse {i}: {parse}, Probability: {prob:.4f}")
    
    return parser

# Example function to inspect weights during training
def inspect_weights(parser, step):
    """Print weights at a specific training step"""
    print(f"\nWeights at step {step}:")
    for name, param in parser.lexicon_weight.named_parameters():
        print(f"  {name}: {param.item():.4f}")

# Example usage:
parser = create_and_train_parser()