import torch
import torch.nn as nn
from helchriss.ccg.lexicon import CCGSyntacticType, LexiconEntry, SemProgram
from helchriss.ccg.parser import G2L2Parser, ForwardApplication, BackwardApplication

def create_domain_lexicon(domain):
    return


# Example usage
def create_sample_lexicon():
    """Create a sample lexicon for testing"""
    lexicon = {}
    
    # Define primitive types
    OBJ = CCGSyntacticType("objset")
    INT = CCGSyntacticType("int")
    
    # Define complex types
    OBJ_OBJ = CCGSyntacticType("objset", OBJ, OBJ, "/")  # objset/objset
    OBJ_OBJ_BACK = CCGSyntacticType("objset", OBJ, OBJ, "\\")  # objset\objset
    INT_OBJ = CCGSyntacticType("int", OBJ, INT, "/")  # int/objset
    
    # Create lexicon entries with PyTorch tensor weights
    
    # Nouns
    lexicon["cube"] = [
        LexiconEntry("cube", OBJ, SemProgram("filter", ["CUBE"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    lexicon["sphere"] = [
        LexiconEntry("sphere", OBJ, SemProgram("filter", ["SPHERE"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Adjectives
    lexicon["red"] = [
        LexiconEntry("red", OBJ_OBJ, SemProgram("filter", ["RED"], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    lexicon["blue"] = [
        LexiconEntry("blue", OBJ_OBJ, SemProgram("filter", ["BLUE"], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    lexicon["shiny"] = [
        LexiconEntry("shiny", OBJ_OBJ, SemProgram("filter", ["SHINY"], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Count
    lexicon["count"] = [
        LexiconEntry("count", INT_OBJ, SemProgram("count", [], ["x"]), torch.tensor(0.0, requires_grad=True)),
        LexiconEntry("count", INT_OBJ, SemProgram("id-count", [], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Prepositions
    lexicon["of"] = [
        LexiconEntry("of", OBJ_OBJ_BACK, SemProgram("id", [], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    # Determiners
    lexicon["the"] = [
        LexiconEntry("the", OBJ_OBJ, SemProgram("id", [], ["x"]), torch.tensor(0.0, requires_grad=True))
    ]
    
    return lexicon

def main():
    # Create sample lexicon and rules
    lexicon = create_sample_lexicon()
    rules = [ForwardApplication, BackwardApplication]
    
    # Create parser as a PyTorch module
    parser = G2L2Parser(lexicon, rules)
    
    # Test sentences
    test_sentences = [
        "red cube",
        "the red cube",
        "count red cube", 
        "shiny blue sphere",
        "count the shiny blue sphere"
    ]
    
    # Target program for optimization example
    target_program = "count(filter(RED, filter(CUBE)))"
    
    # Set up optimizer
    optimizer = torch.optim.Adam(parser.parameters(), lr=0.1)
    
    # Parse and print initial results
    print("Initial parsing results:")
    for sentence in test_sentences:
        print(f"\nParsing: '{sentence}'")
        parses = parser.parse(sentence)
        
        if parses:
            print(f"Found {len(parses)} valid parse(s):")
            for i, parse in enumerate(parses):
                print(f"{i+1}. {parse}")
                print(f"   Type: {parse.syn_type}")
                print(f"   Program: {parse.sem_program}")
                print(f"   Weight: {parse.weight.item()}")
                print(f"   Probability: {torch.exp(parse.weight).item():.6f}")
        else:
            print("No valid parses found.")
    
    # Example training loop to optimize for "count red cube" → target_program
    print("\n\nTraining to optimize 'count red cube' to generate target program...")
    for epoch in range(1000):
        optimizer.zero_grad()
        
        # Forward pass - compute loss
        loss = parser.forward("count red cube", target_program)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    # Parse and print results after optimization
    print("\nParsing results after optimization:")
    for sentence in test_sentences:
        print(f"\nParsing: '{sentence}'")
        parses = parser.parse(sentence)
        
        if parses:
            print(f"Found {len(parses)} valid parse(s):")
            for i, parse in enumerate(parses):
                print(f"{i+1}. {parse}")
                print(f"   Type: {parse.syn_type}")
                print(f"   Program: {parse.sem_program}")
                print(f"   Weight: {parse.weight}")
                print(f"   Probability: {torch.exp(parse.weight):.6f}")
        else:
            print("No valid parses found.")
    
    # Show how to get parse probability for backpropagation
    print("\nDemonstrating how to get parse probability for backpropagation:")
    sentence = "count red cube"
    parses = parser.parse(sentence)

    if parses:
        probs = parser.get_parse_probability(parses)
        for i, parse in enumerate(parses):
            print(parse.sem_program)
            print(probs[i].exp())
if __name__ == "__main__":
    main()