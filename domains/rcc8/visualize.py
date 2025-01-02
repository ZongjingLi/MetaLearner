if __name__ == "__main__":
    # Test with batch states
    import torch
    from domains.rcc8.rcc8_domain import rcc8_executor, rcc8_predicates
    import matplotlib.pyplot as plt
    context = {
        0: {"state": torch.tensor([[0.0, 0.0, 1.0],]), "end": 1.0},  # [B1, 3]
        1: {"state": torch.tensor([[1.99, 0.0, 1.0],[1.0, 3.0, 1.0]
                                 ]), "end": 1.0}   # [B2, 3]
    }
    
    # Test all predicates
    test_programs = [
        "(disconnected $0 $1)",
        "(externally_connected $0 $1)",
        "(partial_overlap $0 $1)",
        "(equal $0 $1)",
        "(tangential_proper_part $0 $1)",
        "(non_tangential_proper_part $0 $1)",
        "(tangential_proper_part_inverse $0 $1)",
        "(non_tangential_proper_part_inverse $0 $1)"
    ]
    
    for program in test_programs:
        result = rcc8_executor.evaluate(program, context)["end"]
        print(f"\n{program}")
        print(f"Result matrix:\n{result}")
        
        # Visualize
        fig = rcc8_predicates.visualize(context, result, program)
        plt.show()