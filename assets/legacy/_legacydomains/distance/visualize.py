if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    # Test with batch states
    from domains.distance.distance_domain import distance_executor, distance_predicates
    batch_size1, batch_size2, batch_size3 = 3, 2, 2
    context = {
        0: {"state": torch.tensor([[0.0, 0.0],
                                 [2.0, 2.0],
                                 [4.0, 0.0]]), "end": 1.0},  # [B1, 2]
        1: {"state": torch.tensor([[1.0, 0.0],
                                 [3.0, 1.0]]), "end": 1.0},  # [B2, 2]
        2: {"state": torch.tensor([[2.0, 0.0],
                                 [1.0, 2.0]]), "end": 1.0}   # [B3, 2]
    }
    
    # Test all predicates
    test_programs = [
        "(very_near $0 $1)",
        "(near $0 $1)",
        "(moderately_far $0 $1)",
        "(far $0 $1)",
        "(very_far $0 $1)",
        "(euclidean_distance $0 $1)",
        "(manhattan_distance $0 $1)",
    ]
    
    for program in test_programs:
        result = distance_executor.evaluate(program, context)["end"]
        print(f"\n{program}")
        print(f"Result shape: {result.shape}")
        print(f"Result matrix:\n{result}")
        
        # Visualize
        fig = distance_predicates.visualize(context, result, program)
        plt.show()