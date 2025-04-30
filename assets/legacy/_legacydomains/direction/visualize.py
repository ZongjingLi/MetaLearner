if __name__ == "__main__":
    # Test with batch states
    import torch
    from domains.direction.direction_domain import direction_executor, direction_predicates
    import matplotlib.pyplot as plt
    batch_size1, batch_size2 = 4, 3
    context = {
        0: {"state": torch.randn(batch_size1, 2), "end": 1.0},  # [B1, 2]
        1: {"state": torch.randn(batch_size2, 2), "end": 1.0}   # [B2, 2]
    }
    
    # Test all predicates
    test_programs = [
        "(north_of $0 $1)",
        "(south_of $0 $1)",
        "(east_of $0 $1)",
        "(west_of $0 $1)",
        "(northeast_of $0 $1)",
        "(northwest_of $0 $1)",
        "(southeast_of $0 $1)",
        "(southwest_of $0 $1)",
        #"(angle_between $0 $1)"
    ]
    
    for program in test_programs:
        result = direction_executor.evaluate(program, context)["end"]  # [B1, B2]
        print(f"\n{program}")
        print(f"Result shape: {result.shape}")
        print(f"Result matrix:\n{result}")
        
        # Visualize
        fig = direction_predicates.visualize(
            context, result, program)
        plt.show()