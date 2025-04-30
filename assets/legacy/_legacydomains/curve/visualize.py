if __name__ == "__main__":
    from domains.curve.curve_domain import curve2d_executor, curve2d_predicates
    # Test with example curves
    t = torch.linspace(0, 2*torch.pi, 50)
    
    # Create some test curves
    circle = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    spiral = torch.stack([t*torch.cos(t)/6, t*torch.sin(t)/6], dim=1)
    line = torch.stack([t/3, t/3], dim=1)
    wave = torch.stack([t/3, torch.sin(t)], dim=1)
    
    context = {
        0: {"state": torch.randn([1,64]), "end": 1.0},  # [B1=2, N=50, 2]
        1: {"state": torch.randn([1,64]), "end": 1.0}      # [B2=2, N=50, 2]
    }
    
    # Test predicates
    test_programs = [
        "(is_closed $0)",
        "(is_straight $1)",
        "(is_circular $0)",
        "(similar_shape $0 $1)",
        "(same_length $0 $1)",
        "(parallel_to $0 $1)",
        "(intersects $0 $1)",
        "(left_of $0 $1)",
        "(above $0 $1)",
        "(get_curvature $0)",
        "(get_complexity $0)",
        "(is_uniform $0)"
    ]
    
    for program in test_programs:
        result = curve2d_executor.evaluate(program, context)["end"]
        print(f"\n{program}")
        print(f"Result shape: {result.shape}")
        print(f"Result:\n{result}")
        
        # Visualize
        fig = curve2d_predicates.visualize(context, result, program)
        plt.show()
