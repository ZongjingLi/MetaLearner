import torch
import matplotlib.pyplot as plt
from meta_construction import *
# Initialize the concept diagram as shown in the main block

if __name__ == "__main__":
    from domains.generic.generic_domain import generic_executor
    from domains.line.line_domain import line_executor
    from domains.rcc8.rcc8_domain import rcc8_executor
    from domains.curve.curve_domain import curve_executor
    from domains.distance.distance_domain import distance_executor
    from domains.direction.direction_domain import direction_executor

concept_diagram = ConceptDiagram()

# Add domains
concept_diagram.add_domain("GenericDomain", generic_executor)
concept_diagram.add_domain("LineDomain", line_executor)
concept_diagram.add_domain("CurveDomain", curve_executor)
concept_diagram.add_domain("RCC8Domain", rcc8_executor)
concept_diagram.add_domain("DistanceDomain", distance_executor)
concept_diagram.add_domain("DirectionDomain", direction_executor)

# Add morphisms
concept_diagram.add_morphism("GenericDomain", "LineDomain", 
                           MetaphorMorphism(generic_executor, line_executor))
concept_diagram.add_morphism("GenericDomain", "DistanceDomain", 
                           MetaphorMorphism(generic_executor, distance_executor))
concept_diagram.add_morphism("GenericDomain", "DirectionDomain", 
                           MetaphorMorphism(generic_executor, direction_executor))
concept_diagram.add_morphism("DistanceDomain", "DirectionDomain", 
                           MetaphorMorphism(distance_executor, direction_executor))
concept_diagram.add_morphism("CurveDomain", "LineDomain", 
                           MetaphorMorphism(curve_executor, line_executor))
concept_diagram.add_morphism("LineDomain", "RCC8Domain", 
                           MetaphorMorphism(line_executor, rcc8_executor))
concept_diagram.add_morphism("DistanceDomain", "RCC8Domain", 
                           MetaphorMorphism(line_executor, rcc8_executor))

# Example 1: Visualize the overall concept diagram
print("Visualizing overall concept diagram...")
concept_diagram.visualize(figsize=(12, 8))

# Example 2: Visualize specific paths
print("\nVisualizing path from GenericDomain to RCC8Domain...")
concept_diagram.visualize_path("GenericDomain", "RCC8Domain")

# Example 3: Visualize predicate tracing
print("\nVisualizing predicate tracing...")
# Get a path from GenericDomain to RCC8Domain
path = concept_diagram.get_path("GenericDomain", "RCC8Domain")[0]

# Create example state tensor
batch_size = 4
state_dim = generic_executor.state_dim[0]
example_state = torch.randn(batch_size, state_dim)

# Example predicate from RCC8 domain
target_predicate = "disconnected"  # Disconnected predicate from RCC8

# Perform metaphorical evaluation
eval_result = concept_diagram._evaluate_metaphor(
    state=example_state,
    predicate_expr=f"({target_predicate} $0 $1)",
    source_domain="GenericDomain",
    target_domain="RCC8Domain",
    top_k=1
)

# Visualize connection matrices along the path
print("\nVisualizing connection matrices along the path...")
concept_diagram.visualize_connection_matrices(path, target_predicate)

# Example 4: Detailed predicate tracing visualization
print("\nDetailed predicate tracing visualization...")
morphisms_dict = {}
for src, tgt, idx in path:
    morphisms_dict[(src, tgt, idx)] = concept_diagram.get_morphism(src, tgt, idx)

visualize_predicate_tracing(
    path=path,
    domains=concept_diagram.domains,
    morphisms=morphisms_dict,
    target_predicate=target_predicate,
    figsize=(15, len(path) * 4)
)

# Print path information
print("\nPath details:")
for src, tgt, idx in path:
    print(f"{src} → {tgt} (morphism index: {idx})")
    morphism = concept_diagram.get_morphism(src, tgt, idx)
    connection_matrix, _ = morphism.predicate_matrix()
    print(f"Connection matrix shape: {connection_matrix.shape}")