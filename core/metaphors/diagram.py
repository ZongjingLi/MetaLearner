import os
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import functools

from rinarak.knowledge.executor import CentralExecutor
from rinarak.knowledge.executor import type_dim

from core.metaphors.base import StateMapper

"""Schema Class: 
a dataclass that stores
"""
@dataclass
class Schema(object):
    end : torch.Tensor
    mask : torch.Tensor
    dtype : str
    domain : str

def product(*schemas: Schema) -> Schema:
    """
    Compute the tensor product of multiple Schema objects without in-place operations
    to maintain differentiability.
    
    Args:
        *schemas: Variable number of Schema objects to combine
        
    Returns:
        Schema: A new Schema with combined dimensions according to tensor product rules
        
    Raises:
        ValueError: If schemas have different domains or if no schemas are provided
    """
    if len(schemas) == 0:
        raise ValueError("At least one Schema must be provided")
    
    # Check if all schemas are in the same domain
    domains = {schema.domain for schema in schemas}
    if len(domains) > 1:
        raise ValueError(f"All schemas must be in the same domain. Found: {domains}")
    
    # Extract the first schema to use its domain and dtype
    first_schema = schemas[0]
    
    # For the case of a single schema, return it directly
    if len(schemas) == 1:
        return first_schema
    
    # Get masks and ends from all schemas
    masks = [schema.mask for schema in schemas]
    ends = [schema.end for schema in schemas]
    
    # Calculate the shape of the final mask and end tensors
    leading_dims = [end.shape[0] for end in ends]
    last_dims = [end.shape[1] for end in ends]
    
    # Calculate final shapes
    final_mask_shape = leading_dims + [1]
    final_end_shape = leading_dims + [sum(last_dims)]
    
    # Step 1: Process the masks using einsum for tensor product
    # First, create a mask tensor for each schema that's expanded correctly
    expanded_masks = []
    
    for i, mask in enumerate(masks):
        # For n schemas, we need a tensor with n+1 dimensions
        # All dimensions should be 1 except for the i-th dimension (which should match the original)
        # and the last dimension (which is always 1)
        shape = [1] * (len(schemas) + 1)
        shape[i] = leading_dims[i]
        
        # Reshape to this shape
        expanded_mask = mask.reshape(shape)
        expanded_masks.append(expanded_mask)
    
    # Multiply all masks together (broadcasting will handle the expansion)
    final_mask = expanded_masks[0]
    for i in range(1, len(expanded_masks)):
        final_mask = final_mask * expanded_masks[i]
    
    # Step 2: Process the end tensors
    # We'll build up the end tensor by concatenating slices
    
    # Initialize list to hold slices
    slices = []
    
    # Current offset in the last dimension of the output tensor
    offset = 0
    
    # We need to handle all combinations of indices from the leading dimensions
    # Do this by creating a CartesianProduct of ranges
    indices_list = [range(dim) for dim in leading_dims]
    
    # Loop through all combinations of indices
    from itertools import product as cartesian_product
    for indices in cartesian_product(*indices_list):
        # Create a slice for this combination of indices
        # It will have shape [sum(last_dims)]
        slice_parts = []
        
        # Extract the corresponding slice from each schema's end tensor
        for schema_idx, schema in enumerate(schemas):
            end = schema.end
            idx = indices[schema_idx]
            
            # Extract the slice from this schema at the given index
            end_slice = end[idx]  # Shape: [last_dim_i]
            slice_parts.append(end_slice)
        
        # Concatenate all parts to form this slice
        full_slice = torch.cat(slice_parts)
        slices.append(full_slice)
    
    # Reshape all slices into the final tensor shape
    # First, stack all slices into a tensor of shape [num_combinations, sum(last_dims)]
    stacked_slices = torch.stack(slices)
    
    # Then reshape to the final shape [d1, d2, ..., dn, sum(last_dims)]
    final_end = stacked_slices.reshape(final_end_shape)
    
    return Schema(
        end=final_end,
        mask=final_mask,
        dtype=first_schema.dtype,
        domain=first_schema.domain
    )

def schemas_to_context(schemas):
    """
    Transform a list of Schema objects into a context dictionary.
    
    Args:
        schemas (List[Schema]): List of Schema objects to transform
        
    Returns:
        dict: A context dictionary with indices as keys and dictionaries containing
              'state' and 'end' as values.
              
    Example:
        If schemas is a list of Schema objects, the output will be:
        {
            0: {"state": state_0, "end": 1.0},
            1: {"state": state_1, "end": 1.0},
            ...
        }
    """
    context = {}
    
    for i, schema in enumerate(schemas):
        # Extract the state from the schema
        # Assuming the schema.end tensor represents the state or can be converted to it
        # You may need to adjust this based on your actual schema structure
        state = schema.end
        
        # Create the context entry for this schema
        context[i] = {
            "state": state,
            "end": schema.mask  # Fixed end value as per requirement
        }
    
    return context

def generate_lambda_predicate(predicate_name, num_params):
    """
    Generate a predicate string with numbered parameters.
    
    Args:
        predicate_name (str): The name of the predicate
        num_params (int): The number of parameters for the predicate
        
    Returns:
        str: A formatted predicate string with numbered parameters
        
    Example:
        generate_predicate("contact", 2) returns "(contact $0 $1)"
    """
    # Create the list of parameters
    params = [f"${i}" for i in range(num_params)]
    
    # Join parameters with spaces
    params_str = " ".join(params)
    
    # Format the complete predicate
    predicate = f"({predicate_name} {params_str})"
    
    return predicate

"""Concept Domain"""
class ConceptDomain(nn.Module):
    """this is just a wrapper class of the central executor"""
    def __init__(self, executor : CentralExecutor):
        super().__init__()
        domain = executor.domain
        self.name = domain.domain_name
        self.domain = domain
        self.types = domain.types
        self.predicates = domain.predicates
        self.executor = executor

    
    def get_lexicon_entries(self):
        entries = self.executor.get_lexicon_entries()

        return entries

"""Metaphor Morphism Class:
connection between two domains, matchings types and scores between source type and target type"""
class MetaphorMorphism(nn.Module):
    def __init__(self, source_domain : ConceptDomain, target_domain : ConceptDomain, hidden_dim = 128):
        super().__init__()
        self.source_domain = source_domain
        self.target_domain = target_domain

        """entailment relations between source types and target types"""
        self.source_types = list(source_domain.types)
        self.target_types = list(target_domain.types)
        # here type matching is stored as the logits, some default connections are fixed
        self.type_matching = nn.Parameter(torch.randn(len(self.source_types), len(self.target_types)))

        self.type_mappers = nn.ModuleDict({})
        for s_type in source_domain.types:
            s_dim = type_dim(source_domain.types[s_type])[0][0]
            for t_type in target_domain.types:
                t_dim = type_dim(target_domain.types[t_type])[0][0]
                """f_s: as the state mapping from source state to the target state"""
                self.type_mappers[f"{s_type}_{t_type}_map"] = \
                    StateMapper(
                        source_dim=s_dim,
                        target_dim=t_dim,
                        hidden_dim=hidden_dim
                    )
                self.type_mappers[f"{s_type}_{t_type}_classify"] = \
                    StateMapper(
                        source_dim = s_dim,
                        target_dim = 1,
                        hidden_dim = hidden_dim
                    )

        """source symbols and target symbols can be mapped by a connection matrix"""
        self.source_symbols = list(source_domain.predicates)
        self.target_symbols = list(target_domain.predicates)
        # store the symbol conections
        self.symbol_matching = nn.Parameter(torch.randn(len(self.source_symbols), len(self.target_symbols)))
    
    def forward(self, schema: Schema, target_type=None) -> Schema:
        """Map a schema to the target type schema and update the feature and the logit scores
        If the target_type is None, then choose the most probable mapping in the type matching and perform transition
        """
        # First, get the source type from the schema
        source_type = schema.dtype
        source_idx = self.source_types.index(source_type)
        
        # If target_type is not specified, choose the most probable one based on type_matching
        if target_type is None:
            type_logits = self.type_matching[source_idx]
            target_idx = torch.argmax(type_logits).item()
            target_type = self.target_types[target_idx]
            # Get the confidence score for this mapping
            type_confidence = type_logits[target_idx]
        else:
            # If target_type is specified, get its index and confidence
            target_idx = self.target_types.index(target_type)
            type_confidence = self.type_matching[source_idx, target_idx]
        
        # Apply the appropriate state mapper to transform the source state to target state
        mapper_key = f"{source_type}_{target_type}_map"
        classifier_key = f"{source_type}_{target_type}_classify"
        
        # Apply the feature mapping - transform from source state to target state
        mapped_state = self.type_mappers[mapper_key](schema.end)
        
        # Get the classification confidence (how well this maps)
        mapping_confidence = self.type_mappers[classifier_key](schema.end).squeeze(-1)
        
        # Compute the final confidence as a combination of type matching and state mapping confidence
        # Using sigmoid to convert logits to probability domain for multiplication, then back to logit
        combined_conf_prob = torch.sigmoid(type_confidence) * torch.sigmoid(mapping_confidence)
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-10
        combined_conf_prob = torch.clamp(combined_conf_prob, epsilon, 1 - epsilon)
        #final_confidence = torch.log(combined_conf_prob / (1 - combined_conf_prob))
        
        # Create a new mask that combines the original mask with the confidence
        new_mask = schema.mask * combined_conf_prob
        
        # Return a new Schema with the mapped state and updated mask
        return Schema(
            end=mapped_state,
            mask=new_mask,
            dtype=target_type,
            domain=self.target_domain.name
        ), torch.sigmoid(type_confidence)

"""Concept Diagram Class :
the skeleton graph of the somains connected by the Metaphor Morphisms
"""
class ConceptDiagram(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = (
            "cuda:0" if torch.cuda.is_available()
            else "mps:0" if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.domains = nn.ModuleDict({})
        self.metaphors = nn.ModuleDict({})
        
        self.eps = 1e-6

    """Basic Setup for the nodes and edges in the Conceptual Diagram"""
    def add_node(self, domain, executor, logits=False):
        """
        Adds a domain (node) to the concept diagram.
        
        Args:
            domain (str): The name of the domain
            executor (CentralExecutor): The executor for this domain
            logits (bool or torch.Tensor): Confidence logits for this domain. 
                                          If bool, converts to tensor with default confidence
        
        Returns:
            None
        """
        if isinstance(logits, bool): 
            logits = torch.logit(torch.tensor(1.0 - self.eps))
        self.domains[domain] = ConceptDomain(executor)

    def get_node(self, domain, logits=True) -> ConceptDomain:
        """
        Retrieves a domain (node) from the concept diagram.
        
        Args:
            domain (str): The name of the domain to retrieve
            logits (bool): Flag indicating whether to return confidence logits
        
        Returns:
            ConceptDomain: The domain object for the specified name
        
        Raises:
            KeyError: If the domain doesn't exist in the diagram
        """
        return self.domains[domain]

    def add_edge(self, source, target, edge_id=None, morphism=None):
        """
        Adds an edge (metaphor) between source and target domains.
        In a multigraph, multiple edges can connect the same pair of domains.
        
        Args:
            source (str): The source domain name
            target (str): The target domain name
            edge_id (str, optional): Identifier for this specific edge. If None, will generate one
            morphism (MetaphorMorphism, optional): The morphism to use. If None, creates a new one
        
        Returns:
            str: The edge_id used for storing this edge
        """
        # Create default morphism if none provided
        if morphism is None:
            morphism = MetaphorMorphism(self.get_node(source), self.get_node(target))
        
        if edge_id is None:
            # Generate a unique identifier based on existing edges between these domains
            existing_count = sum(1 for key in self.metaphors.keys() 
                               if key.startswith(f"{source}__{target}__"))
            edge_id = f"{source}__{target}__{existing_count}"
        
        # Store the morphism in the metaphors dictionary
        self.metaphors[edge_id] = morphism
        
        return edge_id

    def get_edge(self, edge_id):
        """
        Get a specific edge by its ID.
        
        Args:
            edge_id (str): The identifier for the edge
            
        Returns:
            MetaphorMorphism: The morphism associated with this edge
        
        Raises:
            KeyError: If the edge_id doesn't exist
        """
        if edge_id in self.metaphors:
            return self.metaphors[edge_id]
        raise KeyError(f"Edge with ID '{edge_id}' not found in the diagram")
    
    def get_edges_between(self, source, target):
        """
        Get all edges between two domains.
        
        Args:
            source (str): The source domain name
            target (str): The target domain name
            
        Returns:
            dict: Dictionary of {edge_id: morphism} for all edges between source and target
        """
        prefix = f"{source}__{target}__"
        return {
            edge_id: morphism for edge_id, morphism in self.metaphors.items()
            if edge_id.startswith(prefix)
        }
    
    def get_all_edges(self):
        """
        Get all edges in the diagram.
        
        Returns:
            dict: Dictionary of all {edge_id: morphism} pairs
        """
        return dict(self.metaphors)
    
    def get_outgoing_edges(self, source):
        """
        Get all edges that originate from a specific domain.
        
        Args:
            source (str): The source domain name
            
        Returns:
            dict: Dictionary of {edge_id: morphism} for all edges starting from source
        """
        prefix = f"{source}__"
        return {
            edge_id: morphism for edge_id, morphism in self.metaphors.items()
            if edge_id.startswith(prefix)
        }
    
    def get_incoming_edges(self, target):
        """
        Get all edges that point to a specific domain.
        
        Args:
            target (str): The target domain name
            
        Returns:
            dict: Dictionary of {edge_id: morphism} for all edges pointing to target
        """
        # This requires parsing the edge_id to extract parts
        return {
            edge_id: morphism for edge_id, morphism in self.metaphors.items()
            if edge_id.split("__")[1] == target
        }
    
    def parse_edge_id(self, edge_id):
        """
        Parse an edge_id to get source, target, and index.
        
        Args:
            edge_id (str): The edge identifier in format "source__target__index"
            
        Returns:
            tuple: (source, target, index)
        """
        parts = edge_id.split("__")
        if len(parts) != 3:
            raise ValueError(f"Invalid edge_id format: {edge_id}")
        return parts[0], parts[1], parts[2]
    
    """Collect the Lexicon Entries for each Concept Domain"""
    def get_lexicon_entries(self):
        """collect all the lexicon entries in each of the ConceptDomain"""
        entries = []
        for key, concept_domain in self.domains.items():
            for entry in concept_domain.get_lexicon_entries():
                entries.append(entry)
        return entries

    def get_predicate(self, predicate):
        """return the predicate diction {name, parameters, type}"""
        for name, domain in self.domains.items():
            for pname in domain.predicates:
                if pname == predicate:
                    return name, domain.predicates[pname]
        return

    """Conceptual Metaphor Extentions the Concept Domains"""
    def get_schema_path(self, source_schema, target_domain, target_type):
        """Given a source schema, find all the paths from its domain to the target domain and target_type
        transitions between schemas using the Metaphor Morphisms
        
        Args:
            source_schema (Schema): The starting schema
            target_domain (str): The name of the target domain
            target_type (str): The type in the target domain to reach
            
        Returns:
            list: List of tuples (edge_path, schema_path, probability) where:
                - edge_path is a list of edge_ids representing the path
                - schema_path is a list of schemas representing the transformations
                - probability is the cumulative probability of the entire path
        """
        source_domain = source_schema.domain
        
        # If already in target domain with correct type, return empty path with probability 1.0
        if source_domain == target_domain and source_schema.dtype == target_type:
            return [([], [source_schema], 1.0)]
        
        # Track visited states to avoid cycles
        visited = set()
        
        # Use BFS to find paths
        def bfs_paths():
            # Queue entries are (current_domain, current_schema, edge_path, schema_path, cumulative_prob)
            queue = [(source_domain, source_schema, [], [source_schema], 1.0)]
            results = []
            
            while queue:
                current_domain, current_schema, edge_path, schema_path, cum_prob = queue.pop(0)
                
                # Generate a state key to track visited states
                state_key = (current_domain, current_schema.dtype)
                if state_key in visited:
                    continue
                visited.add(state_key)
                
                # Get all outgoing edges from current domain
                outgoing_edges = self.get_outgoing_edges(current_domain)
                
                for edge_id, morphism in outgoing_edges.items():
                    # Parse the edge to get target domain
                    _, edge_target, _ = self.parse_edge_id(edge_id)
                    
                    # Apply the morphism to get the new schema
                    # If target_type specified and we're at target domain, use it
                    if edge_target == target_domain:
                        new_schema, transition_prob = morphism(current_schema, target_type=target_type)
                    else:
                        new_schema, transition_prob = morphism(current_schema)
                    
                    # Calculate transition probability from the mask values
                    # The mask in schema contains confidence values in probability space
                    # Take the average confidence as the probability of this transition
                    #transition_prob = torch.mean(new_schema.mask).item()
                    #print(transition_prob)
                    
                    # Update cumulative probability (multiply by transition probability)
                    new_cum_prob = cum_prob * transition_prob
                    #print(new_cum_prob, transition_prob)
                    
                    # New path information
                    new_edge_path = edge_path + [edge_id]
                    new_schema_path = schema_path + [new_schema]
                    
                    # Check if we've reached the target
                    if new_schema.domain == target_domain and new_schema.dtype == target_type:
                        results.append((new_edge_path, new_schema_path, new_cum_prob))
                    else:
                        # Continue searching (only if probability is not too low)
                        if new_cum_prob > 1e-5:  # Threshold to avoid exploring very unlikely paths
                            queue.append((
                                edge_target, 
                                new_schema, 
                                new_edge_path, 
                                new_schema_path, 
                                new_cum_prob
                            ))
            
            return results
        
        # Run BFS to find all paths
        paths = bfs_paths()
        
        # Sort paths by probability (highest first)
        paths.sort(key=lambda p: p[2], reverse=True)
        
        return paths
    
    def get_transformed_schema(self, input_schema, target_domain, expected_type):
        """
        Get the transformed schema by finding paths from input_schema to target_domain with expected_type.
        Returns a weighted combination of transformed schemas from all valid paths,
        where the weights are normalized path probabilities.

        Args:
            input_schema (Schema): The starting schema
            target_domain (str): The name of the target domain
            expected_type (str): The type in the target domain to reach

        Returns:
            Schema: A weighted combination of all possible transformed schemas
        """
        # Use the existing get_schema_path method
        paths = self.get_schema_path(input_schema, target_domain, expected_type)

        if not paths:
            raise ValueError(f"No valid path found from {input_schema.domain} to {target_domain} with type {expected_type}")

        # Extract all final schemas and their probabilities
        final_schemas = [path[1][-1] for path in paths]  # Last schema in each path
        path_probs = [path[2] for path in paths]  # Probability of each path

        # Normalize the probabilities to sum to 1
        total_prob = sum(path_probs)
        normalized_probs = [prob / total_prob for prob in path_probs]


        # If there's only one path, return its schema directly
        if len(paths) == 1:
            #print(paths[0][1][-1].end.shape)
            return final_schemas[0]

        # Create a weighted combination of all schemas
        # First, verify that all schemas have compatible shapes
        # We'll assume that the "end" tensors should have the same shape except possibly the last dimension
        # and that "mask" tensors should have the same shape

        # Check end tensor shapes
        end_shapes = [schema.end.shape[:-1] for schema in final_schemas]
        if not all(shape == end_shapes[0] for shape in end_shapes):
            raise ValueError("Cannot combine schemas with incompatible end tensor shapes")

        # Check mask tensor shapes
        mask_shapes = [schema.mask.shape for schema in final_schemas]
        if not all(shape == mask_shapes[0] for shape in mask_shapes):
            raise ValueError("Cannot combine schemas with incompatible mask tensor shapes")

        # Get the maximum last dimension size for "end" tensors to determine the combined size
        max_last_dim = max(schema.end.shape[-1] for schema in final_schemas)

        # Create a new "end" tensor with the combined shape
        leading_dims = list(final_schemas[0].end.shape[:-1])  # All dimensions except the last
        new_end_shape = leading_dims + [max_last_dim]

        new_end = torch.zeros(
            new_end_shape,
            dtype=final_schemas[0].end.dtype,
            device=final_schemas[0].end.device
        )

        # Create a new "mask" tensor with the same shape as the original masks
        new_mask = torch.zeros(
            final_schemas[0].mask.shape,
            dtype=final_schemas[0].mask.dtype,
            device=final_schemas[0].mask.device
        )


        # Fill in the new tensors as a weighted combination of the original tensors
        for i, (schema, weight) in enumerate(zip(final_schemas, normalized_probs)):
            # For the "end" tensor, we need to handle possibly different last dimension sizes
            last_dim_size = schema.end.shape[-1]

            # Create slices for the leading dimensions (all of them)
            indices = [slice(None)] * len(new_end_shape)

            # Set the last dimension slice to only include the relevant portion
            indices[-1] = slice(0, last_dim_size)

            # Update the new_end tensor with the weighted contribution from this schema
            new_end[tuple(indices)] += schema.end * weight
            #print(schema.end)
            # Update the new_mask tensor with the weighted contribution
            new_mask += schema.mask * weight

        # Create and return the combined schema
        return Schema(
            end=new_end,
            mask=new_mask,
            dtype=final_schemas[0].dtype,
            domain=target_domain
        )

    """TODO: 实在理不清了, 先按照每个参数按照概率加权然后合并来算，以后再考虑topK(真的服了)"""
    def evaluate_predicate(self, predicate, input_schemas, top_k = None):
        """
        for each argument
            1. search paths from the source schema (arg) to the domain predicate located (and the corresponding arg type)
            2. find the expected "end" and "mask" according to the path probability (normalize them)
        then we have the expected arguments.
        make the product of them to get the new product schema, then execute on the predicate using the corresponding domain executor
        on the "end" of the product schema
        """
        domain, predicate_bind = self.get_predicate(predicate) # like {'name': 'domain', 'parameters': ['?x-function'], 'type': 'set'}
        expected_params = predicate_bind['parameters']
        result_type = predicate_bind['type']
        domain_executor = self.domains[domain].executor
        
        if len(input_schemas) != len(expected_params):
            raise ValueError(f"Expected {len(expected_params)} arguments for predicate {predicate}, got {len(input_schemas)}")

        """collect the expected transform from source to target"""
        argument_schemas = []

        for i, (param, input_schema) in enumerate(zip(expected_params, input_schemas)):
            # Extract the expected type from the parameter (e.g., '?x-function' -> 'function')
            expected_type = param.split('-')[1] if '-' in param else None
            if expected_type is None:
                raise ValueError(f"Could not determine type for parameter {param}")

            transformed_schema = self.get_transformed_schema(input_schema, domain, expected_type)

            argument_schemas.append(transformed_schema)
        """if input in same source domain, the find the retraction"""
    
        # Make the product of the argument schemas to get the new product schema
        # Use the product function we defined earlier
        product_schema = product(*argument_schemas)

        expression = generate_lambda_predicate(predicate, len(expected_params))
        context = schemas_to_context(argument_schemas)

        results = domain_executor.evaluate(expression, context)

 
        return Schema(end = results["end"], mask = product_schema.mask, dtype = result_type, domain = domain)