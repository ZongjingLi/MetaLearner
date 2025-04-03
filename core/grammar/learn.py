import torch
import torch.nn as nn

import itertools
from typing import Dict, List, Tuple, Set, Optional, Any
try:
    from .lexicon import SemProgram, CCGSyntacticType, LexiconEntry
except:
    from lexicon import SemProgram, CCGSyntacticType, LexiconEntry

def enumerate_search(type_dict, function_dict, max_depth=3):
    """
    Enumerate all possible programs up to a given depth, including partial programs.
    
    Args:
        type_dict: Dictionary mapping type names to their representation
        function_dict: Dictionary mapping function names to their specifications
        max_depth: Maximum depth of program expressions to consider
        
    Returns:
        List of tuples (CCGSyntacticType, SemProgram) representing all valid programs and partial programs
    """
    # Initialize cache to store already computed programs for each type
    memo = {}
    
    def get_programs_for_type(target_type, current_depth):
        """
        Recursively generate all programs of a given type up to the specified depth.
        
        Args:
            target_type: The type of programs to generate
            current_depth: Current recursion depth
            
        Returns:
            List of SemProgram objects of the specified type
        """
        # Check memoization cache first
        cache_key = (target_type, current_depth)
        if cache_key in memo:
            return memo[cache_key]
        
        # Base case: we've reached maximum depth
        if current_depth > max_depth:
            return []
        
        results = []
        
        # 1. Add all constants of this type directly
        for func_key, func_info in function_dict.items():
            if func_info['type'] == target_type and not func_info['parameters']:
                # This is a constant of the target type
                program = SemProgram(func_key, [])
                results.append(program)
        
        # Stop if we've reached max depth
        if current_depth == max_depth:
            memo[cache_key] = results
            return results
        
        # 2. Add applications of functions that return this type
        for func_key, func_info in function_dict.items():
            if func_info['type'] != target_type:
                continue
                
            if func_info['parameters']:
                # This function takes parameters
                
                # 2.1. Try to fill in all parameters
                param_options = []
                for param_type in func_info['parameters']:
                    param_programs = get_programs_for_type(param_type, current_depth + 1)
                    param_options.append(param_programs)
                
                # Skip if any parameter has no options
                if all(options for options in param_options):
                    # Generate all combinations of parameter values
                    for param_combo in itertools.product(*param_options):
                        program = SemProgram(func_key, list(param_combo))
                        results.append(program)
                
                # 2.2. Add partial applications with some parameters filled
                if len(func_info['parameters']) > 1:
                    for i in range(len(func_info['parameters'])):
                        param_type = func_info['parameters'][i]
                        param_progs = get_programs_for_type(param_type, current_depth + 1)
                        
                        for param_prog in param_progs:
                            # Create a program with just this parameter filled
                            partial_args = []
                            lambda_vars = []
                            
                            for j in range(len(func_info['parameters'])):
                                if j == i:
                                    partial_args.append(param_prog)
                                else:
                                    # Add the variable name to lambda_vars
                                    var = f"x{j}"
                                    lambda_vars.append(var)
                            
                            # Create the partial program with lambda variables for unfilled parameters
                            partial_program = SemProgram(func_key, partial_args, lambda_vars)
                            results.append(partial_program)
        
        # Save results to memoization cache
        memo[cache_key] = results
        return results
    
    def generate_slash_combinations(num_args):
        """
        Generate all possible combinations of slash directions for a function with num_args arguments.
        
        Args:
            num_args: Number of arguments the function takes
            
        Returns:
            List of tuples, where each tuple contains slash directions ('/' or '\\')
        """
        directions = ['/', '\\']
        return list(itertools.product(directions, repeat=num_args))
    
    def build_complex_type(result_type, param_types, slash_directions):
        """
        Build a properly nested CCG type with the given slash directions.
        
        Args:
            result_type: Return type of the function
            param_types: List of parameter types
            slash_directions: List of slash directions (one per parameter)
            
        Returns:
            A properly nested CCGSyntacticType
        """
        # Start with the result type
        current_type = CCGSyntacticType(result_type)
        
        # Add each parameter with its slash direction, from right to left
        for param_type, direction in zip(reversed(param_types), reversed(slash_directions)):
            current_type = CCGSyntacticType(
                "complex",  # Name for complex types
                CCGSyntacticType(param_type),  # Argument type
                current_type,  # Result type
                direction  # Slash direction
            )
        
        return current_type
    
    # Generate all program-type pairs
    all_programs = []
    
    # 1. Add primitive types with their programs
    for type_name in type_dict:
        primitive_type = CCGSyntacticType(type_name)
        programs = get_programs_for_type(type_name, 1)
        
        for program in programs:
            all_programs.append((primitive_type, program))
    
    # 2. Add complex types for functions
    for func_key, func_info in function_dict.items():
        if not func_info['parameters']:
            continue  # Skip constants
            
        result_type = func_info['type']
        param_types = func_info['parameters']
        
        # 2.1. Create lambda abstractions for this function with lambda variables
        lambda_vars = [f"x{i}" for i in range(len(param_types))]
        lambda_program = SemProgram(func_key, [], lambda_vars)
        
        # 2.2. Generate all possible slash combinations
        slash_combinations = generate_slash_combinations(len(param_types))
        
        # 2.3. Create a complex type for each slash combination
        for slashes in slash_combinations:
            complex_type = build_complex_type(result_type, param_types, slashes)
            all_programs.append((complex_type, lambda_program))
        
        # 2.4. Also add partial applications with some arguments filled
        if len(param_types) > 1 and max_depth > 1:
            # Get programs for each parameter type
            param_programs = []
            for param_type in param_types:
                progs = get_programs_for_type(param_type, 2)  # Use depth 2 for args
                param_programs.append(progs)
            
            # For each possible way to fill some (but not all) parameters
            for num_to_fill in range(1, len(param_types)):
                for positions in itertools.combinations(range(len(param_types)), num_to_fill):
                    # Get the remaining parameter types (unfilled)
                    remaining_params = [param_types[i] for i in range(len(param_types)) if i not in positions]
                    
                    # Generate all possible ways to fill the selected positions
                    filling_options = []
                    for pos in positions:
                        filling_options.append(param_programs[pos])
                    
                    # For each combination of fillings
                    for filling in itertools.product(*filling_options):
                        # Create the partially applied program
                        args = []
                        remaining_vars = []
                        
                        for i in range(len(param_types)):
                            if i in positions:
                                # This position is filled
                                idx = positions.index(i)
                                args.append(filling[idx])
                            else:
                                # This position is a lambda variable
                                var = f"x{i}"
                                remaining_vars.append(var)
                        
                        partial_prog = SemProgram(func_key, args, remaining_vars)
                        
                        # Generate all slash combinations for the remaining parameters
                        for slashes in generate_slash_combinations(len(remaining_params)):
                            partial_type = build_complex_type(result_type, remaining_params, slashes)
                            all_programs.append((partial_type, partial_prog))
    
    return all_programs