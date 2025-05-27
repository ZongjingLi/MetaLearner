import torch
import torch.nn as nn
import itertools
from typing import Dict, List, Tuple, Set, Optional, Any

try:
    from .lexicon import SemProgram, CCGSyntacticType, LexiconEntry
except:
    from lexion import SemProgram, CCGSyntacticType, LexiconEntry


def is_subtype_of(subtype, supertype, type_hierarchy=None):
    """
    Check if one type is a subtype of another.
    
    Args:
        subtype: The potential subtype
        supertype: The potential supertype
        type_hierarchy: Optional dictionary defining the type hierarchy
        
    Returns:
        True if subtype is a subtype of supertype, False otherwise
    """
    # Base case: types are identical
    if subtype == supertype: return True
        
    # Any type is the universal supertype
    if supertype == "Any": return True
        
    # If a type hierarchy is provided, check it
    if type_hierarchy and subtype in type_hierarchy:
        if supertype in type_hierarchy[subtype]:
            return True
            
        # Check transitive supertypes
        for direct_super in type_hierarchy[subtype]:
            if is_subtype_of(direct_super, supertype, type_hierarchy):
                return True
                
    return False


def enumerate_search(type_dict, function_dict, max_depth=3, type_hierarchy=None):
    """
    Enumerate all possible programs up to a given depth, including partial programs.
    
    Args:
        type_dict: Dictionary mapping type names to their representation
        function_dict: Dictionary mapping function names to their specifications
        max_depth: Maximum depth of program expressions to consider
        type_hierarchy: Optional dictionary defining the type hierarchy for downcasting
        
    Returns:
        List of tuples (CCGSyntacticType, SemProgram) representing all valid programs and partial programs
    """
    # Add the identity function to the function dictionary if it doesn't exist already
    if 0 and "Id:Misc" not in function_dict:
        function_dict["Id:Misc"] = {
            'type': "Any",  # Output type is Any
            'parameters': ["Any"]  # Input type is Any
        }
    
    # Add type compatibility check function
    def type_compatibility_check(ccg_type1, ccg_type2):
        """
        Check if two CCG syntactic types are compatible according to the type hierarchy.
        
        Args:
            ccg_type1: First CCG type
            ccg_type2: Second CCG type
            
        Returns:
            True if the types are compatible, False otherwise
        """
        # If either is None, they're not compatible
        if ccg_type1 is None or ccg_type2 is None:
            return False
        
        # If they're the same type, they're compatible
        if ccg_type1.name == ccg_type2.name:
            return True
        
        # If either is a primitive type, check type hierarchy
        if ccg_type1.name != "complex" and ccg_type2.name != "complex":
            return type_hierarchy and is_subtype_of(ccg_type1.name, ccg_type2.name, type_hierarchy)
        
        # If one is complex and one isn't, they're not compatible
        if ccg_type1.name != "complex" or ccg_type2.name != "complex":
            return False
        
        # For complex types, recursively check compatibility of components
        # Result types should be compatible
        if not type_compatibility_check(ccg_type1.result_type, ccg_type2.result_type):
            return False
        
        # Argument types should be compatible (in the opposite direction - contravariance)
        if not type_compatibility_check(ccg_type2.arg_type, ccg_type1.arg_type):
            return False
        
        # Slash directions should match
        return ccg_type1.slash == ccg_type2.slash
    
    memo = {}  # Cache to store already computed programs for each type
    
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
        
        results = []  # Initialize results list
        
        # Add all constants of this type directly
        results.extend(find_matching_constants(target_type))
        
        # Stop if we've reached max depth
        if current_depth == max_depth:
            memo[cache_key] = results
            return results
        
        # Add applications of functions that return this type
        results.extend(find_function_applications(target_type, current_depth))
        
        # Special handling for Id:Misc when target_type is not "Any"
        # We want to generate Id:Misc applications with the specific target type
        if target_type != "Any" and "Id:Misc" in function_dict:

            # For each program of this type, generate its Id:Misc application
            for prog in get_programs_for_type(target_type, current_depth + 1):
                results.append(SemProgram("Id:Misc", [prog]))
        
        # Store in memo cache and return
        memo[cache_key] = results
        return results
    
    def find_matching_constants(target_type):
        """
        Find all constant functions (no parameters) that return the target type.
        
        Args:
            target_type: The type to match
            
        Returns:
            List of SemProgram objects representing constants of the target type
        """
        constants = []
        for func_key, func_info in function_dict.items():
            # Match exact type or check type compatibility if hierarchy provided
            if (func_info['type'] == target_type or 
                (type_hierarchy and is_subtype_of(func_info['type'], target_type, type_hierarchy))) and not func_info['parameters']:
                # This is a constant of the target type
                constants.append(SemProgram(func_key, []))
        return constants
    
    def find_function_applications(target_type, current_depth):
        """
        Find all possible function applications that result in the target type.
        
        Args:
            target_type: The return type to match
            current_depth: Current recursion depth
            
        Returns:
            List of SemProgram objects representing function applications
        """
        applications = []
        
        for func_key, func_info in function_dict.items():
            # Special handling for Id:Misc function
            if func_key == "Id:Misc":
                # For Id:Misc, we match any target type
                # We'll handle it directly in get_programs_for_type
                continue
                
            # Check if function's return type is compatible with target type
            if not (func_info['type'] == target_type or 
                   (type_hierarchy and is_subtype_of(func_info['type'], target_type, type_hierarchy))) or not func_info['parameters']:
                continue
            
            # Get options for each parameter
            param_options = []
            for param_type in func_info['parameters']:
                # For parameters, we need to match in reverse: find programs that can be used as the parameter type
                if type_hierarchy:
                    # Find all subtypes that can be used as this parameter type
                    compatible_types = set([param_type])
                    for t in type_dict:
                        if is_subtype_of(t, param_type, type_hierarchy):
                            compatible_types.add(t)
                    
                    # Collect programs for all compatible types
                    all_programs = []
                    for comp_type in compatible_types:
                        all_programs.extend(get_programs_for_type(comp_type, current_depth + 1))
                    param_options.append(all_programs)
                else:
                    # No type hierarchy, just match exact type
                    param_programs = get_programs_for_type(param_type, current_depth + 1)
                    param_options.append(param_programs)
            
            # Complete function applications (all parameters filled)
            applications.extend(create_complete_applications(func_key, param_options))
            
            # Partial function applications (some parameters filled)
            if len(func_info['parameters']) > 1:
                applications.extend(create_partial_applications(func_key, func_info['parameters'], current_depth))
        
        return applications
    
    def create_complete_applications(func_key, param_options):
        """
        Create complete function applications with all parameters filled.
        
        Args:
            func_key: The function name
            param_options: List of lists of possible values for each parameter
            
        Returns:
            List of SemProgram objects with all parameters filled
        """
        results = []
        
        # Skip if any parameter has no options
        if not all(options for options in param_options):
            return results
        
        # Generate all combinations of parameter values
        for param_combo in itertools.product(*param_options):
            results.append(SemProgram(func_key, list(param_combo)))
        
        return results
    
    def create_partial_applications(func_key, param_types, current_depth):
        """
        Create partial function applications with some parameters filled.
        
        Args:
            func_key: The function name
            param_types: List of parameter types
            current_depth: Current recursion depth
            
        Returns:
            List of SemProgram objects with some parameters filled
        """
        results = []
        
        for i, param_type in enumerate(param_types):
            # Handle type compatibility for parameters
            compatible_param_progs = []
            if type_hierarchy:
                # Find all types that can be used for this parameter
                for t in type_dict:
                    if is_subtype_of(t, param_type, type_hierarchy) or t == param_type:
                        compatible_param_progs.extend(get_programs_for_type(t, current_depth + 1))
            else:
                compatible_param_progs = get_programs_for_type(param_type, current_depth + 1)
            
            for param_prog in compatible_param_progs:
                # Create a program with just this parameter filled
                partial_args = []
                lambda_vars = []
                
                for j in range(len(param_types)):
                    if j == i:
                        partial_args.append(param_prog)
                    else:
                        # Add the variable name to lambda_vars
                        var = f"x{j}"
                        lambda_vars.append(var)
                
                # Create the partial program with lambda variables for unfilled parameters
                results.append(SemProgram(func_key, partial_args, lambda_vars))
        
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
    
    def generate_all_programs():
        """
        Generate all program-type pairs including primitive and complex types.
        
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        all_programs = []
        
        # Add primitive types with their programs
        all_programs.extend(generate_primitive_type_programs())
        
        # Add complex types for functions
        all_programs.extend(generate_complex_type_programs())
        
        # Special handling for Id:Misc as identity function for all types
        all_programs.extend(generate_identity_function_types())
        
        return all_programs
    
    def generate_primitive_type_programs():
        """
        Generate programs for primitive types.
        
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        primitive_programs = []
        
        # Generate programs for each primitive type
        for type_name in type_dict:
            primitive_type = CCGSyntacticType(type_name)
            programs = get_programs_for_type(type_name, 1)
            
            for program in programs:
                primitive_programs.append((primitive_type, program))
            
            # Add programs that can be used as this type through downcasting
            if type_hierarchy:
                for other_type in type_dict:
                    if other_type != type_name and is_subtype_of(other_type, type_name, type_hierarchy):
                        # This type can be downcasted to the target type
                        other_programs = get_programs_for_type(other_type, 1)
                        for program in other_programs:
                            primitive_programs.append((primitive_type, program))
        
        return primitive_programs
    
    def generate_complex_type_programs():
        """
        Generate programs for complex types (functions).
        
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        complex_programs = []
        
        for func_key, func_info in function_dict.items():
            if not func_info['parameters']:
                continue  # Skip constants
                
            # Skip the special Id:Misc function here, we'll handle it separately
            if func_key == "Id:Misc":
                continue
            
            result_type = func_info['type']
            param_types = func_info['parameters']
            
            # Add complete function abstractions
            complex_programs.extend(generate_complete_function_types(func_key, result_type, param_types))
            
            # Add partial applications
            if len(param_types) > 1 and max_depth > 1:
                complex_programs.extend(generate_partial_function_types(func_key, result_type, param_types))
        
        return complex_programs
    
    def generate_identity_function_types():
        """
        Generate type-specific identity functions for all available types.
        
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        id_programs = []
        
        # For each type in the type dictionary
        for type_name in type_dict:
            # Create a type-specific identity function
            id_lambda = SemProgram("Id:Misc", [], ["x0"])  # Lambda function with one variable
            
            # For identity function, both input and output types are the same
            # Create a complex type X -> X for each type
            for slash in ['/', '\\']:
                id_type = CCGSyntacticType(
                    "complex",
                    CCGSyntacticType(type_name),  # Argument type
                    CCGSyntacticType(type_name),  # Result type (same as argument)
                    slash  # Slash direction
                )
                id_programs.append((id_type, id_lambda))
        
        return []#id_programs
    
    def generate_complete_function_types(func_key, result_type, param_types):
        """
        Generate complete function types with lambda abstractions.
        
        Args:
            func_key: The function name
            result_type: Return type of the function
            param_types: List of parameter types
            
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        programs = []
        
        # Create lambda abstractions for this function with lambda variables
        lambda_vars = [f"x{i}" for i in range(len(param_types))]
        lambda_program = SemProgram(func_key, [], lambda_vars)
        
        # Generate all possible slash combinations
        slash_combinations = generate_slash_combinations(len(param_types))
        
        # Create a complex type for each slash combination
        for slashes in slash_combinations:
            complex_type = build_complex_type(result_type, param_types, slashes)
            programs.append((complex_type, lambda_program))
        
        return programs
    
    def generate_partial_function_types(func_key, result_type, param_types):
        """
        Generate partial function types with some arguments filled.
        
        Args:
            func_key: The function name
            result_type: Return type of the function
            param_types: List of parameter types
            
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        programs = []
        
        # Get programs for each parameter type, considering type compatibility
        param_programs = []
        for param_type in param_types:
            if type_hierarchy:
                # Find all types that can be used for this parameter
                compatible_progs = []
                for t in type_dict:
                    if is_subtype_of(t, param_type, type_hierarchy) or t == param_type:
                        compatible_progs.extend(get_programs_for_type(t, 2))  # Use depth 2 for args
                param_programs.append(compatible_progs)
            else:
                progs = get_programs_for_type(param_type, 2)  # Use depth 2 for args
                param_programs.append(progs)
        
        # For each possible way to fill some (but not all) parameters
        for num_to_fill in range(1, len(param_types)):
            for positions in itertools.combinations(range(len(param_types)), num_to_fill):
                programs.extend(
                    generate_partial_types_for_positions(
                        func_key, result_type, param_types, positions, param_programs
                    )
                )
        
        return programs
    
    def generate_partial_types_for_positions(func_key, result_type, param_types, positions, param_programs):
        """
        Generate partial function types with specific positions filled.
        
        Args:
            func_key: The function name
            result_type: Return type of the function
            param_types: List of parameter types
            positions: Tuple of positions to fill
            param_programs: List of lists of possible programs for each parameter
            
        Returns:
            List of tuples (CCGSyntacticType, SemProgram)
        """
        programs = []
        
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
                programs.append((partial_type, partial_prog))
        
        return programs
    
    # Generate and return all program-type pairs
    return generate_all_programs()