"""
Enhanced version of the Lean Structure Extractor with curriculum generation capability.
Generates a structured learning path for Lean structures by hierarchy level.
"""

import json
import re
import os
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional

class CurriculumGenerator(object):
    def __init__(self):
        self.structures = {}  # name -> structure info
        self.dependencies = defaultdict(set)  # structure -> dependencies
        self.reverse_deps = defaultdict(set)  # structure <- dependents
        self.hierarchy_graph = nx.DiGraph()
        self.all_type_names = set()  # Set of all known type names
        self.code_samples = {}  # name -> example usages

    def parse_file(self, filename):
        """Parse a JSONL file containing Lean code corpus."""
        with open(filename, 'r', encoding='utf-8') as f:
            # First pass: collect all structure/class names
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'premises' in entry:
                        for premise in entry['premises']:
                            if premise['kind'] == 'commanddeclaration':
                                full_name = premise['full_name']
                                name = full_name.split('.')[-1]
                                self.all_type_names.add(name)
                except Exception:
                    pass
            
            # Reset file pointer for second pass
            f.seek(0)
            
            # Second pass: extract structures and dependencies
            for line in f:
                try:
                    self._process_json_entry(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
                except Exception as e:
                    print(f"Error processing line: {str(e)}")

    def _process_json_entry(self, entry):
        """Process a single JSON entry from the corpus."""
        if 'premises' not in entry:
            return
        
        for premise in entry['premises']:
            if premise['kind'] == 'commanddeclaration' and 'code' in premise:
                code = premise['code']
                # Check if this is a structure or class definition
                if self._is_structure_or_class_def(code):
                    self._extract_structure(premise['full_name'], code)
                
                # Collect example usages for structures/classes
                self._collect_usage_examples(premise['full_name'], code)

    def _is_structure_or_class_def(self, code):
        """Check if a code block defines a structure or class."""
        # Pattern matches structure, class, class structure, or @[attributes] variants
        pattern = r'^\s*(@\[[^\]]+\]\s*)?(structure|class(\s+structure)?)\s+'
        return re.search(pattern, code, re.MULTILINE) is not None

    def _collect_usage_examples(self, full_name, code):
        """Collect example usages for structures/classes."""
        name_parts = full_name.split('.')
        structure_name = name_parts[-1]
        
        if structure_name in self.all_type_names:
            # Look for usage examples in the code
            usage_examples = []
            
            # Example 1: Constructor usage
            constructor_pattern = r'new\s+' + re.escape(structure_name) + r'\s*{([^}]*)}'
            constructor_matches = re.findall(constructor_pattern, code)
            if constructor_matches:
                for match in constructor_matches[:2]:  # Limit to first two examples
                    usage_examples.append(f"new {structure_name} {{ {match.strip()} }}")
            
            # Example 2: Function with structure as parameter
            param_pattern = r'def\s+(\w+).*\([^)]*:\s*' + re.escape(structure_name) + r'[,)]'
            param_matches = re.findall(param_pattern, code)
            if param_matches:
                for match in param_matches[:2]:
                    usage_examples.append(f"def {match}(...) ... : {structure_name} ...")
            
            # Store the examples if we found any
            if usage_examples:
                if structure_name not in self.code_samples:
                    self.code_samples[structure_name] = []
                self.code_samples[structure_name].extend(usage_examples)

    def _extract_structure(self, full_name, code):
        """Extract structure information and dependencies."""
        # Extract structure name
        name_parts = full_name.split('.')
        structure_name = name_parts[-1]
        namespace = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else ""
        
        # Extract extends clauses
        extends_match = re.search(r'extends\s+([^:,{]+)', code)
        extends = []
        if extends_match:
            extends_str = extends_match.group(1).strip()
            # Handle complex extends with multiple classes, possibly with parameters
            extends_parts = []
            current_part = ""
            paren_level = 0
            
            for char in extends_str:
                if char == ',' and paren_level == 0:
                    extends_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
            
            if current_part:
                extends_parts.append(current_part.strip())
            
            # Extract base structure names
            for part in extends_parts:
                base_name = part.split(' ')[0].strip()  # Get first word before any parameters
                extends.append(base_name)
        
        # Extract field types that might indicate dependencies
        field_types = []
        field_definitions = {}
        
        # Pattern captures field declarations in the form "field_name : type_expression"
        field_pattern = r'(\w+)\s*:([^:=\n]*(?:\([^)]*\))?[^:=\n]*?)(?::=|$|\n)'
        field_matches = re.findall(field_pattern, code)
        
        for field, type_info in field_matches:
            field_types.append(type_info.strip())
            field_definitions[field] = type_info.strip()
        
        # Extract command syntax by determining structure declaration form
        command_syntax = self._extract_command_syntax(code, structure_name)
        
        # Extract documentation comments
        doc_comments = self._extract_documentation(code)
        
        # Also search for type parameters in the structure declaration
        type_params = re.findall(r'\(\s*(\w+)\s*:\s*(Type|Sort)[^)]*\)', code)
        param_names = [param[0] for param in type_params]
        
        # Try to extract a clean version of the structure definition
        clean_definition = self._extract_clean_definition(code)
        
        # Store the structure information
        self.structures[structure_name] = {
            'full_name': full_name,
            'namespace': namespace,
            'code': code,
            'extends': extends,
            'field_types': field_types,
            'field_definitions': field_definitions,
            'type_params': param_names,
            'command_syntax': command_syntax,
            'documentation': doc_comments,
            'clean_definition': clean_definition
        }
        
        # Add dependency edges
        for ext in extends:
            if ext in self.all_type_names:
                self.dependencies[structure_name].add(ext)
                self.reverse_deps[ext].add(structure_name)

    def _extract_clean_definition(self, code):
        """Extract a clean, readable version of the structure definition."""
        # Remove attributes
        clean_code = re.sub(r'@\[[^\]]+\]\s*', '', code)
        
        # Try to get just the structure/class definition part
        definition_match = re.search(r'((?:structure|class|class\s+structure)[^{]*{[^}]*})', clean_code, re.DOTALL)
        if definition_match:
            definition = definition_match.group(1)
            # Clean up whitespace
            definition = re.sub(r'\s+', ' ', definition)
            definition = re.sub(r'{ ', '{\n  ', definition)
            definition = re.sub(r' }', '\n}', definition)
            return definition
        
        # If we couldn't extract a clean definition, return first few lines
        lines = clean_code.split('\n')
        return '\n'.join(lines[:min(5, len(lines))])

    def _extract_command_syntax(self, code, structure_name):
        """Extract the command syntax for creating/using this structure."""
        # Determine if it's a structure or class
        is_class = re.search(r'^\s*(@\[[^\]]+\]\s*)?class\b', code, re.MULTILINE) is not None
        is_structure = re.search(r'^\s*(@\[[^\]]+\]\s*)?structure\b', code, re.MULTILINE) is not None
        
        if is_class and is_structure:
            return f"class structure {structure_name} ... where"
        elif is_class:
            return f"class {structure_name} ... where"
        elif is_structure:
            return f"structure {structure_name} ... where"
        else:
            return f"def {structure_name} ..."

    def _extract_documentation(self, code):
        """Extract documentation comments from the code."""
        # Look for doc comments
        doc_pattern = r'/--\s*(.*?)\s*-/'
        doc_matches = re.findall(doc_pattern, code, re.DOTALL)
        
        if doc_matches:
            # Clean up documentation
            doc = doc_matches[0].strip()
            # Remove extra whitespace
            doc = re.sub(r'\s+', ' ', doc)
            return doc
        
        return ""

    def build_hierarchy(self):
        """Build a directed graph representing the structure hierarchy."""
        # Create nodes for all structures
        for structure, info in self.structures.items():
            self.hierarchy_graph.add_node(structure, **info)
        
        # Add dependency edges
        for structure, deps in self.dependencies.items():
            for dep in deps:
                if dep in self.structures:
                    self.hierarchy_graph.add_edge(structure, dep)
    
    def get_hierarchy_levels(self) -> Dict[int, List[str]]:
        """Organize structures by their hierarchy level."""
        if not self.hierarchy_graph:
            self.build_hierarchy()
        
        # Use topological sort to properly organize the hierarchy
        try:
            # Get a topological sort of the graph (inverted, since we want base classes first)
            topo_sort = list(reversed(list(nx.topological_sort(self.hierarchy_graph))))
            
            # Group by hierarchy level (distance from root)
            levels = defaultdict(list)
            level_map = {}  # node -> level
            
            # Initialize with roots at level 0
            roots = [node for node in self.hierarchy_graph.nodes if self.hierarchy_graph.out_degree(node) == 0]
            for root in roots:
                level_map[root] = 0
                levels[0].append(root)
            
            # Assign levels based on dependencies
            for node in topo_sort:
                if node in level_map:
                    continue
                
                # Find the maximum level of any dependency
                max_dep_level = -1
                for dep in self.dependencies[node]:
                    if dep in level_map:
                        max_dep_level = max(max_dep_level, level_map[dep])
                
                # Assign level one higher than highest dependency
                current_level = max_dep_level + 1
                level_map[node] = current_level
                levels[current_level].append(node)
            
            return dict(levels)
            
        except nx.NetworkXUnfeasible:
            # Handle cycles in the graph
            print("Warning: Cyclic dependencies detected in structure hierarchy.")
            
            # Fall back to previous method
            roots = [node for node in self.hierarchy_graph.nodes if self.hierarchy_graph.out_degree(node) == 0]
            levels = defaultdict(list)
            
            for node in self.hierarchy_graph.nodes:
                max_level = 0
                for root in roots:
                    try:
                        path_length = len(nx.shortest_path(self.hierarchy_graph, node, root)) - 1
                        max_level = max(max_level, path_length)
                    except nx.NetworkXNoPath:
                        continue
                
                levels[max_level].append(node)
            
            return dict(levels)
    
    def print_hierarchy(self):
        """Print the structure hierarchy in a readable format."""
        levels = self.get_hierarchy_levels()
        
        print("Lean Structure Hierarchy:")
        print("=" * 70)
        
        for level in sorted(levels.keys()):
            print(f"\nLevel {level}:")
            print("-" * 70)
            
            for structure in sorted(levels[level]):
                info = self.structures[structure]
                
                # Format namespace and extends info
                namespace = f" (in {info['namespace']})" if info['namespace'] else ""
                extends = info.get('extends', [])
                extends_str = f" extends {', '.join(extends)}" if extends else ""
                
                print(f"• {structure}{namespace}{extends_str}")
                
                # Show dependents (structures that extend this one)
                if structure in self.reverse_deps:
                    dependents = sorted(self.reverse_deps[structure])
                    if dependents:
                        print(f"  Extended by: {', '.join(dependents)}")
        
        print("\nTotal structures found:", len(self.structures))
        
        # Print statistics
        roots = [node for node in self.hierarchy_graph.nodes if self.hierarchy_graph.out_degree(node) == 0]
        leaves = [node for node in self.hierarchy_graph.nodes if self.hierarchy_graph.in_degree(node) == 0]
        
        print(f"\nHierarchy Statistics:")
        print(f"- Root structures (base types): {len(roots)}")
        print(f"- Leaf structures (most derived): {len(leaves)}")
        print(f"- Total hierarchy levels: {len(levels)}")
        
        # Try to detect significant hierarchies
        try:
            longest_path = max((nx.dag_longest_path(self.hierarchy_graph) for _ in roots if roots), key=len)
            if longest_path:
                print("\nLongest inheritance chain:")
                print(" → ".join(longest_path))
        except (ValueError, nx.NetworkXUnfeasible):
            pass

    def generate_curriculum(self, output_dir="lean_curriculum"):
        """Generate a curriculum with increasing complexity levels."""
        levels = self.get_hierarchy_levels()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a summary file
        with open(os.path.join(output_dir, "00_curriculum_overview.md"), "w", encoding="utf-8") as overview_file:
            overview_file.write("# Lean Structure Hierarchy Curriculum\n\n")
            overview_file.write("This curriculum organizes Lean structures by complexity level, ")
            overview_file.write("helping you learn progressively from fundamental to more complex structures.\n\n")
            
            overview_file.write("## Curriculum Structure\n\n")
            for level in sorted(levels.keys()):
                overview_file.write(f"### Level {level}: {self._get_level_title(level)}\n\n")
                overview_file.write(f"- Structures: {len(levels[level])}\n")
                overview_file.write("- Key structures: ")
                
                # List some key structures for this level
                key_structures = []
                for structure in sorted(levels[level]):
                    deps_count = len(self.reverse_deps.get(structure, []))
                    if deps_count > 0:
                        key_structures.append((structure, deps_count))
                
                key_structures.sort(key=lambda x: x[1], reverse=True)
                if key_structures:
                    overview_file.write(", ".join([s[0] for s in key_structures[:5]]))
                else:
                    overview_file.write("None")
                overview_file.write("\n\n")
            
            # Create curriculum navigation guide
            overview_file.write("## Learning Path\n\n")
            overview_file.write("1. Start with Level 0 structures which form the foundation\n")
            overview_file.write("2. Progress through each level, understanding how each structure builds on previous ones\n")
            overview_file.write("3. For each structure, study:\n")
            overview_file.write("   - Its definition and purpose\n")
            overview_file.write("   - Fields and parameters\n")
            overview_file.write("   - Its relationship to other structures\n")
            overview_file.write("   - Example usage patterns\n\n")
            
            # Generate index of all structures
            overview_file.write("## Structure Index\n\n")
            all_structures = []
            for level, structures in levels.items():
                for structure in structures:
                    all_structures.append((structure, level))
            
            all_structures.sort()
            overview_file.write("| Structure | Level | Namespace | Extends |\n")
            overview_file.write("|-----------|-------|-----------|--------|\n")
            
            for structure, level in all_structures:
                info = self.structures[structure]
                namespace = info.get('namespace', '')
                extends = ", ".join(info.get('extends', []))
                overview_file.write(f"| {structure} | {level} | {namespace} | {extends} |\n")
        
        # Create a directory for each level
        for level in sorted(levels.keys()):
            level_dir = os.path.join(output_dir, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)
            
            # Create a README for this level
            level_readme_path = os.path.join(level_dir, "README.md")
            with open(level_readme_path, "w", encoding="utf-8") as level_readme:
                level_readme.write(f"# Level {level}: {self._get_level_title(level)}\n\n")
                
                # Write level description
                level_readme.write(self._get_level_description(level) + "\n\n")
                
                # List all structures at this level
                level_readme.write("## Structures in this Level\n\n")
                for structure in sorted(levels[level]):
                    info = self.structures[structure]
                    namespace = f" (in {info['namespace']})" if info['namespace'] else ""
                    extends = info.get('extends', [])
                    extends_str = f" extends {', '.join(extends)}" if extends else ""
                    
                    level_readme.write(f"### {structure}{namespace}\n\n")
                    
                    # Add documentation if available
                    if info.get('documentation'):
                        level_readme.write(f"{info['documentation']}\n\n")
                    
                    # Add command syntax
                    level_readme.write("**Command Syntax:**\n")
                    level_readme.write(f"```lean\n{info['command_syntax']}\n```\n\n")
                    
                    # Add inheritance info
                    if extends:
                        level_readme.write(f"**Extends:** {', '.join(extends)}\n\n")
                    
                    # Show dependents
                    if structure in self.reverse_deps:
                        dependents = sorted(self.reverse_deps[structure])
                        if dependents:
                            level_readme.write(f"**Extended by:** {', '.join(dependents)}\n\n")
                    
                    # Add field definitions if available
                    if info.get('field_definitions'):
                        level_readme.write("**Fields:**\n\n")
                        for field, type_info in info['field_definitions'].items():
                            level_readme.write(f"- `{field}`: {type_info}\n")
                        level_readme.write("\n")
                    
                    # Add code examples if available
                    if structure in self.code_samples:
                        level_readme.write("**Usage Examples:**\n\n")
                        level_readme.write("```lean\n")
                        for example in self.code_samples[structure]:
                            level_readme.write(f"{example}\n")
                        level_readme.write("```\n\n")
                    
                    # Add links to detailed structure file
                    level_readme.write(f"[See detailed definition](structures/{structure}.md)\n\n")
            
            # Create a structure directory for this level
            structures_dir = os.path.join(level_dir, "structures")
            os.makedirs(structures_dir, exist_ok=True)
            
            # Create a file for each structure in this level
            for structure in sorted(levels[level]):
                self._generate_structure_file(structure, structures_dir)
        
        print(f"Curriculum generated successfully in {output_dir}")
        return output_dir

    def _get_level_title(self, level):
        """Get a title for a level based on its position in the hierarchy."""
        titles = {
            0: "Foundation Structures",
            1: "Basic Derived Structures",
            2: "Intermediate Structures",
            3: "Advanced Structures",
            4: "Specialized Structures",
            5: "Expert-Level Structures"
        }
        
        return titles.get(level, f"Level {level} Structures")

    def _get_level_description(self, level):
        """Get a description for a level based on its position in the hierarchy."""
        descriptions = {
            0: ("This level contains the fundamental structures that form the foundation of the Lean type system. "
                "These structures don't extend other structures and serve as building blocks for more complex types."),
            
            1: ("These structures directly extend the foundation structures. They introduce additional "
                "functionality while maintaining a relatively simple interface. They are essential for "
                "understanding more complex structures."),
            
            2: ("Intermediate structures extend Level 1 structures and introduce more complex concepts. "
                "They often combine multiple concepts from previous levels."),
            
            3: ("Advanced structures build upon intermediate structures to provide comprehensive "
                "functionality. They often involve sophisticated type relationships and multiple inheritance."),
            
            4: ("These specialized structures serve specific use cases and extend advanced structures. "
                "They often incorporate multiple type parameters and complex type relationships."),
            
            5: ("Expert-level structures represent the most complex parts of the type hierarchy. "
                "They typically involve deep inheritance chains and sophisticated type theory concepts.")
        }
        
        return descriptions.get(level, f"Structures at complexity level {level}.")

    def _generate_structure_file(self, structure_name, output_dir):
        """Generate a detailed file for a specific structure."""
        info = self.structures[structure_name]
        
        file_path = os.path.join(output_dir, f"{structure_name}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {structure_name}\n\n")
            
            # Add full name and namespace
            f.write(f"**Full Name:** `{info['full_name']}`\n\n")
            if info['namespace']:
                f.write(f"**Namespace:** `{info['namespace']}`\n\n")
            
            # Add documentation if available
            if info.get('documentation'):
                f.write(f"## Description\n\n{info['documentation']}\n\n")
            
            # Add the clean definition
            f.write("## Definition\n\n")
            f.write("```lean\n")
            f.write(info['clean_definition'])
            f.write("\n```\n\n")
            
            # Add command syntax
            f.write("## Command Syntax\n\n")
            f.write("```lean\n")
            f.write(info['command_syntax'])
            f.write("\n```\n\n")
            
            # Add inheritance information
            f.write("## Inheritance\n\n")
            extends = info.get('extends', [])
            if extends:
                f.write("### Extends\n\n")
                for ext in extends:
                    f.write(f"- `{ext}`\n")
                f.write("\n")
            
            if structure_name in self.reverse_deps:
                f.write("### Extended by\n\n")
                dependents = sorted(self.reverse_deps[structure_name])
                for dep in dependents:
                    f.write(f"- `{dep}`\n")
                f.write("\n")
            
            # Add field definitions
            if info.get('field_definitions'):
                f.write("## Fields\n\n")
                f.write("| Field | Type |\n")
                f.write("|-------|------|\n")
                for field, type_info in info['field_definitions'].items():
                    f.write(f"| `{field}` | `{type_info}` |\n")
                f.write("\n")
            
            # Add type parameters
            if info.get('type_params'):
                f.write("## Type Parameters\n\n")
                for param in info['type_params']:
                    f.write(f"- `{param}`\n")
                f.write("\n")
            
            # Add usage examples
            if structure_name in self.code_samples:
                f.write("## Usage Examples\n\n")
                f.write("```lean\n")
                for example in self.code_samples[structure_name]:
                    f.write(f"{example}\n")
                f.write("```\n\n")
            
            # Add learning notes
            f.write("## Learning Notes\n\n")
            f.write("- **Complexity Level:** ")
            levels = self.get_hierarchy_levels()
            for level, structures in levels.items():
                if structure_name in structures:
                    f.write(f"{level}\n")
                    break
            
            # Add related structures
            f.write("- **Related Structures:** ")
            related = set()
            related.update(info.get('extends', []))
            related.update(self.reverse_deps.get(structure_name, []))
            
            # Also add "siblings" (structures that extend the same base)
            for ext in info.get('extends', []):
                if ext in self.reverse_deps:
                    siblings = self.reverse_deps[ext]
                    related.update(siblings)
            
            # Remove self from related
            if structure_name in related:
                related.remove(structure_name)
            
            if related:
                f.write(", ".join(sorted(related)))
            else:
                f.write("None")
            f.write("\n\n")
            
            # Add the complete code for reference
            f.write("## Complete Definition\n\n")
            f.write("```lean\n")
            f.write(info['code'])
            f.write("\n```\n")

def main(jsonl_file, output_dir="data/lean_curriculum"):
    generator = CurriculumGenerator()
    
    print(f"Parsing Lean corpus from {jsonl_file}...")
    generator.parse_file(jsonl_file)
    
    print("Building hierarchy...")
    generator.build_hierarchy()
    
    print("Printing hierarchy summary...")
    generator.print_hierarchy()
    
    print(f"Generating curriculum in {output_dir}...")
    curriculum_path = generator.generate_curriculum(output_dir)
    
    print(f"Curriculum generation complete. Output directory: {curriculum_path}")
    return curriculum_path



if __name__ == "__main__":
    import sys
    #if len(sys.argv) != 2:
    #    print("Usage: python enhanced_lean_structure_extractor.py <lean_corpus.jsonl>")
    #    sys.exit(1)
    
    main("data/leandojo_benchmark_4/corpus.jsonl")