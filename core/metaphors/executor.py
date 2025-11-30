'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-23 00:19:33
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-03-23 00:19:35
 # @Description:
'''
import os
from typing import List, Union, Mapping, Dict, Any, Tuple, Callable
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor, FunctionExecutor
from helchriss.knowledge.symbolic import Expression,FunctionApplicationExpression, ConstantExpression, VariableExpression
from helchriss.knowledge.symbolic import LogicalAndExpression, LogicalNotExpression,LogicalOrExpression
from helchriss.knowledge.symbolic import TensorState, concat_states
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import TypeBase, AnyType, INT, FLOAT
from core.metaphors.rewrite import NeuralRewriter, RewriteRule, Frame, pth_file
from helchriss.domain import Domain
from .rewrite import NeuralRewriter, LocalFrame


# this is the for type space, basically a wrapper class
from .types import RuleBasedTransformInferer, fill_hole, infer_mlp_caster
from dataclasses import dataclass
import contextlib
import networkx as nx
from pathlib import Path

__all__ = ["ExecutorGroup", "RewriteExecutor"]

def clamp_list(values, min_val=0.3, max_val=1.0):
    return [max(min(v, max_val), min_val) for v in values]

class UnificationFailure(Exception):
    """
    Exception raised when unification between feature structures fails.
    
    This class captures information about why unification failed,
    which structures failed to unify, and at which feature the failure occurred.
    """
    
    def __init__(self, message=None, feature=None, left_structure=None, right_structure=None):
        """
        Initialize a UnificationFailure exception.
        
        Args:
            message (str): Description of why unification failed
            feature (str): The feature name where unification failed
            left_structure (dict): The left feature structure that failed to unify
            right_structure (dict): The right feature structure that failed to unify
        """
        self.feature = feature
        self.left_structure = left_structure
        self.right_structure = right_structure
        
        # Create a detailed message if none provided
        if message is None:
            if feature:
                message = f"Unification failed at feature '{feature}'"
                if left_structure and right_structure:
                    message += f": Cannot unify '{left_structure.get(feature)}' with '{right_structure.get(feature)}'"
            else:
                message = "Unification failed between incompatible structures"
        
        super().__init__(message)
    
    def __str__(self):
        """Return a string representation of the unification failure."""
        result = self.args[0]
        
        if self.left_structure and self.right_structure:
            result += "\nLeft structure: " + str(self.left_structure)
            result += "\nRight structure: " + str(self.right_structure)
            
        return result
    
    def get_conflict_path(self):
        """
        Return the path to the conflicting feature.
        
        Returns:
            str: A dot-notation path to the conflicting feature
        """
        if not self.feature:
            return None
        
        return self.feature
    
    def is_type_mismatch(self):
        """
        Check if the failure was due to type mismatch.
        
        Returns:
            bool: True if the failure was due to type mismatch
        """
        if not (self.feature and self.left_structure and self.right_structure):
            return False
        
        left_val = self.left_structure.get(self.feature)
        right_val = self.right_structure.get(self.feature)
        
        return (left_val is not None and right_val is not None and 
                type(left_val) != type(right_val))


def value_types(values : List[Value]): return [v.vtype for v in values]

class ExecutorGroup(FunctionExecutor):
    """Storage of Domain Executors or some Extention Functions"""
    def __init__(self, domains : List[Union[CentralExecutor, Domain]]):
        super().__init__(None)  

        self.executor_group = nn.ModuleList([])
        for domain_executor in domains:
            if isinstance(domain_executor, FunctionExecutor): executor = domain_executor
            elif isinstance(domain_executor, Domain): executor = FunctionExecutor(domain_executor)
            else : raise Exception(f"input {domain_executor} is not a Domain or FunctionExecutor")
            self.executor_group.append(executor)

        self.extended_registry = nn.ModuleDict({})
    
    def save_ckpt(self, path: str):
        path = Path(path)
        (path / "domains").mkdir(parents=True, exist_ok=True)
        (path / "extended").mkdir(parents=True, exist_ok=True)
        
        for executor in self.executor_group:
            torch.save(executor.state_dict(), path / "domains" / f"{executor.domain.domain_name}.pth")
        
        for name, module in self.extended_registry.items():
            torch.save(module, path / "extended" / f"{name}.ckpt")

    def load_ckpt(self, path: str):
        path = Path(path)
        for executor_file in (path / "domains").glob("*.pth"):
            name = executor_file.stem
            for executor in self.executor_group:
                if executor.domain.domain_name == name:
                    executor.load_state_dict(torch.load(executor_file,  weights_only=True))
                    break

        for extended_file in (path / "extended").glob("*.pth"):
            name = extended_file.stem
            self.extended_registry[name] = torch.load(extended_file,  weights_only=True)


    def format(self, function : str, domain : str) -> str: return f"{function}:{domain}"

    @staticmethod
    def signature(function : str, types : List[TypeBase]):
        typenames = [f"{tp.typename}" for tp in types]
        type_sign = "->".join(typenames)
        return f"{function}#{type_sign}"

    @staticmethod
    def parse_signature(signature: str) ->Tuple[str, List[TypeBase],TypeBase]:
        parts = signature.split('#')
        if len(parts) != 2: raise ValueError(f"Invalid signature format: {signature}")
    
        function_name = parts[0]
        type_signature = parts[1]
    
        type_specs = type_signature.split('->')
        all_types = []
        
        for type_spec in type_specs:
            type_parts = type_spec.split('-')
            if len(type_parts) != 1: raise ValueError(f"Invalid type specification: {type_spec}")
        
            typename = type_parts[0]
            all_types.append(TypeBase(typename))
        # print( all_types[:-1])
    
        input_types = all_types[:-1]


        output_types = all_types[-1]  # Return as list as requested
        return function_name, input_types, output_types

    def domain_function(self, func : str) -> bool: return ":" in func

    def freeze_extended(self, freeze = True):
        for param in self.extended_registry.parameters():
            param.requires_grad = freeze

    @property
    def functions(self) -> List[Tuple[str, List[TypeBase], TypeBase]]:
        functions = []
        for sign in self.extended_registry:

            f_sign, in_types, out_type = self.parse_signature(sign)
            functions.append([f_sign, in_types, out_type])
        for executor in self.executor_group:
            assert isinstance(executor, FunctionExecutor), f"{executor} is not a executor"
            for func_name in executor._function_registry:

                functions.append([
                    self.format(func_name,executor.domain.domain_name),
                    executor.function_input_types[func_name],
                    executor.function_output_type[func_name]])

        return functions

    def function_signature(self, func : str) -> List[Tuple[List[TypeBase], TypeBase]]:
        hyp_sign = []
        for function in self.functions:
            f_sign, in_types, out_type = function
            if func == f_sign: hyp_sign.append([in_types, out_type])
        return hyp_sign
        
    def gather_functions(self, input_types, output_type : Union[TypeBase, bool]) -> List[str]:
        funcs = []
        for function in self.functions:
            f_sign, in_types, out_type = function
            if in_types == input_types and  out_type == output_type:
                funcs.append(f_sign)
        return funcs

    def register_function(self, func : str, in_types : List[TypeBase], out_type : TypeBase, implement : nn.Module):
        signature = self.signature(func, in_types + [out_type])
        self.extended_registry[signature] = implement

    def infer_domain(self, func: str) -> str:
        for executor in self.executor_group: 
            assert isinstance(executor, FunctionExecutor), "not a function executor"
            if func in executor._function_registry: return executor.domain.domain_name

    def execute(self, func : str, args : List[Value], domain = None, grounding = None) -> Value:
        self._grounding = grounding
        """a function could be in some domain or in extention registry"""
        arg_types = value_types(args)
        signature = self.signature(func, arg_types)
        func_call = None

        # 1) check if this is a domain function
        if self.domain_function(func):
            func_name, domain_name = func.split(":")
            for executor in self.executor_group:
                assert isinstance(executor, FunctionExecutor), f"{executor} is not a executor"
                if executor.domain.domain_name == domain_name: # find the correct executor
                    executor._grounding = grounding
                    func_call = executor._function_registry[func_name]


        # 2) check if this is an extention function
        for sign in self.extended_registry:
            if signature in sign:func_call = self.extended_registry[sign]


        ### collect arguments and evaluate on the function call
        args = [arg.value for arg in args]
        if func_call is not None: return func_call(*args)
        else:raise ModuleNotFoundError(f"{signature} is not found.")


class RewriteExecutor(FunctionExecutor):
    """this is some kind of wrap out of an executor, equipped with a reductive graph"""
    def __init__(self, executor):
        super().__init__(None)
        self.base_executor : FunctionExecutor = executor
        self.rewriter : NeuralRewriter = NeuralRewriter() # use this structur to store the caster and hierarchies
        self.inferer  = RuleBasedTransformInferer()
        self._gather_format = "{}:{}"

        """maintain the evaluation graph just to visualize and sanity check"""
        self.eval_graph : nx.DiGraph = nx.DiGraph()
        self.node_count = {}
        self.record = 1

    def init_graph(self):
        self.eval_graph = nx.DiGraph()
        self.node_count = {}
        self.record = 1

    def save_ckpt(self, ckpt_path = "tmp.ckpt"):
        self.base_executor.save_ckpt(ckpt_path)
        self.rewriter.save_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path = "tmp.ckpt"):
        self.base_executor.load_ckpt(ckpt_path)
        self.rewriter.load_ckpt(ckpt_path)
        return self

    def gather_format(self, name, domain): return self._gather_format.format(name, domain)
    
    @staticmethod
    def format(function : str): return function.split(":")[0]

    def infer_rewrite_expr(self, expr : Expression, reducer = None) -> List[Tuple[str, List[TypeBase], List[TypeBase]]]:
        """given an expression, use the unifer to infer if there exist casting of types or change in the local frames"""
        metaphor_exprs = []
        def dfs(expr : Expression):
            if isinstance(expr, FunctionApplicationExpression):
                func_name = expr.func.name

                source_arg_types : List[TypeBase] = [dfs(arg)[0]        for arg in expr.args] # A List of Types

                target_signatures = self.base_executor.function_signature(func_name)

                min_mismatch = len(source_arg_types) + 1
                best_matched = None # find the best matched function signature
                assert len(target_signatures) != 0,f"did not find any function {func_name}"

                for hyp_sign in target_signatures:

                    arg_types, out_type = hyp_sign
                    
                    mismatch_count = 0
                    for i,tp in enumerate(arg_types):
                        if source_arg_types[i].typename != arg_types[i].typename:
                            mismatch_count += 1
                    if mismatch_count < min_mismatch:
                        min_mismatch = mismatch_count
                        best_matched = hyp_sign


                if not min_mismatch == 0:
    
                    metaphor_exprs.append([func_name, best_matched[0], source_arg_types, best_matched[1]])
                output_type = best_matched[1]
    
                #print(output_type.alias, output_type.typename)
                return [output_type, best_matched[0]]
            
            else: raise NotImplementedError(f"did not write how to infer from {expr}")
        dfs(expr)

        return metaphor_exprs
    
    def add_metaphors(self, metaphors : List[Tuple[str, List[TypeBase], List[TypeBase]]], caster = None):
        if not isinstance(metaphors, List) : metaphors = [metaphors]
        output_metaphors = []
        for metaphor in metaphors:
            target_func, target_types, source_types, out_type = metaphor

            input_type  = source_types  #(y) self.function_input_type(*reduce_func.split(":"))      # actual input type for the function
            expect_type = target_types  #(x) expect input type for the function
            output_type = out_type      #(o) the output type for the target function

            ### 1) create the type casting rewrite rule and add a NeuralNet to fill the hole
            filler = fill_hole(input_type, output_type)
            self.base_executor.register_function(target_func, input_type, output_type, filler)
            output_metaphors.append(metaphor) ### add the extention of fill-hole
            
            ### 2) create the local frame that gathers other `source` functions to the `target` function

            if caster is None: caster = self.inferer.infer_caster(input_type, expect_type)
            rewrite_frame : LocalFrame = LocalFrame(target_func, expect_type, input_type, caster)

            reduce_hypothesis = self.base_executor.gather_functions(input_type, output_type)
            for reduce_func in reduce_hypothesis:

                rewrite_frame.add_source_caster(reduce_func, 0.0) # init the reduction `g`->`f` weight logits 0.0
                output_metaphors.append([reduce_func, target_types, source_types, out_type])

            hash_frame = target_func + str(hash((tuple(input_type) + tuple(expect_type))))
            
            self.rewriter.add_frame(hash_frame, rewrite_frame) # multiple frame lead to the same procedure

        return output_metaphors
    
    @property
    def types(self): return self.base_executor.types

    @property
    def functions(self): return self.base_executor.functions

    def function_out_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")

        return self.base_executor.function_out_type(func_name, domain)

    def function_input_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")
        return self.base_executor.function_input_type(func_name, domain)

    def display(self, fname=None):
        """Display the computational graph with proper visualization of nodes and edges."""
        import matplotlib.pyplot as plt
        import networkx as nx
        from networkx import spring_layout
        import numpy as np
        
        G = check_graph_node_names(self.eval_graph)
        H = G.copy()
        for n, d in H.nodes(data=True):
            # Keep only basic attributes for layout
            attrs_to_remove = ['inputs', 'output', 'args', 'weight']
            for attr in attrs_to_remove:
                d.pop(attr, None)
        
        for u, v, d in H.edges(data=True):
            d.pop('output', None)

    
        #from networkx.drawing.nx_agraph import graphviz_layout
        pos = balanced_tree_pos(H)

        #pos = hierarchy_pos(H)
        #pos = radial_tree_pos(H)
        #pos = nx.spring_layout(H, k=14, iterations=150, seed=42)
        #pos = graphviz_layout(G, prog='dot')  # Top-down tree
        #pos = graphviz_layout(G, prog='twopi')  # Radial tree
        edge_colors = []
        for u, v, d in G.edges(data=True):
            color = d.get('color', '#cccccc')
            edge_colors.append(color)

        
        node_colors = []
        node_weights = []
        
        for node, data in G.nodes(data=True):
            # Handle node colors
            color = data.get('color', '#1f77b4')  # default blue
            node_colors.append(color)
            
            # Handle node weights (for alpha transparency)
            weight = data.get('weight', 1.0)
            try:
                if hasattr(weight, 'item'):  # torch tensor
                    weight_val = float(weight.item())
                else:
                    weight_val = float(weight)
                # Normalize weight to [0.3, 1.0] range for visibility
                weight_val = max(0.01, min(1.0, abs(weight_val)))
            except (ValueError, TypeError):
                weight_val = 1.0
            node_weights.append(weight_val)
        
        # Create the plot
        plt.figure("Rewrite Computational Graph", figsize=(12, 8))

        

        # Draw nodes with individual colors and transparency
        from collections import OrderedDict
        pos= OrderedDict((node, pos[node]) for node in G.nodes)
        for i, (node, (x, y)) in enumerate(pos.items()):
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[node], 
                node_color=[node_colors[i]], 
                node_size=2000, 
                alpha=  node_weights[i],
                edgecolors='black',
                linewidths=1,
            )

        nx.draw_networkx_edges(
            G, pos, 
            edge_color=edge_colors, 
            arrows=True, 
            arrowsize=15, 
            arrowstyle='-|>',
            width=2,
            alpha=0.7
        )
        
        # Add node labels and information
        for i, (node, (x, y)) in enumerate(pos.items()):
            data = G.nodes[node]
            
            # Node weight label (top)
            weight = data.get('weight', 'N/A')
            try:
                if hasattr(weight, 'item'):
                    weight_str = f"{float(weight.item()):.3f}"
                else:
                    weight_str = f"{float(weight):.3f}"
            except (ValueError, TypeError):
                weight_str = str(weight)
            
            plt.text(x, y + 0.15, weight_str, 
                    fontsize=8, color='white', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
            
            # Node name (center)
            node_text_color = data.get("text_color", "black")
            # Truncate long node names
            display_name = node.split("&")[0]#node if len(str(node)) < 20 else str(node)[:17] + "..."
            plt.text(x, y, display_name, 
                    fontsize=7, color=node_text_color, weight='bold',
                    ha='center', va='center')
            
            # Output information (bottom)
            if 'output' in data and data['output'] is not None:
                output = data['output']
                if 1:
                    # Handle output value
                    if hasattr(output, 'value'):
                        if hasattr(output.value, 'item'):  # torch tensor
                            if sum(list(output.value.shape)) < 9:
                                val_str = f"{output.value}"
                            else:
                                from helchriss.utils import stprint_str
                                val_str = f"{output.value.shape}"
                        else: val_str = str(output.value)
                    else: val_str = str(output)
                    
                    # Handle output type
                    if hasattr(output, 'vtype'):
                        if hasattr(output.vtype, 'alias'):
                            type_str = str(output.vtype)
                        else:
                            type_str = str(output.vtype)
                    else:
                        type_str = "Unknown"
                    
                
                    out_label = f"V: {val_str}\nT: {type_str}"
                    
                
                plt.text(x, y - 0.2, out_label, 
                        fontsize=7, color=node_text_color,
                        ha='center', va='center')
        
        # Add edge labels
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            label_parts = []
            
            # Add weight if available
            if 'weight' in d:
                try:
                    weight = d['weight']
                    if hasattr(weight, 'item'):
                        weight_val = float(weight.item())
                    else:
                        weight_val = float(weight)
                    label_parts.append(f"W: {weight_val:.3f}")
                except (ValueError, TypeError):
                    label_parts.append(f"W: {d['weight']}")
            
            # Add output info if available
            if 'output' in d and d['output'] is not None:
                try:
                    output = d['output']
                    if hasattr(output, 'value'):
                        if hasattr(output.value, 'item'):
                            val_str = f"{float(output.value.item()):.3f}"
                        else:
                            val_str = str(output.value)[:10]
                        label_parts.append(f"V: {val_str}")
                except:
                    pass
            
            if label_parts:
                edge_labels[(u, v)] = '\n'.join(label_parts)
        
        # Draw edge labels with better positioning
        if 1 and edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, 
                edge_labels=edge_labels, 
                font_size=7,
                font_color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
        
        # Style the plot
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        
        # Remove axes
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False, 
                       labelbottom=False, bottom=False, top=False)
        
        plt.title("Computational Graph Visualization", fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save if filename provided
        if fname is not None:
            plt.savefig(f"{fname}.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            #print(f"Graph saved as {fname}.png")
        
        plt.show()
        return G, pos  # Return graph and positions for further use if needed

    def evaluate(self, expression, grounding):
        self.init_graph()
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)
        #print(expression)
        grounding = grounding if self._grounding is not None else grounding

        with self.with_grounding(grounding):
            outputs, out_name = self._evaluate(expression)

            self.eval_graph.add_node("outputs", weight = 1.0, inputs = outputs, color = "#0d0d0d", output = outputs)
            self.eval_graph.add_edge(out_name, "outputs", output = outputs, color = "#0d0d0d")

            return outputs
        
     
    def _evaluate(self, expr : Expression):
        """Internal implementation of the executor. This method will be called by the public method :meth:`execute`.
        This function basically implements a depth-first search on the expression tree.
        Args:
            expr: the expression to execute.

        Returns:
            The result of the execution.
        """

        if isinstance(expr, FunctionApplicationExpression):
            func_name = expr.func.name
            
            # recusive call self.evaluate(arg) to evaluate the args in the subtree
            args : List[Value] = []
            arg_names : List[str] = []
            for arg in expr.args: 
                arg_value, arg_name = self._evaluate(arg) # A List of Values
                args.append(arg_value)
                arg_names.append(arg_name)
            
            arg_types : List[TypeBase] = [arg.vtype for arg in args]
            sign = self.base_executor.signature(func_name, arg_types)


            count_func_sign = self.add_count_function_node(sign)
            self.eval_graph.add_node(count_func_sign)


            for arg_n in arg_names:
                self.eval_graph.add_edge(arg_n, count_func_sign, output = arg_value, color = "#0d0d0d")

            # weight of each rewrite is a basic-rewrite
            rewrite_distr, rewrite_graph = self.rewriter.rewrite_distr(func_name, args)

            self.add_rewrite_subgraph(rewrite_distr, rewrite_graph, sign)
            
            # expected execution over all basic-rewrites
            expect_output = 0.

            for (t_f, t_args, weight) in rewrite_distr:
                measure : Value = self.base_executor.execute(t_f.split("#")[0], t_args, grounding = self.grounding)

                expect_output += weight * measure.value

                ### add the output value for the evaluation graph
                func_sign = t_f
                node_count = self.node_count[func_sign]
                func_count_sign = f"{func_sign}_{node_count}"
                self.eval_graph.nodes[func_count_sign]["output"] = measure          


            return Value(measure.vtype, expect_output), count_func_sign

        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const
        elif isinstance(expr, VariableExpression):
            assert isinstance(expr.name, Value)
            return expr.const
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')
    
    

    def add_count_function_node(self, sign):
        nd = sign#self.format(node)

        if sign in self.node_count: self.node_count[sign] += 1
        else: self.node_count[sign] = 1
        #self.eval_graph.add_node(f"{nd}#{self.node_count[node]}")

        return f"{nd}_{self.node_count[sign]}"

    def add_rewrite_subgraph(self, distr, rewrite_graph, func_sign):
        reduce_funcs = distr
        _, reduce_edges = rewrite_graph
        node_count = self.node_count[func_sign]

        for reduce_func in reduce_funcs:
            node_sign, node_args, node_weight = reduce_func
            if node_sign == func_sign: # attach point is the origion of rewrite
                func_count_sign = f"{func_sign}_{node_count}"

                self.eval_graph.nodes[func_count_sign]["weight"] = node_weight
                self.eval_graph.nodes[func_count_sign]["args"]   = node_args
            
            else: # create new node for args
                node_count_sign = self.add_count_function_node(node_sign)

                self.eval_graph.add_node(node_count_sign, weight = node_weight, args = node_args, color = "#048393", text_color = '#0d0d0d')
            
        for edge in reduce_edges:
            src_node = f"{edge[0]}_{self.node_count[edge[0]]}"
            tgt_node = f"{edge[1]}_{self.node_count[edge[1]]}"

            if src_node != func_sign:
                    self.eval_graph.add_edge(tgt_node, src_node, weight = float(edge[2].detach()), color = "#048393")



@dataclass
class SearchNode:
    """not the frame search node, the eval search node"""
    fn    : str               # executable fn on value
    value : List[Value]       # the value of the node, args of some function
    src   : None              # the source rewriter
    maintain : bool = None           # the virtual node
    next = []
    next_weights = []
    distr : None = None                    # the distribution value on the search node
    reachable : None  = None           # whether the node is reachable



class SearchVisualizer:
    """Visualizer for logging and graphing search tree operations"""
    def __init__(self):
        self.query_trees = []
        self.active = 1  # Using int for boolean (1 = active, 0 = inactive)

    def log(
        self,
        query_fn: str,
        args: List[Value],
        nodes: List[SearchNode],
        mode: str
    ) -> None:
        """
        Log search tree data if the visualizer is active.
        
        Args:
            query_fn: Name of the query function
            args: List of Value objects passed to the query
            nodes: List of root SearchNodes for the search tree
            mode: Operation mode (e.g., "search", "expand", "evaluate")
        """
        if self.active:
            self.query_trees.append((query_fn, args, nodes, mode))

    def create_graph(self, nodes: List[SearchNode]) -> nx.DiGraph:
        """
        Create a directed graph from a list of SearchNodes using DFS traversal.
        
        Args:
            nodes: List of root SearchNodes to build the graph from
        
        Returns:
            nx.DiGraph: NetworkX directed graph with node/edge metadata
        """
        graph = nx.DiGraph()

        def dfs(current_node: SearchNode) -> None:
            """Recursive DFS helper to traverse SearchNodes and build the graph"""
            # Ensure value is a list (fallback for None)
            if current_node.value is None:
                current_node.value = []

            # Build node label with function name, value types, and rounded values
            vtypes = [str(v.vtype) for v in current_node.value]
            rounded_values = [
                str((torch.round(torch.tensor(v.value) * 100) / 100).detach().numpy())
                for v in current_node.value
            ]
            node_label = f"{current_node.fn}\n{vtypes}\n{rounded_values}"

            # Add node with metadata (use string representation of node as unique ID)
            graph.add_node(
                str(current_node),
                fn=current_node.fn,
                value=[v.value for v in current_node.value],
                label=node_label
            )

            # Traverse child nodes and add edges with weights
            for next_node, weight in zip(current_node.next, current_node.next_weights):
                dfs(next_node)
                # Add edge with rounded weight (convert tensor to numpy if needed)
                edge_weight = torch.tensor(weight).detach().numpy() if not isinstance(weight, (int, float)) else weight
                graph.add_edge(str(current_node), str(next_node), weight=edge_weight)

        # Start DFS from the first node (assuming nodes list has one root)
        if nodes:
            dfs(nodes[0])

        return graph

class SearchExecutor(FunctionExecutor):
    """stores local rewrite frames then """
    def __init__(self, executor):
        super().__init__()
        self.base_executor : FunctionExecutor = executor
        self.rewrite_frames = nn.ModuleDict({}) # a bundle of rewriter rules that shares the same rewriter
        self.storage = SearchVisualizer()
        self.unification_p = 0.001
        self.supressed = 0

    """Save and Load Utils and Add Frames"""
    def save_ckpt(self, ckpt_path) -> int:
        if not os.path.exists(f"{ckpt_path}/frames"): os.makedirs(f"{ckpt_path}/frames")
        for frame_name in self.frames: torch.save(self.frames[frame_name], f"{ckpt_path}/frames/{frame_name}.pth")#.save_ckpt(f"{ckpt_path}/frames/{frame_name}")
        self.base_executor.save_ckpt(ckpt_path)
        return self

    def load_ckpt(self, ckpt_path) -> int:
        frames_dir = f"{ckpt_path}/frames"
        for filename in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, filename)
            if os.path.isfile(file_path) and pth_file(filename):
                self.frames[filename[:-4]] = torch.load(file_path, weights_only = False)
        self.base_executor.load_ckpt(ckpt_path)
        return self

    def add_frame(self,name : str, frame : Frame):
        """raw method of adding a frame to the dictionary"""
        self.rewrite_frames[name] = frame

    """Rewrite Search Graph Implementations"""
    def edges(self, node : SearchNode) -> List[SearchNode]:
        src_fn  = node.fn
        src_val = node.value
        src_tp  = value_types(src_val)
        nodes = []
        edges = []
        for key, frame in self.rewrite_frames.items():
            assert isinstance(frame, Frame), f"{frame} is not a `Frame`"

            if frame.source_type == src_tp:
                nxt_val, nxt_pr = frame.apply(src_val) # applied value and pr
                """enumerate matches that start with src_fn"""
                for fn_gn in frame.matches:
                    (fn, gn) = fn_gn.split("@")
                    if fn == src_fn:
                        fn_logit = frame.matches[fn_gn] # function match logit
                        nodes.append(SearchNode(gn, nxt_val, None))
                        edges.append(torch.sigmoid(nxt_pr + fn_logit))
        return nodes, edges

    def rewrite_search_tree(self, init_node : SearchNode,  max_iters = 100) :
        """graph start with value and end with fn by rewrites"""
        nodes : List[SearchNode] = []
        edges : List             = []

        #. start the bfs queue
        itrs = 0
        done = False
        queue : List[SearchNode] = [init_node]
        while not done:
            curr_node = queue.pop(0)
            frontier_nodes, frontier_edges = self.edges(curr_node)
            for node, edge_weight in zip(frontier_nodes, frontier_edges):
                # node is the nodes the curr_node connected ; edge is a weight
                assert isinstance(node, SearchNode), f"{node} not a search node"
                queue.append(node)
                curr_node.next.append(node)
                curr_node.next_weights.append(edge_weight)
                edges.append((curr_node, node, edge_weight))
            nodes.append(curr_node)
            itrs += 1
            done =  len(queue) == 0 or (itrs >= max_iters)
        return nodes
    
    def subtree_filter_target(self, init_node :SearchNode, nodes : List[SearchNode], target : str) -> List[SearchNode]:
        def dfs(node : SearchNode):
            node.maintain = 0
            for son in node.next:
                dfs(son)
                assert isinstance(son, SearchNode), f"{son} is not SearchNode"
                if son.maintain:
                    node.maintain = 1
                    break
            if node.fn == target: node.maintain = 1

        def subtree_select(node : SearchNode, df : int):
            if df: node.maintain = 1
            for son in node.next:
                assert isinstance(son, SearchNode), f"{son} is not SearchNode"
                if node.fn == target:
                    son.maintain = 1
                    subtree_select(son, 1)
        
        dfs(init_node) # select the parent of the target node
        subtree_select(init_node, 0) # select the whole subtree of the target node

        return [node for node in nodes if node.maintain == 1]

    def subtree_filter_margin(self, init_node : SearchNode, nodes : List[SearchNode], margin = 0.1):
        def dfs(node : SearchNode):
            for son, weight in zip(node.next, node.next_weights):
                assert isinstance(son, SearchNode)
                son.reachable = node.reachable * weight
                if son.reachable >= margin:
                    son.maintain = 1
                    dfs(son)
                else: son.maintain = 0
        init_node.reachable = 1.0
        dfs(init_node)
        subtree_nodes = [node for node in nodes]
        return subtree_nodes

    def rewrite_distr(self, value : List[Value], query_fn : str, mode = "eval") :
        """ return the rewrite paths from fn to target value
        Args:
            value : the input value to evaluate on
            fn    : the target fn to locate and evaluate
        Return:
            a distribution over search nodes
        """
        src_node = SearchNode("super_source", None, None, None, reachable = 1.)
        src_node.next = []
        src_node.next_weights = []
        rw_nodes : List[SearchNode] = [src_node]

        output_type = self.base_executor.function_signature(query_fn)[0][-1]
        functions = self.base_executor.gather_functions(value_types(value), output_type)
        for fn in functions:
            nd = SearchNode(fn, value, None)
            nd.next = []
            nd.next_weights = []
            subtree_nodes = self.rewrite_search_tree(nd)
            if mode == "eval": # select the subtree contains the query fn
                subtree_nodes = self.subtree_filter_target(subtree_nodes[0], subtree_nodes, query_fn)
            if mode == "update": # select the subtree that is reachable by margin
                subtree_nodes = self.subtree_filter_margin(subtree_nodes[0], subtree_nodes, margin = 0.01)
    
            if subtree_nodes: # not an empty subtree
                src_node.next.append(subtree_nodes[0]) # add the subtree head
                src_node.next_weights.append(1.)       # add the transition 1
                rw_nodes.extend(subtree_nodes)


        def dfs(node : SearchNode, success_reach):
            # the node is reachable and have no applicable rewrite.
            if node.next_weights:
                node.distr = node.reachable * (1. - torch.max(torch.tensor(node.next_weights)))
            else:
                node.distr = node.reachable
            for son, weight in zip(node.next, node.next_weights):
                assert isinstance(son, SearchNode)
                son.reachable = node.reachable * weight
                if son.fn == query_fn :
                    success_reach = max(success_reach, son.reachable)
                success_reach = max(success_reach,dfs(son, success_reach))
            return success_reach

        success_reach = dfs(src_node, 0.) # dfs on the super source node

        self.storage.log(query_fn, value, rw_nodes, mode)

        return [(node.fn, node.value, node.distr) for node in rw_nodes], success_reach

    def reduction_call(self, fn : str , args : List[Value]):
        """ evaluate the fn calls on the distribution over all possible rewrites
        start with the init_node that contains the value and the 
        1. locate each node that terminates, find the subtree that trace back to the init_node
        2. for the subtree, evaluate the expectation over the subtree distribution
        """        
        rewrite_pairs, success_reach = self.rewrite_distr(args, fn, mode = "eval")

        execute_pairs = rewrite_pairs[1:]
        if not isinstance(success_reach, torch.Tensor): success_reach = torch.tensor(success_reach)
        if success_reach < self.unification_p and not self.supressed:
            raise UnificationFailure(f"failed to unify {fn}({args}->{success_reach})", 
                                     left_structure = fn,
                                     right_structure = args)

        expect_output = 0.
        for (f, vargs, weight) in execute_pairs:
            measure : Value = self.base_executor.execute(f, vargs, grounding = self.grounding)

            if isinstance(measure.value, torch.Tensor) and (measure.vtype == INT or measure.vtype == FLOAT):

                expect_output += measure.value.reshape([-1])[0] * weight
            else:expect_output += measure.value * weight
        
        return Value(measure.vtype, expect_output), -torch.log(success_reach)

    """Evaluate Chain of Expressions"""
    def evaluate(self, expression, grounding):
        self.storage
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)
        grounding = grounding if self._grounding is not None else grounding
        with self.with_grounding(grounding):
            outputs, loss = self._evaluate(expression)
        return outputs, loss
    
    def _evaluate(self, expr : Expression) -> Tuple[Value, str]:
        if isinstance(expr, FunctionApplicationExpression):
            # recusive call self.evaluate(arg) to evaluate the args in the subtree
            fn = expr.func.name
            args : List[Value] = []
            arg_loss  = []
            for arg in expr.args: 
                arg_value, subloss = self._evaluate(arg) # A List of Values
                args.append(arg_value)
                arg_loss.append(subloss)
            arg_types : List[TypeBase] = [arg.vtype for arg in args]
            
            output, loss = self.reduction_call(fn, args)
            return output, sum(arg_loss) + loss
        
        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const, 0.
        elif isinstance(expr, VariableExpression):
            assert isinstance(expr.name, Value)
            return expr.const, 0.
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')
    

    """Update Search Rules and Purge/Merge Entries""" 
    def update_chain(self, args : List[Value], fn : str, topK = 4):
        """assert exist a path from the arg to the fn located space by some frames
        add frames and update fn on the margin types.
        Args:
            args : List[TypeBase] as the input arg types for the fn
            fn   : str as the input function name (specific to domain tag)
        """
        # 1. add rewriters that locate from the value node to the fn node.
        execute_pairs, reach_loss = self.rewrite_distr(args, fn, mode = "update")
        execute_pairs = execute_pairs[1:]

        # for each node that not in the 
        for (gn, vargs, weight) in execute_pairs:
            for (in_tps, out_tps) in self.base_executor.function_signature(fn):
  
                arg_types = value_types(vargs)

                src_tps = arg_types
                tgt_tps = in_tps 
    
                caster = infer_mlp_caster(src_tps, tgt_tps)
                learn0_frame = Frame(src_tps, tgt_tps, caster)
                #print(learn0_frame)
                learn0_frame.matches[(f"{gn}@{fn}")] = torch.tensor(0.)

                src_sig = 'x'.join([str(tp) for tp in src_tps])
                tgt_sig = 'x'.join([str(tp) for tp in tgt_tps])
                frame_sig = f"frame:{src_sig}->{tgt_sig}_"
                id_itr = 0
                done = 0
                while not done:
                    if (frame_sig + str(id_itr)) in self.rewrite_frames:id_itr += 1
                    else: done = 1
                print(f"{gn}@{fn} by",frame_sig + str(id_itr))
                self.rewrite_frames[frame_sig + str(id_itr)] = learn0_frame

        # 2. add the filler that defined on the value node.

        return self
    
    def purge_frames(self, p = 0.1):
        """purge the the last p percent of all the newly added connection
        the last p percent of the newly added frames need to be removed.
        Args:
            p : the percent of newly added frames needed to be removed
        """
        return self
    
    def merge_frames(self, p = 0.1):
        """merge frames that shares the     same rewriter, distinguish rewriter using subsets of data
        Args:
            p : the metric of how close two rewriter need to be merged together
        merge two frames with shared rewriter and combine the 
        """
        return self

    def additive_evaluation(self, query, grounding = {}):
        try:
            out = self.evaluate(query, grounding = grounding)
        except UnificationFailure as ue:
            query_fn = ue.left_structure
            value = ue.right_structure
            self.update_chain(value, query_fn)
            print("update chain")
            print(ue)
            out = self.evaluate(query, grounding = grounding)
        print(out)

import networkx as nx

def convert_graph_to_visualization_data(G):
    """
    Convert a NetworkX graph to visualization data format needed for D3.js
    
    Args:
        G: NetworkX graph (eval_graph from the model)
    
    Returns:
        dict: JSON-serializable dictionary with nodes and edges data
    """
    # Initialize nodes and edges lists
    nodes = []
    edges = []
    
    # Process nodes
    for node_id, node_data in G.nodes(data=True):
        # Create node entry
        node = {
            "id": str(node_id),
            "weight": float(node_data.get('weight', 1.0)),
            "color": node_data.get('color', '#3a5f7d'),
            "text_color": node_data.get('text_color', 'white')
        }
        
        # Add output information if available
        if 'output' in node_data:
            output_data = node_data['output']
            output_info = {
                "value": float(output_data.value) if hasattr(output_data, 'value') else str(output_data),
                "vtype": output_data.vtype.alias.split(':')[0] if hasattr(output_data.vtype, 'alias') else str(output_data.vtype)
            }
            node["output"] = output_info
        
        nodes.append(node)
    
    # process edges
    for source, target, edge_data in G.edges(data=True):
        # Create edge entry
        edge = {
            "source": str(source),
            "target": str(target),
            "weight": float(edge_data.get('weight', 1.0)),
            "color": edge_data.get('color', '#555555')
        }
        
        # Add output information if available
        if 'output' in edge_data:
            output_data = edge_data['output']
            output_info = {
                "value": float(output_data.value) if hasattr(output_data, 'value') else str(output_data),
                "vtype": output_data.vtype.alias.split(':')[0] if hasattr(output_data.vtype, 'alias') else str(output_data.vtype)
            }
            edge["output"] = output_info
        
        edges.append(edge)
    
    import matplotlib.pyplot as plt
    #nx.draw(G)
    #plt.show()

    return {
        "nodes": nodes,
        "edges": edges,
   
    }

def check_graph_node_names(graph):
    """
    Modify a NetworkX DiGraph to remove the part after ':' from node names.
    Also updates all edges to use the new node names.
    
    Args:
        graph (nx.DiGraph): Input directed graph
        
    Returns:
        nx.DiGraph: New graph with modified node names
    """
    # Create a new graph
    new_graph = nx.DiGraph()
    
    # Create mapping from old names to new names
    name_mapping = {}
    for node in graph.nodes():
        if ':' in str(node):
            new_name = str(node).replace(":","&")
        else:
            new_name = str(node)
        name_mapping[node] = new_name
    
    # Add nodes with new names and preserve node attributes
    for old_node, new_node in name_mapping.items():
        node_attrs = graph.nodes[old_node] if graph.nodes[old_node] else {}

        new_graph.add_node(new_node, **node_attrs)
    
    # Add edges with new node names and preserve edge attributes
    for u, v in graph.edges():
        new_u = name_mapping[u]
        new_v = name_mapping[v]
        edge_attrs = graph.edges[u, v] if graph.edges[u, v] else {}

        new_graph.add_edge(new_u, new_v, **edge_attrs)
    
    return new_graph



from collections import defaultdict, deque

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Create a hierarchical tree layout.
    
    Parameters:
    - G: networkx graph (should be a tree)
    - root: root node (if None, finds one automatically)
    - width: horizontal space allocated for each level
    - vert_gap: gap between levels
    - vert_loc: vertical location of root
    - xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('Cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            # Find root (node with no predecessors)
            root = [n for n, d in G.in_degree() if d == 0]
            if not root:
                root = list(G.nodes())[0]  # fallback
            else:
                root = root[0]
        else:
            root = list(G.nodes())[0]  # pick arbitrary root for undirected

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if parent is not None:
            children.remove(parent)
        
        if not children:
            return pos
        
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                               vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def radial_tree_pos(G, root=None, scale=1):
    """
    Create a radial tree layout where nodes are positioned in concentric circles.
    """
    if root is None:
        if isinstance(G, nx.DiGraph):
            roots = [n for n, d in G.in_degree() if d == 0]
            root = roots[0] if roots else list(G.nodes())[0]
        else:
            root = list(G.nodes())[0]
    
    # BFS to assign levels
    levels = {}
    queue = deque([(root, 0)])
    visited = {root}
    
    while queue:
        node, level = queue.popleft()
        levels[node] = level
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))
    
    # Group nodes by level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    pos = {}
    import math
    
    for level, nodes in level_nodes.items():
        if level == 0:  # root
            pos[nodes[0]] = (0, 0)
        else:
            radius = level * scale
            angle_step = 2 * math.pi / len(nodes)
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos[node] = (x, y)
    
    return pos

def balanced_tree_pos(G, root=None, k=2):
    """
    Create a balanced binary/k-ary tree layout.
    Assumes the tree is roughly balanced.
    """
    
    if root is None:
        if isinstance(G, nx.DiGraph):
            G = G.reverse()
            roots = [n for n, d in G.in_degree() if d == 0]

            root = roots[-1] if roots else list(G.nodes())[0]

        else:
            # For undirected, find a good root (center-ish node)
            root = nx.center(G)[0]

    pos = {}
    
    def assign_positions(node, level, position, width, parent=None):
        pos[node] = (position, -level)
        #print(node)
        children = [n for n in G.neighbors(node) if n != parent]
        if not children:
            return
        
        child_width = width / len(children)
        start_pos = position - width/2 + child_width/2
        
        for i, child in enumerate(children):
            child_pos = start_pos + i * child_width
            assign_positions(child, level + 1, child_pos, child_width, node)
    
    assign_positions(root, 0, 0, 4.0)  # Start with width of 4
    return pos
