'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-23 00:19:33
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-03-23 00:19:35
 # @Description:
'''
import os
from typing import List, Union, Mapping, Dict, Any, Tuple
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor, FunctionExecutor
from helchriss.knowledge.symbolic import Expression,FunctionApplicationExpression, ConstantExpression, VariableExpression
from helchriss.knowledge.symbolic import LogicalAndExpression, LogicalNotExpression,LogicalOrExpression
from helchriss.knowledge.symbolic import TensorState, concat_states
from helchriss.dsl.dsl_values import Value
from helchriss.domain import Domain
from helchriss.dsl.dsl_types import AnyType
from .unification import ReductiveUnifier, LocalFrame


# this is the for type space, basically a wrapper class
from .types import *
from dataclasses import dataclass
import contextlib
import networkx as nx



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

class ExecutorGroup(FunctionExecutor):
    """keeps tracks a list of CentralExecutors and can call a function by name and domain name"""
    def __init__(self, domains : List[Union[CentralExecutor, Domain]], concept_dim = 128):
        super().__init__(None)
        self._gather_format = "{}:{}"

        self._type_registry = {}
        self.executors_group = nn.ModuleList([])

        self.input_types = {}
        self.output_type = {}
        self.reserved = {"boolean"}

        for domain_executor in domains:
            if isinstance(domain_executor, FunctionExecutor):
                executor = domain_executor
            elif isinstance(domain_executor, Domain):
                executor = FunctionExecutor(Domain)
            else: raise Exception("input is not a Domain or FunctionExecutor")
            self.executors_group.append(executor)
            for func in executor.function_output_type:
                domain_name = executor.domain.domain_name
    
                func_name = self.format(func, domain_name)
                func_intypes = executor.function_input_types[func]

                args_types = []
                for func_intype in func_intypes:
                    if func_intype not in self.reserved:
                        intype_name = self.format(func_intype, domain_name)
                        in_type = executor.types[func_intype].replace("'","")
                        args_types.append(TypeBase(in_type, intype_name))
                    else: args_types.append(TypeBase("vector[float,[1]]", func_intype))
                self.input_types[func_name] = args_types

                
                func_otype = executor.function_output_type[func]
                if func_otype not in self.reserved:
                    otype_name = self.format(func_otype, domain_name)
                    otype = executor.types[func_otype].replace("'","")
                    self.output_type[func_name] = TypeBase(otype,otype_name)
                else: self.output_type[func_name] = TypeBase("vector[float,[1]]", func_otype)


            self.update_type_registry(domain_executor)
            #self.update_function_registry(domain_executor)

    def save_ckpt(self, ckpt_path):
        if not os.path.exists(f"{ckpt_path}/frames"): os.makedirs(f"{ckpt_path}/domains")
        for executor in self.executors_group:

            torch.save(executor.state_dict(), f"{ckpt_path}/domains/{executor.domain.domain_name}.pth")
        return 0

    def load_ckpt(self, ckpt_path):
        for executor in self.executors_group:
            try:
                executor.load_state_dict(torch.load(f"{ckpt_path}/domains/{executor.domain.domain_name}.pth"))
            except:
                pass
        return 0

    @property
    def gather_format(self): return self._gather_format
    
    @property
    def types(self): return self._type_registry

    def function_output_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")
        for executor in self.executors_group:

            if executor.domain.domain_name == domain:
                return self.output_type[self.format(func_name, domain)]
        assert 0, f"didn't find type {func_name} in the domain {domain}"
    
    def function_input_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")

        for executor in self.executors_group:

            if executor.domain.domain_name == domain:
                return self.input_types[self.format(func_name, domain)]
        assert 0, f"didn't find type {func_name}"

    def format(self, name : str, domain : str) -> str: return self._gather_format.format(name, domain)

    def update_type_registry(self, executor : FunctionExecutor):
        domain_name = executor.domain.domain_name
        for type, vtype in executor.types.items():
            self._type_registry[self.format(type, domain_name)] = vtype.replace("'","")

    def infer_type(self, type_name):
        """use to infer if a type does not need the domain as the postfix"""
        return
    
    def infer_domain(self, func_name):
        """infer the domain"""
        for executor in self.executors_group:
            if func_name in executor._function_registry: return executor.domain.domain_name
        return False

    def execute(self, func : str, args : List[Value], domain = None) -> Value:
        if ":" in func: # handle the case for the reference of domain is ambiguous
            func_name, domain_name = func.split(":")
        else:
            domain = self.infer_domain(func)
            if domain: domian_name = domain
            func_name = func
        args = [arg.value for arg in args]
        for executor in self.executors_group:
            if executor.domain.domain_name == domain_name: # find the correct executor
                func = executor._function_registry[func_name]
                return func(*args)

        assert False, "cannot found the function implementation"



def clamp_list(values, min_val=0.3, max_val=1.0):
    return [max(min(v, max_val), min_val) for v in values]

class ReductiveExecutor(FunctionExecutor):
    """this is some kind of wrap out of an executor, equipped with a reductive graph"""
    def __init__(self, executor):
        super().__init__(None)
        self.base_executor : FunctionExecutor = executor
        self.reduce_unifier : ReductiveUnifier = ReductiveUnifier() # use this structur to store the caster and hierarchies
        self._gather_format = "{}:{}"

        """maintain the evaluation graph just to visualize and sanity check"""
        self.eval_graph : nx.DiGraph = nx.DiGraph()
        self.node_count = {}
        self.node_outputs = []
        self.node_inputs = []
        self.record = 1


    def save_ckpt(self, ckpt_path = "tmp.ckpt"):
        self.base_executor.save_ckpt(ckpt_path)
        self.reduce_unifier.save_ckpt(ckpt_path)
        return 0

    def load_ckpt(self, ckpt_path = "tmp.ckpt"):
        self.base_executor.load_ckpt(ckpt_path)
        self.reduce_unifier.load_ckpt(ckpt_path)
        return self

    def gather_format(self, name, domain):
        return self._gather_format.format(name, domain)

    def init_graph(self):
        self.eval_graph = nx.DiGraph()
        self.node_count = {}
        self.node_outputs = []
        self.node_inputs = []
        self.record = 1
    
    def infer_reductions(self, expr : Expression, reducer = None, verbose = 0) -> List[Tuple[str, List[TypeBase], List[TypeBase]]]:
        """given an expression, use the unifer to infer if there exist casting of types or change in the local frames"""
        metaphor_exprs = []
        def dfs(expr : Expression):
            if isinstance(expr, FunctionApplicationExpression):
                func_name = expr.func.name
                output_type = self.base_executor.function_output_type(func_name)

                source_arg_types : List[TypeBase] = [dfs(arg)[0] for arg in expr.args] # A List of Types
                target_arg_types : List[TypeBase] = self.function_input_type(func_name)

                for i,tp in enumerate(target_arg_types):
                    if reducer is None:
                        if source_arg_types[i].alias != target_arg_types[i].alias:
                            metaphor_exprs.append([func_name, target_arg_types, source_arg_types])
                            break
                    else:  #TODO: use the path search to check type consistencey
                        pass
                return [output_type, target_arg_types]

            elif isinstance(expr, ConstantExpression):
                """TODO: add the return type for constants"""
                assert isinstance(expr.const, Value)
                return expr.const
    
            return [None,None]
        dfs(expr)
        if verbose:
            from helchriss.utils import stprint
            stprint(metaphor_exprs)
        return metaphor_exprs
    
    def add_metaphors(self, metaphors : List[Tuple[str, List[TypeBase], List[TypeBase]]], caster = None):
        for metaphor in metaphors:
            target_func, target_types, source_types = metaphor

            input_type = source_types #self.function_input_type(*reduce_func.split(":"))      # actual input type for the function
            expect_type = self.function_input_type(*target_func.split(":"))     # expect input type for the function
            output_type = self.function_input_type(*target_func.split(":"))     # the output type for the target function

            """create the local frame that gathers other `source` functions to the `target` function"""
            if caster is None: caster = infer_caster(input_type, expect_type)
            cast_frame : LocalFrame = LocalFrame(target_func, expect_type, output_type, input_type, caster)
            
            reduce_hypothesis = self.gather_functions(source_types)
            for reduce_func in reduce_hypothesis:
                cast_frame.add_source_caster(reduce_func, 0.0) # init the reduction `g`->`f` weight logits 0.0

            frame_name = target_func + str(hash((tuple(input_type) + tuple(expect_type) + tuple(output_type))))
            


            self.reduce_unifier.add_function_frame(frame_name, cast_frame) # multiple frame lead to the same procedure

        return 1
    
    def gather_functions(self, input_type : List[TypeBase], output_type : TypeBase = None) -> List[str]:
        output_funcs = []
        for func in self.base_executor.input_types:
            tgt_input_type = self.function_input_type(func)
            type_check = len(tgt_input_type) == len(input_type) and sum([tgt_input_type[i].alias != input_type[i].alias for i in range(len(input_type))]) == 0
            if type_check:
                if output_type is not None:
                    if output_type == self.function_output_type(func) == output_type:
                        output_funcs.append(func)
                else:
                    output_funcs.append(func)
        return output_funcs

    @property
    def types(self): return self.base_executor.types

    @property
    def functions(self): return self.base_executor.functions

    def function_output_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")
        return self.base_executor.function_output_type(func_name, domain)

    def function_input_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")
        return self.base_executor.function_input_type(func_name, domain)
    
    def display(self, fname = None):
        G = self.eval_graph
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout

        H = G.copy()
        for n, d in H.nodes(data=True):
            d.pop('inputs', None)  # remove 'weight' if it exists
            d.pop('output', None)  # remove 'weight' if it exists
        for u, v, d in H.edges(data=True):
            d.pop('output', None)  # remove 'weight' if it exists

        pos = graphviz_layout(H, prog='dot')  # Layout for the nodes
        edge_colors = [d.get('color', '#ffffff') for (u, v, d) in G.edges(data=True)]
        node_colors = [d.get('color', '#ffffff') for n, d in G.nodes(data=True)]

        node_weights = [float(data['weight']) for node, data in G.nodes(data=True)]
        plt.figure("redutive execution",figsize=(8, 6), frameon = False)

        for node, color, alpha in zip(G.nodes(), node_colors, node_weights):
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=2500, alpha=alpha )
        #nx.draw_networkx_labels(G, pos, font_color="white", font_size = 9)
        nx.draw_networkx_edges(G, pos, edge_color= edge_colors , arrows=True, arrowsize=10, arrowstyle="-|>")
        
        for node, (x, y) in pos.items():
            label = f"{float(G.nodes[node]['weight']):.2f}"
            d = G.nodes[node]
            node_color = G.nodes[node]["text_color"] if 'text_color' in G.nodes[node]  else "white"
            if isinstance(d['output'].vtype, TypeBase): out_label = f"V:{(d['output'].value)}\nTp: {d['output'].vtype.alias.split(':')[0]}"
            else: out_label = f"V:{(d['output'].value)}\nTp: {d['output'].vtype}"
            plt.text(x + -3., y + 4.95, label, fontsize=8, color='white', va='center' )
            plt.text(x + -3., y - 5.95, out_label, fontsize=5, color=node_color, va='center' )
            plt.text(x + -3., y, node, fontsize=9, color=node_color, va='center' )


        edge_labels = {(u, v): f"{float(d['weight']):.2f}" if 'weight' in d else '' for u, v, d in G.edges(data=True)}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        for (u, v, d) in G.edges(data=True):
            (x1, y1) = pos[u]
            (x2, y2) = pos[v]
            if 'output' not in d: continue
            try: label = f"V:{float(d['output'].value):.2f} Tp: {d['output'].vtype.alias.split(':')[0]}"
            except: label = f"V:{d['output'].value} Tp: {d['output'].vtype.alias.split(':')[0]}"
            xm, ym = (x1 + x2) / 2,  (y1 + y2) / 2
            bias_x, bias_y = 0.3,  -0.1
            plt.text(xm + bias_x, ym + bias_y, label, fontsize=9, color='#3a5f7d', ha='center', va='center' )
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        ax = plt.gca()
        ax.set_facecolor('none')  # Transparent background
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        if fname is not None:
            plt.savefig(f"{fname}.png")
        plt.show()
        return

    def evaluate(self, expression, grounding):
        self.init_graph()
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)

        grounding = grounding if self._grounding is not None else grounding

        with self.with_grounding(grounding):
            outputs, out_name = self._evaluate(expression)
            self.eval_graph.add_node("outputs", weight = 1.0, inputs = outputs, color = "#0d0d0d", output = outputs)
            self.eval_graph.add_edge(out_name, "outputs", output = outputs, color = "#0d0d0d")
            
            return outputs
    
    @property
    def reserved(self): return ["Id"]
    
    def process_reserved(self, func_name): 
        if func_name not in self.node_count: self.node_count[func_name] = 0
        else: self.node_count[func_name] += 1
        node_count = self.node_count[func_name]
        count_func_name = f"{func_name.split(':')[0]}_{node_count}"
        self.eval_graph.add_node(count_func_name, weight = 1.0, color = "#3a5f7d")

        if func_name.split(":")[0] in self.reserved:
            output_type : TypeBase =  AnyType#self.base_executor.function_input_type(func_name)
            expect_type : List[TypeBase] = [AnyType]
        else:
            output_type : TypeBase       = self.base_executor.function_output_type(func_name)
            expect_type : List[TypeBase] = self.base_executor.function_input_type(func_name)

        return expect_type, output_type, count_func_name ,node_count
    
    def process_measure(self, func_name : str, args : List[Value]) -> Value:
        if 0 and func_name.split(":")[0] in self.reserved:
            return args
        else:
            measure : Value = self.base_executor.execute(func_name, args)
        return measure

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
            expect_type, output_type, count_func_name ,node_count = self.process_reserved(func_name)

            # recusive call self.evaluate(arg) to evaluate the args in the subtree
            args : List[Value] = []
            for arg in expr.args:
                arg_value, arg_name = self._evaluate(arg) # A List of Values
                args.append(arg_value)
                self.eval_graph.add_edge(arg_name, count_func_name, output = arg_value, color = "#0d0d0d")
            

            
            # cast the input args to the expected type (traverse the cast DAG)
            print(func_name, args)
            cast_args, weight, cast_info = self.reduce_unifier.type_cast(args, expect_type)


            # collect the (f,t) reductions as they are a pair of natual type
            reduce_funcs, reduce_graph = self.reduce_unifier.reduce_args(func_name, cast_args)

            """handle the visualization of the reduction graph"""
            reduce_nodes, reduce_edges = reduce_graph

            for node in reduce_nodes:
                if node != func_name:
                    for reduce_func in reduce_funcs:
                        if reduce_func[0] == node:
                            node_weight = reduce_func[2]
                    self.eval_graph.add_node(f"{node.split(':')[0]}_{node_count}", weight = node_weight, color = "#048393", text_color = '#0d0d0d')
                else:
                    for reduce_func in reduce_funcs:
                        if reduce_func[0] == node:
                            node_weight = reduce_func[2]
                    self.eval_graph.nodes[count_func_name]["weight"] = node_weight

            for edge in reduce_edges:
    
                src_node = f"{edge[0].split(':')[0]}_{node_count}"
                tgt_node = f"{edge[1].split(':')[0]}_{node_count}"

                if src_node != func_name:
                    self.eval_graph.add_edge(f"{src_node}", tgt_node, weight = float(edge[2].detach()[0]), color = "#048393")

            # for the identity part, just don't calculate the expecation over

            if func_name.split(":")[0] in self.reserved:
                output_type = cast_args[0]

                return Value(output_type, args), count_func_name
            
            expect_output = 0.0
            for reduce_func in reduce_funcs:
                rfunc_name, reduce_args, weight = reduce_func
                measure = self.process_measure(rfunc_name, reduce_args)

                expect_output += measure.value * weight

                if func_name != rfunc_name: self.eval_graph.nodes[f"{rfunc_name.split(':')[0]}_{node_count}"]["output"] = measure
                else: self.eval_graph.nodes[count_func_name]["output"] = measure                    


            return Value(output_type, expect_output), count_func_name

        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const
        elif isinstance(expr, VariableExpression):

            assert isinstance(expr.name, Value)
            return expr.const
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')
        
import json
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